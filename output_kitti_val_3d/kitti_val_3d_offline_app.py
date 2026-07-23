import os
import time
import numpy as np
import trimesh
import viser
import viser.transforms as viser_tf

def read_kitti_calib(calib_filepath):
    """Read P2 projection matrix from KITTI calibration file."""
    if not os.path.exists(calib_filepath):
        return None
    with open(calib_filepath, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = list(map(float, line.strip().split()[1:]))
                return np.array(vals).reshape(3, 4)
    return None

def parse_kitti_label_file(filepath):
    """Parse KITTI 3D object detection label file (GT or Prediction)."""
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14 or parts[0].lower() == "dontcare":
                continue
            boxes.append({
                "type": parts[0],
                "bbox_2d": [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                "h": float(parts[8]), "w": float(parts[9]), "l": float(parts[10]),
                "x": float(parts[11]), "y": float(parts[12]), "z": float(parts[13]),
                "ry": float(parts[14]),
                "score": float(parts[15]) if len(parts) > 15 else 1.0
            })
    return boxes

def build_3d_box_corners(box):
    """
    Compute 8 corners of a KITTI 3D Bounding Box in camera coordinates.
    Note: (x, y, z) in KITTI is the bottom-center of the 3D bounding box.
    """
    h, w, l = box["h"], box["w"], box["l"]
    x, y, z = box["x"], box["y"], box["z"]
    ry = box["ry"]

    R_y = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners_local = np.vstack([x_corners, y_corners, z_corners])
    corners_cam = (R_y @ corners_local).T + np.array([x, y, z])
    center_cam = np.array([x, y - h / 2.0, z])

    return corners_cam, center_cam, R_y

def load_ply_with_conf(filepath):
    """Load PLY file with custom confidence attribute if available, or fallback to trimesh."""
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('confidence', '<f4')])
    try:
        with open(filepath, 'rb') as f:
            header = ''
            is_conf_ply = False
            while True:
                line = f.readline().decode('ascii', errors='ignore')
                header += line
                if 'confidence' in line:
                    is_conf_ply = True
                if line.startswith('end_header'):
                    break
            
            if is_conf_ply:
                data = np.fromfile(f, dtype=dt)
                pts = np.stack([data['x'], data['y'], data['z']], axis=-1)
                cls = np.stack([data['red'], data['green'], data['blue']], axis=-1)
                conf = data['confidence']
                return pts, cls, conf
    except Exception:
        pass

    # Fallback to standard trimesh PLY loader
    cloud = trimesh.load(filepath)
    pts = np.asarray(cloud.vertices, dtype=np.float32)
    cls = np.asarray(cloud.colors[:, :3], dtype=np.uint8) if cloud.colors is not None else np.ones_like(pts, dtype=np.uint8) * 200
    conf = np.ones(len(pts), dtype=np.float32)
    return pts, cls, conf

def main():
    port = 12342
    output_dir = "/root/projcet_zgx/vggt/output_kitti_val_3d"
    ply_dir = os.path.join(output_dir, "ply")

    val_txt = "/root/dataset/kitti/ImageSets/val.txt"
    pred_dir = "/root/projcet_zgx/vggt/kitti_pred/data"
    gt_dir = "/root/dataset/kitti/training/label_2"
    calib_dir = "/root/dataset/kitti/training/calib"

    with open(val_txt, "r") as f:
        val_ids = [line.strip() for line in f if line.strip()][:1000]

    # Pre-index valid scenes where PLY file exists
    valid_scenes = []
    all_categories = set()

    print("==========================================================================")
    print(f"🚀 Indexing Offline PLY Scenes with Confidence Control...")
    
    for vid in val_ids:
        ply_path = os.path.join(ply_dir, f"kitti_val_{vid}.ply")
        if not os.path.exists(ply_path):
            continue

        p_pred = os.path.join(pred_dir, f"{vid}.txt")
        p_gt = os.path.join(gt_dir, f"{vid}.txt")
        p_calib = os.path.join(calib_dir, f"{vid}.txt")

        gt_boxes = parse_kitti_label_file(p_gt)
        pred_boxes = parse_kitti_label_file(p_pred)

        for b in gt_boxes + pred_boxes:
            all_categories.add(b["type"])

        valid_scenes.append({
            "val_id": vid,
            "ply_path": ply_path,
            "p_gt": p_gt,
            "p_pred": p_pred,
            "p_calib": p_calib,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes
        })

    category_list = sorted(list(all_categories)) if len(all_categories) > 0 else ["Car", "Pedestrian", "Cyclist", "Truck", "Van"]
    print(f"Indexed {len(valid_scenes)} Offline Scenes!")
    print(f"Detected Object Categories: {category_list}")
    print("==========================================================================\n")

    if len(valid_scenes) == 0:
        print("❌ No offline PLY files found in output_kitti_val_3d/ply! Please run process_1000_offline.py first.")
        return

    # Launch Pure Offline Viser 3D Web Server
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # GUI Controls
    scene_options = [f"Val [{data['val_id']}] (GT:{len(data['gt_boxes'])}, Pred:{len(data['pred_boxes'])})" for data in valid_scenes]
    gui_scene_select = server.gui.add_dropdown("Select Val Scene", options=scene_options, initial_value=scene_options[0])
    
    gui_btn_prev = server.gui.add_button("◀ Previous Scene")
    gui_btn_next = server.gui.add_button("Next Scene ▶")
    gui_play = server.gui.add_checkbox("Auto Cycle Scenes", initial_value=False)
    
    # 1. Color Toggles: GT (Green) & Prediction (Red)
    gui_show_gt = server.gui.add_checkbox("Show GT 3D Boxes (Green)", initial_value=True)
    gui_show_pred = server.gui.add_checkbox("Show Prediction 3D Boxes (Red)", initial_value=True)
    
    # 2. Text Label Display Toggle
    gui_show_labels = server.gui.add_checkbox("Show 3D Text Labels", initial_value=True)

    # 3. Category Filter Checkboxes
    gui_category_toggles = {}
    with server.gui.add_folder("Category Filter (类别筛选)"):
        for cat in category_list:
            gui_category_toggles[cat] = server.gui.add_checkbox(f"Show {cat}", initial_value=True)

    # 4. Camera, Confidence Filter & Point Cloud Controls
    gui_show_camera = server.gui.add_checkbox("Show Camera Frustum", initial_value=True)
    gui_conf = server.gui.add_slider("Confidence Filter (%)", min=0.0, max=90.0, step=5.0, initial_value=30.0)
    gui_point_size = server.gui.add_slider("Point Size (Meters)", min=0.02, max=0.30, step=0.01, initial_value=0.08)
    gui_downsample = server.gui.add_slider("Point Subsample Ratio", min=1, max=10, step=1, initial_value=1)

    # Info Display
    gui_info = server.gui.add_markdown(f"**Current Val Scene**: `{valid_scenes[0]['val_id']}`")

    # State variables
    current_scene = 0
    point_cloud_node = None
    camera_nodes = []
    box_nodes = []
    loaded_ply_cache = {}

    def load_ply_cached(ply_path):
        if ply_path in loaded_ply_cache:
            return loaded_ply_cache[ply_path]
        pts, cls, conf = load_ply_with_conf(ply_path)
        loaded_ply_cache[ply_path] = (pts, cls, conf)
        return pts, cls, conf

    def render_boxes_for_scene(box_list, box_color, prefix_name, show_labels):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for b_idx, box in enumerate(box_list):
            obj_type = box["type"]
            cat_toggle = gui_category_toggles.get(obj_type)
            if cat_toggle is not None and not cat_toggle.value:
                continue

            corners_cam, center_cam, R_y = build_3d_box_corners(box)
            h, w, l = box["h"], box["w"], box["l"]
            score = box["score"]

            line_segments = np.array([[corners_cam[p1], corners_cam[p2]] for p1, p2 in edges])
            
            line_node = server.scene.add_line_segments(
                name=f"/{prefix_name}_{b_idx}_lines",
                points=line_segments,
                colors=box_color,
                line_width=3.5
            )
            box_nodes.append(line_node)

            mesh_box_node = server.scene.add_box(
                name=f"/{prefix_name}_{b_idx}_mesh",
                dimensions=(l, h, w),
                position=center_cam,
                wxyz=viser_tf.SO3.from_matrix(R_y).wxyz,
                color=box_color,
                opacity=0.20
            )
            box_nodes.append(mesh_box_node)

            if show_labels:
                lbl_text = f"GT: {obj_type}" if "gt" in prefix_name else f"Pred: {obj_type} {score:.2f}"
                lbl_pos = center_cam + np.array([0, h / 2.0 + (0.4 if "gt" in prefix_name else 0.1), 0])
                label_node = server.scene.add_label(
                    name=f"/{prefix_name}_{b_idx}_label",
                    text=lbl_text,
                    position=lbl_pos
                )
                box_nodes.append(label_node)

    def update_visualization():
        nonlocal point_cloud_node, camera_nodes, box_nodes, current_scene
        
        s_data = valid_scenes[current_scene]
        vid = s_data["val_id"]
        show_gt = gui_show_gt.value
        show_pred = gui_show_pred.value
        show_labels = gui_show_labels.value
        show_camera = gui_show_camera.value
        conf_pct = gui_conf.value
        pt_size = gui_point_size.value
        subsample = gui_downsample.value

        active_cats = [cat for cat, cb in gui_category_toggles.items() if cb.value]

        # Update Info Markdown
        gui_info.content = f"### ⚡ KITTI Val Scene `{vid}` (Val Index: {current_scene + 1}/{len(valid_scenes)})\n" \
                           f"- 🟢 **Ground Truth (GT) Boxes**: `{len(s_data['gt_boxes'])}` boxes\n" \
                           f"- 🔴 **Prediction Boxes**: `{len(s_data['pred_boxes'])}` boxes\n" \
                           f"- 🏷️ **Text Labels**: `{'ON' if show_labels else 'OFF'}` | 🔍 **Active Classes**: `{', '.join(active_cats)}`\n" \
                           f"- 📁 **Offline PLY**: `{os.path.basename(s_data['ply_path'])}`"

        for node in camera_nodes:
            node.remove()
        camera_nodes.clear()

        for node in box_nodes:
            node.remove()
        box_nodes.clear()

        # Load PLY point cloud & confidence directly from disk
        pts, cls, conf = load_ply_cached(s_data["ply_path"])

        # Confidence Filter Threshold Calculation
        conf_thresh = np.percentile(conf, conf_pct) if len(conf) > 0 else 0
        mask = conf >= conf_thresh

        filtered_pts = pts[mask]
        filtered_colors = cls[mask]

        if subsample > 1:
            filtered_pts = filtered_pts[::subsample]
            filtered_colors = filtered_colors[::subsample]

        # Show camera frustum at camera origin
        if show_camera:
            P2 = read_kitti_calib(s_data["p_calib"])
            if P2 is not None:
                fx = P2[0, 0]
                fov = 2 * np.arctan(1242.0 / (2 * fx))
            else:
                fov = 1.0

            cam_node = server.scene.add_camera_frustum(
                name=f"/camera_scene_{vid}",
                fov=fov,
                aspect=1242.0 / 375.0,
                scale=1.5,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            camera_nodes.append(cam_node)

        # 1. Render Ground Truth 3D Boxes (Pure GREEN [0, 230, 118])
        if show_gt:
            green_color = (0, 230, 118)
            render_boxes_for_scene(s_data["gt_boxes"], green_color, "gt_box", show_labels)

        # 2. Render Prediction 3D Boxes (Pure RED [255, 0, 0])
        if show_pred:
            red_color = (255, 0, 0)
            render_boxes_for_scene(s_data["pred_boxes"], red_color, "pred_box", show_labels)

        # Update Point Cloud Node
        if point_cloud_node is not None:
            point_cloud_node.remove()

        point_cloud_node = server.scene.add_point_cloud(
            name="/scene_pointcloud",
            points=filtered_pts,
            colors=filtered_colors,
            point_size=pt_size,
            point_shape="circle"
        )

    # Register Callbacks
    @gui_scene_select.on_update
    def _(_):
        nonlocal current_scene
        selected_text = gui_scene_select.value
        current_scene = scene_options.index(selected_text)
        update_visualization()

    @gui_show_gt.on_update
    def _(_):
        update_visualization()

    @gui_show_pred.on_update
    def _(_):
        update_visualization()

    @gui_show_labels.on_update
    def _(_):
        update_visualization()

    for cb in gui_category_toggles.values():
        @cb.on_update
        def _(_):
            update_visualization()

    @gui_show_camera.on_update
    def _(_):
        update_visualization()

    @gui_conf.on_update
    def _(_):
        update_visualization()

    @gui_point_size.on_update
    def _(_):
        update_visualization()

    @gui_downsample.on_update
    def _(_):
        update_visualization()

    @gui_btn_prev.on_click
    def _(_):
        nonlocal current_scene
        current_scene = (current_scene - 1) % len(valid_scenes)
        gui_scene_select.value = scene_options[current_scene]

    @gui_btn_next.on_click
    def _(_):
        nonlocal current_scene
        current_scene = (current_scene + 1) % len(valid_scenes)
        gui_scene_select.value = scene_options[current_scene]

    # Initial Render
    update_visualization()

    print(f"🎉 Pure Offline Viser 3D Web App ready on http://0.0.0.0:{port}!")
    print(f"Loaded {len(valid_scenes)} pre-processed 3D scenes with Confidence Control!")

    # Auto cycle loop
    while True:
        time.sleep(1.5)
        if gui_play.value:
            current_scene = (current_scene + 1) % len(valid_scenes)
            gui_scene_select.value = scene_options[current_scene]

if __name__ == "__main__":
    main()
