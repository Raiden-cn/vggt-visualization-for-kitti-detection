import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import time
import numpy as np
import torch
import viser
import viser.transforms as viser_tf
import trimesh

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

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
            if len(parts) < 14:
                continue
            obj_type = parts[0]
            if obj_type.lower() == "dontcare":
                continue
            
            # 2D Bounding Box [u1, v1, u2, v2]
            u1, v1, u2, v2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            # 3D Dimensions: height, width, length
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            # 3D Location in Camera Coordinates (Bottom Center): x, y, z
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            # Yaw Rotation around Y-axis
            ry = float(parts[14])
            # Confidence score if present
            score = float(parts[15]) if len(parts) > 15 else 1.0

            boxes.append({
                "type": obj_type,
                "bbox_2d": [u1, v1, u2, v2],
                "h": h, "w": w, "l": l,
                "x": x, "y": y, "z": z,
                "ry": ry,
                "score": score
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

    # 3D corners in local box frame (bottom center is origin [0, 0, 0])
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # Top corners are y - h
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners_local = np.vstack([x_corners, y_corners, z_corners])
    corners_cam = (R_y @ corners_local).T + np.array([x, y, z])
    
    # 3D Center of the box (middle center)
    center_cam = np.array([x, y - h / 2.0, z])

    return corners_cam, center_cam, R_y

def save_ply(points, colors, filename):
    """Save 3D point cloud into PLY file."""
    valid_mask = np.isfinite(points).all(axis=-1)
    pts = points[valid_mask]
    cls = colors[valid_mask]
    cloud = trimesh.PointCloud(vertices=pts, colors=cls)
    cloud.export(filename)
    print(f"--> Saved Metric 3D Point Cloud PLY: {filename}")

def main():
    port = 12342
    output_dir = "/root/projcet_zgx/vggt/output_kitti_val_3d"
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)

    val_txt = "/root/dataset/kitti/ImageSets/val.txt"
    pred_dir = "/root/projcet_zgx/vggt/kitti_pred/data"
    gt_dir = "/root/dataset/kitti/training/label_2"
    img_dir = "/root/dataset/kitti/training/image_2"
    calib_dir = "/root/dataset/kitti/training/calib"

    with open(val_txt, "r") as f:
        val_ids = [line.strip() for line in f if line.strip()][:100]

    img_paths = [os.path.join(img_dir, f"{vid}.png") for vid in val_ids]
    pred_paths = [os.path.join(pred_dir, f"{vid}.txt") for vid in val_ids]
    gt_paths = [os.path.join(gt_dir, f"{vid}.txt") for vid in val_ids]
    calib_paths = [os.path.join(calib_dir, f"{vid}.txt") for vid in val_ids]

    print("==========================================================================")
    print(f"🚀 KITTI Val 3D Reconstruction | GT (Red) vs Prediction (Green) Overlay")
    all_categories = set()
    for idx, (vid, p_img, p_pred, p_gt) in enumerate(zip(val_ids, img_paths, pred_paths, gt_paths), 1):
        p_boxes = parse_kitti_label_file(p_pred)
        g_boxes = parse_kitti_label_file(p_gt)
        print(f"  [{idx:02d}] Val ID: {vid} | Image: {os.path.basename(p_img)} | GT Boxes (Red): {len(g_boxes)} | Pred Boxes (Green): {len(p_boxes)}")
        for b in p_boxes + g_boxes:
            all_categories.add(b["type"])
    print("==========================================================================")

    category_list = sorted(list(all_categories))
    print(f"Detected Categories in Dataset: {category_list}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    print(f"Device: {device}, Precision Dtype: {dtype}")
    print("Loading VGGT-1B Pretrained Model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(device).eval()

    scene_data = []

    print("\nRunning VGGT 3D Reconstruction & Anchor-Based Scale Alignment...")
    for idx, (vid, p_img, p_pred, p_gt, p_calib) in enumerate(zip(val_ids, img_paths, pred_paths, gt_paths, calib_paths)):
        img_name = f"{vid}.png"
        print(f"\n  [{idx+1}/{len(val_ids)}] Processing Val Scene {vid}...")

        # 1. Load Calibration & Intrinsics
        P2 = read_kitti_calib(p_calib)
        if P2 is None:
            P2 = np.array([
                [721.5377, 0.0, 609.5593, 44.85728],
                [0.0, 721.5377, 172.8540, 0.2163791],
                [0.0, 0.0, 1.0, 0.002745884]
            ])
        
        K_kitti = P2[:3, :3]
        fx, fy = K_kitti[0, 0], K_kitti[1, 1]
        cx, cy = K_kitti[0, 2], K_kitti[1, 2]

        # 2. Model Inference for Depth Map Prediction
        images_tensor = load_and_preprocess_images([p_img]).to(device) # (1, 3, H, W)
        images_batch = images_tensor[None] # (1, 1, 3, H, W)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_batch)

        w_pts = predictions["world_points"].squeeze(0).squeeze(0).cpu().numpy() # (H, W, 3)
        w_conf = predictions["world_points_conf"].squeeze(0).squeeze(0).cpu().numpy() # (H, W)
        img_np = predictions["images"].squeeze(0).squeeze(0).cpu().numpy().transpose(1, 2, 0) # (H, W, 3)
        H, W, _ = w_pts.shape

        cls_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        # Parse Predictions and Ground Truth labels
        pred_boxes = parse_kitti_label_file(p_pred)
        gt_boxes = parse_kitti_label_file(p_gt)

        # 3. Anchor-Based Depth Scale Extraction
        depth_vggt = w_pts[..., 2]  # (H, W)
        orig_w, orig_h = 1242.0, 375.0
        
        anchor_boxes = pred_boxes if len(pred_boxes) > 0 else gt_boxes
        scale_list = []

        for b_idx, box in enumerate(anchor_boxes):
            u1, v1, u2, v2 = box["bbox_2d"]
            u1_s = int(np.clip(u1 * (W / orig_w), 0, W - 1))
            u2_s = int(np.clip(u2 * (W / orig_w), u1_s + 1, W))
            v1_s = int(np.clip(v1 * (H / orig_h), 0, H - 1))
            v2_s = int(np.clip(v2 * (H / orig_h), v1_s + 1, H))

            crop_z = depth_vggt[v1_s:v2_s, u1_s:u2_s]
            valid_crop_z = crop_z[crop_z > 0.1]
            d_vggt_pred = float(np.median(valid_crop_z)) if len(valid_crop_z) > 0 else 1.0

            z_kitti = box["z"]
            scale_i = z_kitti / d_vggt_pred
            scale_list.append(scale_i)
            print(f"    Anchor Box [{box['type']}]: Z_kitti={z_kitti:.2f}m, d_vggt_pred={d_vggt_pred:.3f}m -> scale_i={scale_i:.3f}")

        if len(scale_list) > 0:
            global_scale = float(np.median(scale_list))
        else:
            global_scale = 16.0  # Fallback scale

        print(f"  --> Global Physical Scale (Median): {global_scale:.4f}")

        # 4. Restore Absolute Metric Depth Map & Unproject via KITTI Intrinsics
        depth_real = depth_vggt * global_scale
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        
        fx_s = fx * (W / orig_w)
        fy_s = fy * (H / orig_h)
        cx_s = cx * (W / orig_w)
        cy_s = cy * (H / orig_h)

        x_cam = (u_grid - cx_s) * depth_real / fx_s
        y_cam = (v_grid - cy_s) * depth_real / fy_s
        z_cam = depth_real

        metric_pts = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

        # Export Absolute Metric PLY File
        ply_file = os.path.join(ply_dir, f"kitti_val_{vid}.ply")
        save_ply(metric_pts.reshape(-1, 3), cls_uint8.reshape(-1, 3), ply_file)

        scene_data.append({
            "val_id": vid,
            "img_name": img_name,
            "metric_points": metric_pts,
            "conf": w_conf,
            "colors": cls_uint8,
            "global_scale": global_scale,
            "K_scaled": (fx_s, fy_s, cx_s, cy_s),
            "H": H, "W": W,
            "pred_boxes": pred_boxes,
            "gt_boxes": gt_boxes,
            "ply_path": ply_file
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n==========================================================================")
    print(f"🎉 {len(scene_data)} Val Scenes Processed with Text Label & Category Filter Controls!")
    print(f"📁 PLY Files Saved in: {ply_dir}")
    print(f"🌐 Launching Interactive Viser 3D Web Server on http://0.0.0.0:{port}")
    print(f"==========================================================================\n")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # GUI Controls
    scene_options = [f"Val [{data['val_id']}] (GT:{len(data['gt_boxes'])}, Pred:{len(data['pred_boxes'])})" for data in scene_data]
    gui_scene_select = server.gui.add_dropdown("Select Val Scene", options=scene_options, initial_value=scene_options[0])
    
    gui_btn_prev = server.gui.add_button("◀ Previous Scene")
    gui_btn_next = server.gui.add_button("Next Scene ▶")
    gui_play = server.gui.add_checkbox("Auto Cycle Scenes", initial_value=False)
    
    # 1. Strict Color Toggles: GT (Green) & Prediction (Red)
    gui_show_gt = server.gui.add_checkbox("Show GT 3D Boxes (Green)", initial_value=True)
    gui_show_pred = server.gui.add_checkbox("Show Prediction 3D Boxes (Red)", initial_value=True)
    
    # 2. Text Label Display Toggle ("GT: Car", "Pred: Car 0.98")
    gui_show_labels = server.gui.add_checkbox("Show 3D Text Labels", initial_value=True)

    # 3. Category Filter Checkboxes (Car, Pedestrian, Cyclist, Truck, Van, Misc)
    gui_category_toggles = {}
    with server.gui.add_folder("Category Filter (类别筛选)"):
        for cat in category_list:
            gui_category_toggles[cat] = server.gui.add_checkbox(f"Show {cat}", initial_value=True)

    # Camera & Point Cloud Controls
    gui_show_camera = server.gui.add_checkbox("Show Camera Frustum", initial_value=True)
    gui_conf = server.gui.add_slider("Confidence Filter (%)", min=0.0, max=90.0, step=5.0, initial_value=30.0)
    gui_point_size = server.gui.add_slider("Point Size (Meters)", min=0.02, max=0.30, step=0.01, initial_value=0.08)

    # Info Display
    gui_info = server.gui.add_markdown(f"**Current Val Scene**: `{scene_data[0]['val_id']}`")

    # State variables
    current_scene = 0  # 0-indexed
    point_cloud_node = None
    camera_nodes = []
    box_nodes = []

    def render_boxes_for_scene(box_list, box_color, prefix_name, show_labels):
        """Helper function to render a set of 3D bounding boxes with category filter & label toggle."""
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for b_idx, box in enumerate(box_list):
            obj_type = box["type"]
            
            # Check Category Filter Toggle
            cat_toggle = gui_category_toggles.get(obj_type)
            if cat_toggle is not None and not cat_toggle.value:
                continue  # Skip if category is unchecked

            corners_cam, center_cam, R_y = build_3d_box_corners(box)
            h, w, l = box["h"], box["w"], box["l"]
            score = box["score"]

            line_segments = np.array([[corners_cam[p1], corners_cam[p2]] for p1, p2 in edges])
            
            # 12 Wireframe Edge Lines
            line_node = server.scene.add_line_segments(
                name=f"/{prefix_name}_{b_idx}_lines",
                points=line_segments,
                colors=box_color,
                line_width=3.5
            )
            box_nodes.append(line_node)

            # Semi-transparent Mesh Box
            mesh_box_node = server.scene.add_box(
                name=f"/{prefix_name}_{b_idx}_mesh",
                dimensions=(l, h, w),
                position=center_cam,
                wxyz=viser_tf.SO3.from_matrix(R_y).wxyz,
                color=box_color,
                opacity=0.20
            )
            box_nodes.append(mesh_box_node)

            # 3D Floating Text Label (Toggled by gui_show_labels)
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
        
        s_data = scene_data[current_scene]
        vid = s_data["val_id"]
        show_gt = gui_show_gt.value
        show_pred = gui_show_pred.value
        show_labels = gui_show_labels.value
        show_camera = gui_show_camera.value
        conf_pct = gui_conf.value
        pt_size = gui_point_size.value
        H, W = s_data["H"], s_data["W"]

        # Active category filters
        active_cats = [cat for cat, cb in gui_category_toggles.items() if cb.value]

        # Update Info text
        gui_info.content = f"### 📦 KITTI Val Scene `{vid}` (Val Index: {current_scene + 1}/{len(scene_data)})\n" \
                           f"- 🟢 **Ground Truth (GT) Boxes**: `{len(s_data['gt_boxes'])}` boxes\n" \
                           f"- 🔴 **Prediction Boxes**: `{len(s_data['pred_boxes'])}` boxes\n" \
                           f"- 🏷️ **Text Labels**: `{'ON' if show_labels else 'OFF'}` | 🔍 **Active Classes**: `{', '.join(active_cats)}`\n" \
                           f"- **Anchor Physical Scale**: `{s_data['global_scale']:.4f}`\n" \
                           f"- **PLY File**: `{os.path.basename(s_data['ply_path'])}`"

        # Clear existing camera and box nodes
        for node in camera_nodes:
            node.remove()
        camera_nodes.clear()

        for node in box_nodes:
            node.remove()
        box_nodes.clear()

        # Get metric 3D point cloud data
        pts = s_data["metric_points"]
        cls = s_data["colors"]
        cfs = s_data["conf"]

        conf_thresh = np.percentile(cfs, conf_pct) if len(cfs) > 0 else 0
        mask = cfs >= conf_thresh

        filtered_pts = pts.reshape(-1, 3)[mask.reshape(-1)]
        filtered_colors = cls.reshape(-1, 3)[mask.reshape(-1)]

        # Show camera frustum at camera origin (0, 0, 0)
        if show_camera:
            fx_s, fy_s, cx_s, cy_s = s_data["K_scaled"]
            fov = 2 * np.arctan(W / (2 * fx_s))

            cam_node = server.scene.add_camera_frustum(
                name=f"/camera_scene_{vid}",
                fov=fov,
                aspect=W / H,
                scale=1.5,
                image=cls,
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

    @gui_btn_prev.on_click
    def _(_):
        nonlocal current_scene
        current_scene = (current_scene - 1) % len(scene_data)
        gui_scene_select.value = scene_options[current_scene]

    @gui_btn_next.on_click
    def _(_):
        nonlocal current_scene
        current_scene = (current_scene + 1) % len(scene_data)
        gui_scene_select.value = scene_options[current_scene]

    # Initial Render
    update_visualization()

    print(f"Viser 3D Web App with Text Label & Category Filter ready on http://0.0.0.0:{port}")

    # Auto cycle loop
    while True:
        time.sleep(1.5)
        if gui_play.value:
            current_scene = (current_scene + 1) % len(scene_data)
            gui_scene_select.value = scene_options[current_scene]

if __name__ == "__main__":
    main()
