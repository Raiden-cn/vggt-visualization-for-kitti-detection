import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import time
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def read_kitti_calib(calib_filepath):
    if not os.path.exists(calib_filepath):
        return None
    with open(calib_filepath, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = list(map(float, line.strip().split()[1:]))
                return np.array(vals).reshape(3, 4)
    return None

def parse_kitti_label_file(filepath):
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

def save_ply_with_conf(points, colors, conf, filename):
    """Save 3D point cloud with RGB colors and Confidence values into binary PLY file."""
    valid_mask = np.isfinite(points).all(axis=-1)
    pts = points[valid_mask].astype(np.float32)
    cls = colors[valid_mask].astype(np.uint8)
    cfs = conf[valid_mask].astype(np.float32)

    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nend_header\n"

    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('confidence', '<f4')])
    ver_array = np.empty(len(pts), dtype=dt)
    ver_array['x'] = pts[:, 0]
    ver_array['y'] = pts[:, 1]
    ver_array['z'] = pts[:, 2]
    ver_array['red'] = cls[:, 0]
    ver_array['green'] = cls[:, 1]
    ver_array['blue'] = cls[:, 2]
    ver_array['confidence'] = cfs

    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        ver_array.tofile(f)

def main():
    output_dir = "/root/projcet_zgx/vggt/output_kitti_val_3d"
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)

    val_txt = "/root/dataset/kitti/ImageSets/val.txt"
    pred_dir = "/root/projcet_zgx/vggt/kitti_pred/data"
    gt_dir = "/root/dataset/kitti/training/label_2"
    img_dir = "/root/dataset/kitti/training/image_2"
    calib_dir = "/root/dataset/kitti/training/calib"

    with open(val_txt, "r") as f:
        val_ids = [line.strip() for line in f if line.strip()][:1000]

    print(f"==========================================================================")
    print(f"🚀 Batch Processing First {len(val_ids)} KITTI Val Scenes to PLY Files (with Confidence)")
    print(f"==========================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    print(f"Device: {device}, Precision Dtype: {dtype}")
    print("Loading VGGT-1B Model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    start_time = time.time()

    for idx, vid in enumerate(val_ids, 1):
        ply_file = os.path.join(ply_dir, f"kitti_val_{vid}.ply")
        p_img = os.path.join(img_dir, f"{vid}.png")
        p_pred = os.path.join(pred_dir, f"{vid}.txt")
        p_gt = os.path.join(gt_dir, f"{vid}.txt")
        p_calib = os.path.join(calib_dir, f"{vid}.txt")

        if not os.path.exists(p_img):
            continue

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

        images_tensor = load_and_preprocess_images([p_img]).to(device)
        images_batch = images_tensor[None]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_batch)

        w_pts = predictions["world_points"].squeeze(0).squeeze(0).cpu().numpy() # (H, W, 3)
        w_conf = predictions["world_points_conf"].squeeze(0).squeeze(0).cpu().numpy() # (H, W)
        img_np = predictions["images"].squeeze(0).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        H, W, _ = w_pts.shape

        cls_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        pred_boxes = parse_kitti_label_file(p_pred)
        gt_boxes = parse_kitti_label_file(p_gt)

        depth_vggt = w_pts[..., 2]
        orig_w, orig_h = 1242.0, 375.0
        
        anchor_boxes = pred_boxes if len(pred_boxes) > 0 else gt_boxes
        scale_list = []

        for box in anchor_boxes:
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

        global_scale = float(np.median(scale_list)) if len(scale_list) > 0 else 16.0

        depth_real = depth_vggt * global_scale
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        
        fx_s = fx * (W / orig_w)
        fy_s = fy * (H / orig_h)
        cx_s = cx * (W / orig_w)
        cy_s = cy * (H / orig_h)

        x_cam = (u_grid - cx_s) * depth_real / fx_s
        y_cam = (v_grid - cy_s) * depth_real / fy_s
        z_cam = depth_real

        metric_pts = np.stack([x_cam, y_cam, z_cam], axis=-1)
        save_ply_with_conf(metric_pts.reshape(-1, 3), cls_uint8.reshape(-1, 3), w_conf.reshape(-1), ply_file)

        if idx % 10 == 0 or idx == len(val_ids):
            elapsed = time.time() - start_time
            print(f"  [{idx}/{len(val_ids)}] Processed & Saved PLY with Confidence for scene {vid} | Elapsed: {elapsed:.1f}s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n==========================================================================")
    print(f"🎉 Completed offline PLY with Confidence generation for 1000 scenes!")
    print(f"📁 PLY Files stored in: {ply_dir}")
    print(f"==========================================================================")

if __name__ == "__main__":
    main()
