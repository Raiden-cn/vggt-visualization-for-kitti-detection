"""
KITTIDatasetVGGT: KITTI 3D detection dataset with VGGT-compatible preprocessing.

Key differences from MonoDETR's KITTI_Dataset:
  - Image preprocessed to VGGT size: 518x154 (width=518, height∝aspect, /14 aligned)
    (KITTI 1242x375 -> width=518 -> height=round(375*518/1242/14)*14 = 11*14=154)
  - Returns raw [0,1] float tensor (VGGT aggregator normalizes internally)
  - Calibration P2 scaled to match new image coordinates
  - Target dict matches MonoDETR format exactly (same keys)
"""

import os
import sys
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

# MonoDETR utilities
sys.path.insert(0, '/root/projcet_zgx/vggt/monodetr')
from lib.datasets.utils import angle2class, gaussian_radius, draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import (
    get_objects_from_label, Calibration,
    get_affine_transform, affine_transform,
)
from lib.datasets.kitti.pd import PhotometricDistort
# NOTE: kitti_eval_python imports are deferred to eval() to avoid numba CUDA compile issues at import time


# VGGT target input size (width=518, height=154 for KITTI 1242x375)
VGGT_W = 518
VGGT_H = 154  # round(375 * 518/1242 / 14) * 14


class KITTIDatasetVGGT(data.Dataset):
    """
    KITTI 3D Object Detection dataset adapted for VGGT backbone.

    Returns:
        img_vggt:  [3, H, W] float32 tensor in [0,1] (raw pixels, VGGT normalizes internally)
        calib_P2:  [3, 4] numpy float32 (scaled to new image coordinates)
        targets:   dict with MonoDETR-compatible GT labels
        info:      dict with metadata
    """

    def __init__(self, split, cfg):
        # Basic config
        self.root_dir   = cfg.get('root_dir')
        self.split      = split
        self.num_classes = 3
        self.max_objs   = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id     = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}

        # VGGT input resolution
        self.vggt_w = cfg.get('vggt_w', VGGT_W)
        self.vggt_h = cfg.get('vggt_h', VGGT_H)
        self.resolution = np.array([self.vggt_w, self.vggt_h])  # W x H

        # For MonoDETR-style feature size computation (patch_size=14 for VGGT)
        self.downsample = 14

        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist     = cfg.get('writelist', ['Car'])
        self.bbox2d_type   = cfg.get('bbox2d_type', 'anno')
        self.meanshape     = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare  = cfg.get('use_dontcare', False)
        self.clip_2d       = cfg.get('clip_2d', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # Split file
        assert self.split in ['train', 'val', 'trainval', 'test']
        split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(split_file).readlines()]

        # Paths
        data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(data_dir, 'image_2')
        self.calib_dir = os.path.join(data_dir, 'calib')
        self.label_dir = os.path.join(data_dir, 'label_2')

        # Augmentation
        self.data_augmentation = split in ['train', 'trainval']
        self.aug_pd   = cfg.get('aug_pd',   False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)
        self.random_flip  = cfg.get('random_flip', 0.5)
        self.random_crop  = cfg.get('random_crop', 0.5)
        self.scale        = cfg.get('scale', 0.4)
        self.shift        = cfg.get('shift', 0.1)
        self.depth_scale  = cfg.get('depth_scale', 'normal')

        # ImageNet stats (used only for MonoDETR-style normalized tensor if needed)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # KITTI class mean sizes [h, w, l]
        self.cls_mean_size = np.array([
            [1.76255119,  0.66068622,  0.84422524],   # Pedestrian
            [1.52563191,  1.62856740,  3.88311640],   # Car
            [1.73698127,  0.59706367,  1.76282397],   # Cyclist
        ])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        self.pd = PhotometricDistort()

    # ── Data loading ────────────────────────────────────────────────────────────

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, f'{idx:06d}.png')
        assert os.path.exists(img_file), f"Image not found: {img_file}"
        return Image.open(img_file).convert('RGB')

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, f'{idx:06d}.txt')
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, f'{idx:06d}.txt')
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    # ── Evaluation ──────────────────────────────────────────────────────────────

    def eval(self, results_dir, logger):
        # Lazy import to avoid numba CUDA compile at import time
        from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
        import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti

        logger.info("==> Loading detections and GTs...")
        img_ids  = [int(i) for i in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)
        test_id  = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        logger.info('==> Evaluating (official)...')
        car_moderate = 0.0
        for category in self.writelist:
            if category not in test_id:
                continue
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(
                gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate

    def __len__(self):
        return len(self.idx_list)

    # ── __getitem__ ─────────────────────────────────────────────────────────────

    def __getitem__(self, item):
        index = int(self.idx_list[item])

        # ── Load image ─────────────────────────────────────────────────────────
        img = self.get_image(index)
        orig_w, orig_h = img.size  # PIL: (W, H)

        # Photometric distortion (before resize, on PIL image)
        if self.data_augmentation and self.aug_pd:
            img_np = np.array(img).astype(np.float32)
            img_np = self.pd(img_np).astype(np.uint8)
            img = Image.fromarray(img_np)

        # Random horizontal flip
        random_flip_flag = False
        if self.data_augmentation and random.random() < self.random_flip:
            random_flip_flag = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # ── Resize to VGGT target (518 x 154) ─────────────────────────────────
        img = img.resize((self.vggt_w, self.vggt_h), Image.BILINEAR)

        # Convert to [0,1] tensor (NO mean/std normalization - VGGT does it internally)
        img_arr = np.array(img).astype(np.float32) / 255.0   # [H, W, 3]
        img_vggt = torch.from_numpy(img_arr.transpose(2, 0, 1))  # [3, H, W]

        # Scale factors (original -> VGGT size)
        scale_x = self.vggt_w / orig_w   # 518 / 1242
        scale_y = self.vggt_h / orig_h   # 154 / 375

        features_size = self.resolution // self.downsample  # [37, 11] W x H

        info = {
            'img_id':   index,
            'img_size': np.array([orig_w, orig_h]),
            'bbox_downsample_ratio': self.resolution / features_size,
        }

        if self.split == 'test':
            calib = self.get_calib(index)
            calib_P2_scaled = self._scale_calib(calib.P2, scale_x, scale_y)
            return img_vggt, calib_P2_scaled, {}, info

        # ── Load labels ────────────────────────────────────────────────────────
        objects = self.get_label(index)
        calib   = self.get_calib(index)

        # Flip labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip([orig_w, orig_h])
            for obj in objects:
                x1, _, x2, _ = obj.box2d
                obj.box2d[0] = orig_w - x2
                obj.box2d[2] = orig_w - x1
                obj.alpha = np.pi - obj.alpha
                obj.ry    = np.pi - obj.ry
                if self.aug_calib:
                    obj.pos[0] *= -1
                # Clamp angles to [-pi, pi]
                obj.alpha = (obj.alpha + np.pi) % (2 * np.pi) - np.pi
                obj.ry    = (obj.ry    + np.pi) % (2 * np.pi) - np.pi

        # Scale P2 calibration
        calib_P2_scaled = self._scale_calib(calib.P2, scale_x, scale_y)

        # Build scaled calibration object for projection operations
        calib_scaled = self._build_scaled_calib(calib, scale_x, scale_y)

        # ── Encode labels ──────────────────────────────────────────────────────
        calibs_per_obj = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices        = np.zeros((self.max_objs,),       dtype=np.int64)
        mask_2d        = np.zeros((self.max_objs,),       dtype=bool)
        labels         = np.zeros((self.max_objs,),       dtype=np.int8)
        depth          = np.zeros((self.max_objs, 1),     dtype=np.float32)
        heading_bin    = np.zeros((self.max_objs, 1),     dtype=np.int64)
        heading_res    = np.zeros((self.max_objs, 1),     dtype=np.float32)
        size_2d        = np.zeros((self.max_objs, 2),     dtype=np.float32)
        size_3d        = np.zeros((self.max_objs, 3),     dtype=np.float32)
        src_size_3d    = np.zeros((self.max_objs, 3),     dtype=np.float32)
        boxes          = np.zeros((self.max_objs, 4),     dtype=np.float32)
        boxes_3d       = np.zeros((self.max_objs, 6),     dtype=np.float32)

        object_num = min(len(objects), self.max_objs)

        for i in range(object_num):
            obj = objects[i]

            if obj.cls_type not in self.writelist:
                continue
            if obj.level_str == 'UnKnown' or obj.pos[-1] < 2:
                continue
            if obj.pos[-1] > 65:
                continue

            # Scale 2D bounding box to new image size
            bbox_2d = obj.box2d.copy()
            bbox_2d[[0, 2]] *= scale_x  # x coords
            bbox_2d[[1, 3]] *= scale_y  # y coords
            bbox_2d = np.clip(bbox_2d,
                              [0, 0, 0, 0],
                              [self.vggt_w-1, self.vggt_h-1, self.vggt_w, self.vggt_h])

            # 2D center
            center_2d = np.array(
                [(bbox_2d[0] + bbox_2d[2]) / 2,
                 (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)
            corner_2d = bbox_2d.copy()

            # 3D center projected to scaled image
            center_3d = obj.pos + [0, -obj.h / 2, 0]  # 3D center in camera space
            center_3d_img, _ = calib_scaled.rect_to_img(center_3d.reshape(1, 3))
            center_3d_img = center_3d_img[0]  # [2]

            if random_flip_flag and not self.aug_calib:
                center_3d_img[0] = self.vggt_w - center_3d_img[0]

            # Filter out-of-image 3D centers
            if not (0 <= center_3d_img[0] < self.vggt_w and
                    0 <= center_3d_img[1] < self.vggt_h):
                continue

            # Normalize coordinates
            center_2d_norm    = center_2d / self.resolution
            center_3d_img_norm = center_3d_img / self.resolution

            w2d = bbox_2d[2] - bbox_2d[0]
            h2d = bbox_2d[3] - bbox_2d[1]
            size_2d_norm = np.array([w2d, h2d]) / self.resolution

            corner_2d_norm = corner_2d.copy()
            corner_2d_norm[[0, 2]] /= self.vggt_w
            corner_2d_norm[[1, 3]] /= self.vggt_h

            l = center_3d_img_norm[0] - corner_2d_norm[0]
            r = corner_2d_norm[2]     - center_3d_img_norm[0]
            t = center_3d_img_norm[1] - corner_2d_norm[1]
            b = corner_2d_norm[3]     - center_3d_img_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l, r, t, b = max(l,0), max(r,0), max(t,0), max(b,0)
                else:
                    continue

            cls_id = self.cls2id[obj.cls_type]
            labels[i]   = cls_id
            size_2d[i]  = w2d, h2d
            boxes[i]    = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_img_norm[0], center_3d_img_norm[1], l, r, t, b

            # Depth
            crop_scale = 1.0  # no crop augmentation in this version
            if self.depth_scale == 'normal':
                depth[i] = obj.pos[-1] * crop_scale
            elif self.depth_scale == 'inverse':
                depth[i] = obj.pos[-1] / crop_scale
            else:
                depth[i] = obj.pos[-1]

            # Heading angle (alpha -> heading bin/residual)
            heading_angle = calib.ry2alpha(obj.ry, (obj.box2d[0] + obj.box2d[2]) / 2)
            heading_angle = (heading_angle + np.pi) % (2 * np.pi) - np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # 3D size (relative to class mean)
            src_size_3d[i] = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
            mean_size = self.cls_mean_size[cls_id]
            size_3d[i] = src_size_3d[i] - mean_size

            if obj.trucation <= 0.5 and obj.occlusion <= 2:
                mask_2d[i] = 1

            calibs_per_obj[i] = calib_P2_scaled

        targets = {
            'calibs':     calibs_per_obj,
            'indices':    indices,
            'img_size':   np.array([orig_w, orig_h]),
            'labels':     labels,
            'boxes':      boxes,
            'boxes_3d':   boxes_3d,
            'depth':      depth,
            'size_2d':    size_2d,
            'size_3d':    size_3d,
            'src_size_3d': src_size_3d,
            'heading_bin': heading_bin,
            'heading_res': heading_res,
            'mask_2d':    mask_2d,
            # Decoding uses original image coordinates, so retain original calibration.
            'calib_obj':  calib,
        }

        return img_vggt, calib_P2_scaled, targets, info

    # ── Helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _scale_calib(P2, scale_x, scale_y):
        """
        Scale a 3x4 P2 matrix to match a resized image.
        P2_new[0,:] *= scale_x  (fx, cx, Tx scaled)
        P2_new[1,:] *= scale_y  (fy, cy, Ty scaled)
        """
        P2_scaled = P2.copy()
        P2_scaled[0, :] *= scale_x
        P2_scaled[1, :] *= scale_y
        return P2_scaled

    @staticmethod
    def _build_scaled_calib(calib, scale_x, scale_y):
        """
        Build a Calibration-like object with scaled P2 but same R0, V2C.
        Returns a Calibration initialized from dict (bypasses file loading).
        """
        P2_scaled = KITTIDatasetVGGT._scale_calib(calib.P2, scale_x, scale_y)
        calib_dict = {
            'P2': P2_scaled,
            'P3': calib.P2,  # dummy, not used
            'R0': calib.R0,
            'Tr_velo2cam': calib.V2C,
        }
        return Calibration(calib_dict)
