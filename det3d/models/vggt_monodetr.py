"""
VGGT-MonoDETR: 3D Object Detection using frozen VGGT backbone + MonoDETR decoder.
"""

import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# VGGT
sys.path.insert(0, '/root/projcet_zgx/vggt')
from vggt.models.vggt import VGGT

# MonoDETR utilities
sys.path.insert(0, '/root/projcet_zgx/vggt/monodetr')
# The custom extension is built in-place in MonoDETR's ops directory.
_ops_dir = '/root/projcet_zgx/vggt/monodetr/lib/models/monodetr/ops'
if _ops_dir not in sys.path:
    sys.path.insert(0, _ops_dir)
from lib.models.monodetr.depthaware_transformer import build_depthaware_transformer
from lib.models.monodetr.matcher import build_matcher
from lib.models.monodetr.monodetr import SetCriterion, MLP
from utils.misc import inverse_sigmoid

# Local
sys.path.insert(0, '/root/projcet_zgx/vggt')
from det3d.models.vggt_backbone_adaptor import VGGTBackboneAdaptor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class VGGTMonoDETR(nn.Module):
    """
    VGGT backbone (frozen) + MonoDETR decoder (trainable).

    Trainable: VGGTBackboneAdaptor, DepthAwareTransformer, Prediction Heads
    Frozen:    VGGT Aggregator, VGGT DepthHead, VGGT PointHead, VGGT CameraHead
    """

    def __init__(
        self,
        vggt_model,
        adaptor,
        depthaware_transformer,
        num_classes=3,
        num_queries=50,
        num_feature_levels=4,
        aux_loss=True,
        with_box_refine=True,
        two_stage=False,
        init_box=False,
        group_num=11,
    ):
        super().__init__()

        # Frozen VGGT
        self.vggt = vggt_model
        self.vggt.eval()
        for p in self.vggt.parameters():
            p.requires_grad = False

        # Trainable adaptor
        self.adaptor = adaptor

        # MonoDETR transformer
        self.depthaware_transformer = depthaware_transformer
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.group_num = group_num
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # Label embedding
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)

        # Prediction heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed   = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed  = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed  = MLP(hidden_dim, hidden_dim, 2, 2)

        if init_box:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim * 2)

        num_pred = depthaware_transformer.decoder.num_layers

        if with_box_refine:
            self.class_embed  = _get_clones(self.class_embed, num_pred)
            self.bbox_embed   = _get_clones(self.bbox_embed,  num_pred)
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.angle_embed  = _get_clones(self.angle_embed,  num_pred)
            self.depth_embed  = _get_clones(self.depth_embed,  num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.depthaware_transformer.decoder.dim_embed  = self.dim_embed_3d
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed  = nn.ModuleList([self.class_embed  for _ in range(num_pred)])
            self.bbox_embed   = nn.ModuleList([self.bbox_embed   for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed  = nn.ModuleList([self.angle_embed  for _ in range(num_pred)])
            self.depth_embed  = nn.ModuleList([self.depth_embed  for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

    def train(self, mode=True):
        """Override to keep VGGT always in eval mode."""
        super().train(mode)
        self.vggt.eval()  # Always keep VGGT frozen in eval mode
        return self

    def forward(self, images, calibs, targets=None, img_sizes=None):
        """
        Args:
            images:    [B, 3, H, W]  raw [0,1] pixel values
            calibs:    [B, 3, 4]     scaled P2 matrix
            targets:   list of dicts (training only)
            img_sizes: [B, 2]        (W, H)
        """
        B = images.shape[0]
        device = images.device

        # Step 1: Frozen VGGT feature extraction. VGGT requires dimensions
        # divisible by its patch size. Detector coordinates/calibration remain
        # at the configured dataset resolution; only the backbone view is
        # resized, so normalized prediction coordinates stay unchanged.
        image_h, image_w = images.shape[-2:]
        vggt_h = image_h - image_h % 14
        vggt_w = image_w - image_w % 14
        if (vggt_h, vggt_w) != (image_h, image_w):
            vggt_images = F.interpolate(
                images,
                size=(vggt_h, vggt_w),
                mode='bilinear',
                align_corners=False,
                antialias=True,
            )
        else:
            vggt_images = images
        images_5d = vggt_images.unsqueeze(1)  # [B, 1, 3, H_vggt, W_vggt]

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images_5d)
            depth, depth_conf = self.vggt.depth_head(
                aggregated_tokens_list,
                images=images_5d,
                patch_start_idx=patch_start_idx,
            )

        # Step 2: Adaptor -> multi-scale feature maps
        srcs, masks, pos, depth_pos_embed, depth_pos_embed_ip, weighted_depth = \
            self.adaptor(
                aggregated_tokens_list,
                patch_start_idx,
                images_5d.shape,
                depth,
                depth_conf,
            )

        # Step 3: MonoDETR transformer decoder
        if self.training:
            query_embeds = self.query_embed.weight
        else:
            query_embeds = self.query_embed.weight[:self.num_queries]

        hs, init_reference, inter_references, inter_references_dim, \
            enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
                srcs, masks, pos, query_embeds,
                depth_pos_embed, depth_pos_embed_ip,
            )

        # Step 4: Decode predictions
        outputs_coords  = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths  = []
        outputs_angles  = []

        img_h = images.shape[2]

        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_classes.append(self.class_embed[lvl](hs[lvl]))

            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # Depth estimation: geometric + regression + depth_map sampling
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            if img_sizes is not None:
                box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1:2], min=1.0)
            else:
                box2d_height = torch.clamp(box2d_height_norm * img_h, min=1.0)

            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)
            depth_reg = self.depth_embed[lvl](hs[lvl])

            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map_sample = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True,
            ).squeeze(1)

            depth_ave = torch.cat([
                ((1. / (depth_reg[:, :, 0:1].sigmoid() + 1e-6) - 1.)
                 + depth_geo.unsqueeze(-1) + depth_map_sample) / 3,
                depth_reg[:, :, 1:2],
            ], dim=-1)
            outputs_depths.append(depth_ave)
            outputs_angles.append(self.angle_embed[lvl](hs[lvl]))

        outputs_coord  = torch.stack(outputs_coords)
        outputs_class  = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth  = torch.stack(outputs_depths)
        outputs_angle  = torch.stack(outputs_angles)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes':  outputs_coord[-1],
            'pred_3d_dim': outputs_3d_dim[-1],
            'pred_depth':  outputs_depth[-1],
            'pred_angle':  outputs_angle[-1],
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord,
                      outputs_3d_dim, outputs_angle, outputs_depth):
        return [
            {'pred_logits': a, 'pred_boxes': b, 'pred_3d_dim': c,
             'pred_angle': d, 'pred_depth': e}
            for a, b, c, d, e in zip(
                outputs_class[:-1], outputs_coord[:-1], outputs_3d_dim[:-1],
                outputs_angle[:-1], outputs_depth[:-1])
        ]


def build_vggt_monodetr(cfg):
    """Build VGGTMonoDETR model and SetCriterion."""

    # 1. Frozen VGGT
    print("Loading frozen VGGT-1B...")
    vggt = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True)
    vggt.eval()
    for p in vggt.parameters():
        p.requires_grad = False
    frozen_params = sum(p.numel() for p in vggt.parameters())
    print(f"  VGGT: {frozen_params:,} params (all frozen)")

    # 2. Adaptor
    adaptor = VGGTBackboneAdaptor(
        hidden_dim=cfg.get('hidden_dim', 256),
        depth_max=cfg.get('depth_max', 80.0),
    )

    # 3. MonoDETR transformer
    depthaware_transformer = build_depthaware_transformer(cfg)

    # 4. Main model
    model = VGGTMonoDETR(
        vggt_model=vggt,
        adaptor=adaptor,
        depthaware_transformer=depthaware_transformer,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        num_feature_levels=cfg.get('num_feature_levels', 4),
        aux_loss=cfg.get('aux_loss', True),
        with_box_refine=cfg.get('with_box_refine', True),
        two_stage=cfg.get('two_stage', False),
        init_box=cfg.get('init_box', False),
        group_num=cfg.get('group_num', 11),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params")

    # 5. Matcher
    matcher = build_matcher(cfg)

    # 6. Criterion (no depth_map loss - VGGT depth is frozen, not predicted)
    weight_dict = {
        'loss_ce':     cfg.get('cls_loss_coef',    2.0),
        'loss_bbox':   cfg.get('bbox_loss_coef',   5.0),
        'loss_giou':   cfg.get('giou_loss_coef',   2.0),
        'loss_dim':    cfg.get('dim_loss_coef',    1.0),
        'loss_angle':  cfg.get('angle_loss_coef',  1.0),
        'loss_depth':  cfg.get('depth_loss_coef',  1.0),
        'loss_center': cfg.get('3dcenter_loss_coef', 10.0),
    }

    if cfg.get('aux_loss', True):
        aux_weight_dict = {}
        for i in range(cfg.get('dec_layers', 6) - 1):
            aux_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center']

    criterion = SetCriterion(
        num_classes=cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg.get('focal_alpha', 0.25),
        losses=losses,
        group_num=cfg.get('group_num', 11),
    )

    return model, criterion
