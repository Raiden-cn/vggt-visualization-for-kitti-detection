"""
VGGT-MonoDETR Training Script
Usage:
    # Single node, 6 GPUs (1-6, skip 0)
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun \
        --nproc_per_node=6 \
        --master_port=29501 \
        det3d/train.py \
        --config det3d/configs/vggt_monodetr_kitti.yaml
"""

import os
import sys
import argparse
import yaml
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'monodetr'))

from det3d.datasets.kitti_dataset import KITTIDatasetVGGT
from det3d.models.vggt_monodetr import build_vggt_monodetr

from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from lib.helpers.save_helper import save_checkpoint, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(output_dir, rank, mode):
    logger = logging.getLogger('vggt_monodetr')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if rank == 0:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'{mode}_{timestamp}.log')
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            '[%(asctime)s] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f'Log file: {log_path}')
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Collate list of (img_vggt, calib_P2, targets, info) tuples.
    Returns:
        imgs:    [B, 3, H, W] float tensor (raw [0,1])
        calibs:  [B, 3, 4] float tensor
        targets: list of dicts (one per sample)
        infos:   list of dicts
    """
    imgs, calibs, targets, infos = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    # Stack calibs: each is (3, 4) numpy array
    calibs_tensor = torch.from_numpy(np.stack(calibs, axis=0)).float()

    # targets: list of dicts; convert numpy arrays to tensors (skip non-array values)
    target_list = []
    for t in targets:
        td = {}
        for k, v in t.items():
            if isinstance(v, np.ndarray):
                td[k] = torch.from_numpy(v)
            else:
                td[k] = v
        target_list.append(td)

    return imgs, calibs_tensor, target_list, list(infos)


def prepare_targets(targets, device):
    """Remove padded object slots, matching MonoDETR's target preparation."""
    prepared = []
    tensor_keys = {
        'labels', 'boxes', 'calibs', 'depth', 'size_3d',
        'heading_bin', 'heading_res', 'boxes_3d', 'size_2d', 'src_size_3d',
    }
    for target in targets:
        mask = target['mask_2d'].to(device=device, dtype=torch.bool)
        item = {}
        for key, value in target.items():
            if key in tensor_keys and isinstance(value, torch.Tensor):
                item[key] = value.to(device=device)[mask]
            elif key != 'calib_obj':
                item[key] = value
        prepared.append(item)
    return prepared


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, optimizer, dataloader, epoch, cfg, logger, rank):
    model.train()
    # Keep VGGT frozen (train() override handles this)

    total_loss = 0.0
    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 10)

    for i, (imgs, calibs, targets, infos) in enumerate(dataloader):
        device = next(model.parameters()).device
        imgs    = imgs.to(device, non_blocking=True)
        calibs  = calibs.to(device, non_blocking=True)
        targets = prepare_targets(targets, device)

        # img_sizes: [B, 2] = (W, H)
        img_sizes = torch.tensor(
            [[imgs.shape[3], imgs.shape[2]]] * imgs.shape[0],
            dtype=torch.float32, device=device
        )

        # Forward
        outputs = model(imgs, calibs, targets=targets, img_sizes=img_sizes)

        # Loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()

        if cfg.get('clip_max_norm', 0) > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg['clip_max_norm']
            )

        optimizer.step()
        total_loss += losses.item()

        if rank == 0 and (i + 1) % log_interval == 0:
            loss_strs = '  '.join(
                f"{k}={loss_dict[k].item():.4f}"
                for k in ['loss_ce', 'loss_bbox', 'loss_depth', 'loss_center']
                if k in loss_dict
            )
            logger.info(
                f"Epoch[{epoch}] [{i+1}/{num_batches}]  "
                f"total={losses.item():.4f}  {loss_strs}"
            )

    return total_loss / num_batches


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, dataset, cfg, logger, rank, output_dir):
    model.eval()
    device = next(model.parameters()).device
    results_dir = os.path.join(output_dir, 'results', 'data')
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
    dist.barrier()

    cls_mean_size = dataset.cls_mean_size

    for imgs, calibs, targets, infos in dataloader:
        imgs   = imgs.to(device, non_blocking=True)
        calibs = calibs.to(device, non_blocking=True)
        img_sizes = torch.tensor(
            [[imgs.shape[3], imgs.shape[2]]] * imgs.shape[0],
            dtype=torch.float32, device=device
        )

        outputs = model(imgs, calibs, img_sizes=img_sizes)

        dets = extract_dets_from_outputs(outputs, K=cfg['num_queries'])
        dets = dets.detach().cpu().numpy()

        # Need per-sample Calibration objects for decode
        calib_objs = [t.get('calib_obj') for t in targets]

        results = decode_detections(
            dets, info={
                'img_size': [info['img_size'] for info in infos],
                'img_id':   [info['img_id']   for info in infos],
            },
            calibs=calib_objs,
            cls_mean_size=cls_mean_size,
            threshold=cfg.get('score_threshold', 0.2),
        )

        # Write KITTI-format results
        for img_id, preds in results.items():
            result_path = os.path.join(results_dir, f'{img_id:06d}.txt')
            with open(result_path, 'w') as f:
                for pred in preds:
                    cls_id = int(pred[0])
                    cls_name = dataset.class_name[cls_id]
                    alpha = pred[1]
                    bbox2d = pred[2:6]
                    dims = pred[6:9]
                    loc  = pred[9:12]
                    ry   = pred[12]
                    score = pred[13]
                    f.write(
                        f"{cls_name} -1 -1 {alpha:.2f} "
                        f"{bbox2d[0]:.2f} {bbox2d[1]:.2f} {bbox2d[2]:.2f} {bbox2d[3]:.2f} "
                        f"{dims[0]:.2f} {dims[1]:.2f} {dims[2]:.2f} "
                        f"{loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f} "
                        f"{ry:.2f} {score:.4f}\n"
                    )

    # All ranks write their sampler shard before rank 0 runs official evaluation.
    dist.barrier()
    car_moderate = 0.0
    if rank == 0:
        try:
            car_moderate = dataset.eval(results_dir, logger)
        except Exception as e:
            logger.warning(f"Eval failed: {e}")

    dist.barrier()

    return car_moderate


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser('VGGT-MonoDETR Training')
    parser.add_argument('--config', default='det3d/configs/vggt_monodetr_kitti.yaml')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint path for --eval_only (defaults to output_dir/ckpt/checkpoint.pth)')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # DDP setup
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Output directory
    experiment_name = cfg.get('experiment_name', 'vggt_monodetr_kitti')
    output_dir = args.output_dir or os.path.join(
        cfg.get('save_path', 'outputs'), experiment_name)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)

    mode = 'eval' if args.eval_only else 'train'
    logger = setup_logger(output_dir, rank, mode)
    if rank == 0:
        logger.info(f"Config: {cfg}")
        logger.info(f"World size: {world_size}, Output: {output_dir}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    val_dataset   = KITTIDatasetVGGT('val',   cfg)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        sampler=val_sampler,
        num_workers=cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if args.eval_only:
        train_dataset = train_sampler = train_loader = None
    else:
        train_dataset = KITTIDatasetVGGT('train', cfg)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['batch_size'],
            sampler=train_sampler,
            num_workers=cfg.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    if rank == 0:
        if args.eval_only:
            logger.info(f"Evaluation set: {len(val_dataset)} samples")
        else:
            logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── Model ──────────────────────────────────────────────────────────────────
    model, criterion = build_vggt_monodetr(cfg)
    model = model.to(device)
    criterion = criterion.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        f"Rank {rank}: trainable tensors={len(trainable_params)}, "
        f"params={sum(p.numel() for p in trainable_params):,}"
    )

    if args.eval_only:
        checkpoint_path = args.checkpoint or os.path.join(
            output_dir, 'ckpt', 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        epoch, best_result, best_epoch = load_checkpoint(
            model=model, optimizer=None, filename=checkpoint_path,
            map_location=device, logger=logger,
        )
        logger.info(
            f"Starting evaluation: checkpoint={checkpoint_path}, epoch={epoch}"
        )
        car_moderate = evaluate(
            model, val_loader, val_dataset, cfg, logger, rank, output_dir)
        if rank == 0:
            logger.info(f"Evaluation complete: Car Moderate 3D AP_R40={car_moderate:.4f}")
        dist.destroy_process_group()
        return

    # DDP: only wrap trainable parts, or wrap the whole model
    # Since VGGT is frozen (no grad), DDP won't try to sync its gradients
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    # Only optimize trainable parameters
    param_groups = [
        {
            'params': [p for p in model.module.adaptor.parameters() if p.requires_grad],
            'lr': cfg['lr'] * 0.1,  # adaptor uses lower LR
        },
        {
            'params': [
                p for name, p in model.module.named_parameters()
                if p.requires_grad and 'adaptor' not in name and 'vggt' not in name
            ],
            'lr': cfg['lr'],
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.get('weight_decay', 1e-4),
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[cfg.get('lr_drop', 90)],
        gamma=0.1,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_result = 0.0
    best_epoch  = 0

    ckpt_path = os.path.join(output_dir, 'ckpt', 'checkpoint.pth')
    if cfg.get('resume_model') and os.path.exists(ckpt_path):
        start_epoch, best_result, best_epoch = load_checkpoint(
            model=model.module, optimizer=optimizer,
            filename=ckpt_path, map_location=device, logger=logger,
        )
        lr_scheduler.last_epoch = start_epoch - 1
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, best={best_result:.4f}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)

    for epoch in range(start_epoch, cfg['max_epoch']):
        train_sampler.set_epoch(epoch)
        np.random.seed(np.random.get_state()[1][0] + epoch)

        t0 = time.time()
        avg_loss = train_one_epoch(
            model, criterion, optimizer,
            train_loader, epoch, cfg, logger, rank,
        )
        lr_scheduler.step()
        elapsed = time.time() - t0

        if rank == 0:
            logger.info(
                f"Epoch [{epoch}/{cfg['max_epoch']-1}] "
                f"avg_loss={avg_loss:.4f}  time={elapsed:.1f}s  "
                f"lr={optimizer.param_groups[1]['lr']:.2e}"
            )

            # Persist progress before evaluation so a slow or interrupted
            # evaluator can never discard a completed training epoch.
            save_checkpoint(
                get_checkpoint_state(
                    model.module, optimizer, epoch, best_result, best_epoch),
                filename=os.path.join(output_dir, 'ckpt', 'checkpoint'),
            )
            logger.info(f"Checkpoint saved after epoch {epoch}")
        dist.barrier()

        # Official KITTI evaluation is CPU-heavy and may take much longer than
        # an epoch. It can be disabled for uninterrupted training and run later
        # with --eval_only against any saved checkpoint.
        eval_frequency = max(1, int(cfg.get('eval_frequency', 5)))
        should_evaluate = cfg.get('eval_during_training', True) and (
            (epoch + 1) % eval_frequency == 0 or epoch == cfg['max_epoch'] - 1
        )
        if should_evaluate:
            car_moderate = evaluate(
                model.module, val_loader, val_dataset, cfg, logger, rank, output_dir)
            if rank != 0:
                continue

            logger.info(f"  Val Car Moderate mAP: {car_moderate:.4f}")

            if car_moderate > best_result:
                best_result = car_moderate
                best_epoch  = epoch
                save_checkpoint(
                    get_checkpoint_state(model.module, optimizer, epoch, best_result, best_epoch),
                    filename=os.path.join(output_dir, 'ckpt', 'best_model'),
                )
            save_checkpoint(
                get_checkpoint_state(model.module, optimizer, epoch, best_result, best_epoch),
                filename=os.path.join(output_dir, 'ckpt', 'checkpoint'),
            )
            logger.info(f"  Best: {best_result:.4f} @ epoch {best_epoch}")
        elif rank == 0 and not cfg.get('eval_during_training', True) and epoch == start_epoch:
            logger.info(
                "Evaluation during training is disabled; use --eval_only with a saved checkpoint."
            )

    dist.destroy_process_group()


def get_checkpoint_state(model, optimizer, epoch, best_result, best_epoch):
    # VGGT is frozen and loaded from the local pretrained cache on every run.
    # Excluding it keeps checkpoints small without changing resume behavior.
    model_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith('vggt.')
    }
    return {
        'epoch': epoch + 1,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict(),
        'best_result': best_result,
        'best_epoch': best_epoch,
    }


if __name__ == '__main__':
    main()
