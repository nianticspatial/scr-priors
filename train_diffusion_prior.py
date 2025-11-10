# Code adapted from https://github.com/luost26/diffusion-point-cloud/blob/main/train_ae.py

# MIT License
#
# Copyright (c) 2021 Shitong Luo
# Copyright (c) 2025 Niantic Spatial, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from diffusion.utils.misc import *
from diffusion.models.ddpm import GaussianDiffusion
from diffusion.dataloading import ScanNetData
from diffusion.dataloading.transforms import RandomRotate

# Arguments
parser = argparse.ArgumentParser()

# Datasets and loaders
parser.add_argument('--dataset_path', type=Path, required=True,
                   help='Path to preprocessed ScanNet point cloud data directory (containing scene folders with point_cloud.npz files)')
parser.add_argument('--batch_size', type=int, default=128,
                   help='Number of point clouds to process in each training batch (default: 128)')
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False],
                   help='Whether to apply random rotations to point clouds during training (default: False)')

# Optimizer and scheduler
parser.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint file to resume training from (default: None - start from scratch)')
parser.add_argument('--lr', type=float, default=1e-3,
                   help='Initial learning rate for Adam optimizer (default: 1e-3)')
parser.add_argument('--weight_decay', type=float, default=0,
                   help='L2 regularization weight decay parameter (default: 0)')
parser.add_argument('--max_grad_norm', type=float, default=10,
                   help='Maximum gradient norm for gradient clipping (default: 10)')
parser.add_argument('--end_lr', type=float, default=1e-4,
                   help='Final learning rate for linear decay schedule (default: 1e-4)')
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND,
                   help='Training iteration to start learning rate decay (default: 150,000)')
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND,
                   help='Training iteration to finish learning rate decay (default: 300,000)')

# Training
parser.add_argument('--seed', type=int, default=2020,
                   help='Random seed for reproducible training (default: 2020)')
parser.add_argument('--logging', type=eval, default=True, choices=[True, False],
                   help='Whether to enable TensorBoard logging and checkpoint saving (default: True)')
parser.add_argument('--log_root', type=str, default='./diffusion/logs_ae',
                   help='Root directory for saving training logs and checkpoints (default: ./diffusion/logs_ae)')
parser.add_argument('--device', type=str, default='cuda',
                   help='Device to train on ("cuda" or "cpu") (default: cuda)')
parser.add_argument('--max_iters', type=int, default=float('inf'),
                   help='Maximum number of training iterations (default: infinite - train until interrupted)')
parser.add_argument('--ckpt_freq', type=float, default=1000,
                   help='Frequency (in iterations) to save model checkpoints (default: 1000)')
parser.add_argument('--tag', type=str, default=None,
                   help='Optional tag to append to log directory name for experiment identification (default: None)')
parser.add_argument('--n_points', type=int, default=5120,
                   help='Number of points to sample from each point cloud during training (default: 5120)')

args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix=args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
transform = None
if args.rotate:
    transform = RandomRotate(['pcd'])
logger.info('Transform: %s' % repr(transform))

logger.info('Loading datasets...')

train_dset = ScanNetData(
    path=args.dataset_path,
    transform=transform,
    n_points=args.n_points,
)

train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.batch_size,
    num_workers=0,
))

objective = 'pred_noise'
time_steps = 200
scale_pc = torch.tensor(20.)
centroid_pc = torch.tensor([0.0, 0.0, 0.0])

# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = GaussianDiffusion(ckpt['args'], objective=objective, scale_pc=scale_pc, centroid_pc=centroid_pc, timesteps = time_steps).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = GaussianDiffusion(args, objective=objective, scale_pc=scale_pc, centroid_pc=centroid_pc, timesteps = time_steps).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)

def lr_lambda(current_step: int):
    if current_step < args.sched_start_epoch:
        # Linear warmup
        return float(current_step) / float(max(1, args.sched_start_epoch))
    # Linear decay with end_lr
    progress = float(current_step - args.sched_start_epoch) / float(max(1, args.sched_end_epoch - args.sched_start_epoch))
    return max(args.end_lr / optimizer.defaults['lr'], 1.0 - progress)

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

if args.resume is not None:
    optimizer.load_state_dict(ckpt['others']['optimizer'])
    scheduler.load_state_dict(ckpt['others']['scheduler'])

# Train, validate 
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch['pcd'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()

    # Forward
    loss = model(x)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

    return loss.item()

# Main loop
logger.info('Start training...')
model.train()

try:
    if args.resume is not None:
        it = args.resume.split('_')[-1].split('.')[0]
        it = int(it)
    else:
        it = 1
    while it <= args.max_iters:
        loss = train(it)
        if it % args.ckpt_freq == 0 or it == args.max_iters:
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, -loss, it, opt_states)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
