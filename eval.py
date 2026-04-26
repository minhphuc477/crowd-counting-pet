import argparse
import random
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset
import util.misc as utils
from engine import evaluate
from models import build_model
from models.backbones import resolve_timm_backbone_name


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # Nhóm tham số mô hình: cấu hình kiến trúc chính và các siêu tham số cốt lõi.
    # - Tham số cho backbone dùng để trích xuất đặc trưng thị giác ở nhiều mức.
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - Tham số cho transformer encoder/decoder để mô hình hóa quan hệ không gian-ngữ cảnh.
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    
    # Nhóm tham số hàm loss: điều chỉnh trọng số giữa các mục tiêu học.
    # - Tham số cho matcher (Hungarian) dùng ghép truy vấn dự đoán với ground-truth.
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - Hệ số trọng số cho từng thành phần loss để cân bằng mục tiêu học.
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights

    # Nhóm tham số dữ liệu: đường dẫn, tên tập dữ liệu và các tùy chọn tiền xử lý.
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # Nhóm tham số phụ trợ: logging, checkpoint, thiết bị chạy và các tùy chọn tiện ích.
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--inference_threshold', default=0.5, type=float)
    parser.add_argument('--threshold_sweep', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--threshold_min', default=0.30, type=float)
    parser.add_argument('--threshold_max', default=0.95, type=float)
    parser.add_argument('--threshold_step', default=0.025, type=float)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)

    # Nhóm tham số huấn luyện phân tán: thiết lập tiến trình, đồng bộ và backend giao tiếp.
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # Cố định seed ngẫu nhiên để kết quả thực nghiệm có thể tái lập giữa các lần chạy.
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    resume_checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            resume_checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            resume_checkpoint = utils.load_checkpoint(args.resume, map_location='cpu')
        utils.restore_args_from_checkpoint(args, resume_checkpoint)
        if 'best_threshold' in resume_checkpoint and not args.threshold_sweep:
            args.inference_threshold = resume_checkpoint['best_threshold']

    if args.backbone in {'auto', 'auto_swin'}:
        args.backbone = resolve_timm_backbone_name(args.backbone)

    # Khởi tạo toàn bộ mô hình theo cấu hình hiện tại để sẵn sàng cho huấn luyện hoặc suy luận.
    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)

    # Tạo dataset và dataloader theo cấu hình để bảo đảm luồng dữ liệu đúng định dạng đầu vào.
    val_image_set = 'val'
    dataset_val = build_dataset(image_set=val_image_set, args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    cur_epoch = 0

    # Nạp trọng số pretrained để suy luận/đánh giá với mô hình đã huấn luyện.
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        utils.load_model_state(model_without_ddp, checkpoint['model'])
        cur_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    
    # Thực hiện giai đoạn đánh giá: chỉ suy luận và đo chất lượng mà không cập nhật trọng số.
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    test_stats = evaluate(
        model,
        data_loader_val,
        device,
        vis_dir=vis_dir,
        inference_threshold=args.inference_threshold,
        threshold_sweep=args.threshold_sweep,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    mae, mse = test_stats['mae'], test_stats['mse']
    threshold = test_stats.get('threshold', args.inference_threshold)
    line = f'\nepoch: {cur_epoch}, mae: {mae}, mse: {mse}, threshold: {threshold}' 
    print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    try:
        main(args)
    finally:
        utils.cleanup_distributed_mode()
