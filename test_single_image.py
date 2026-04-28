import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms

import util.misc as utils
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
    parser.add_argument('--img_path', default='', help='path to the image to evaluate')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--inference_threshold', default=0.5, type=float)

    # Nhóm tham số huấn luyện phân tán: thiết lập tiến trình, đồng bộ và backend giao tiếp.
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, pred, vis_dir, img_path, split_map=None):
    """
    Visualize predictions
    """
    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # Vẽ các điểm dự đoán màu xanh để quan sát trực quan chất lượng đếm.
        size = 3
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # Vẽ bản đồ phân tách vùng sparse/dense để kiểm tra quyết định của quadtree splitter.
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # Lưu ảnh trực quan hóa ra đĩa để phục vụ debug, báo cáo và đối chiếu mô hình.
        if vis_dir is not None:
            # Loại bỏ vùng padding/không hợp lệ trước khi lưu ảnh để kết quả hiển thị chính xác hơn.
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = img_path.split('/')[-1].split('.')[0]
            img_save_path = os.path.join(vis_dir, '{}_pred{}.jpg'.format(name, len(pred[idx])))
            cv2.imwrite(img_save_path, sample_vis)
            print('image save to ', img_save_path)


@torch.no_grad()
def evaluate_single_image(model, img_path, device, inference_threshold=0.5, vis_dir=None):
    model.eval()

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # Đọc ảnh đầu vào từ đường dẫn chỉ định trước khi chạy suy luận.
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Biến đổi ảnh sang tensor và chuẩn hóa theo cấu hình huấn luyện.
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(device)
    img_h, img_w = samples.tensors.shape[-2:]

    # Chạy suy luận để lấy đầu ra dự đoán từ mô hình mà không cập nhật tham số.
    outputs = model(samples, test=True)
    raw_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    outputs_scores = raw_scores[:, :, 1][0]
    keep = outputs_scores > inference_threshold
    outputs_points = outputs['pred_points'][0][keep].detach().cpu()
    print('prediction: ', int(keep.sum().item()))
    
    # Trực quan hóa dự đoán lên ảnh để kiểm tra định tính vùng đúng/sai của mô hình.
    if vis_dir: 
        points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
        split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
        visualization(samples, [points], vis_dir, img_path, split_map=split_map)
    

def main(args):
    # Thiết lập đường dẫn ảnh và checkpoint mô hình để chạy thử nhanh.
    if not getattr(args, 'img_path', '') or not args.resume:
        raise ValueError('Please provide both --img_path and --resume for single-image evaluation.')

    # Khởi tạo toàn bộ mô hình theo cấu hình hiện tại để sẵn sàng cho huấn luyện hoặc suy luận.
    device = torch.device(args.device)

    resume_checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            resume_checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            resume_checkpoint = utils.load_checkpoint(args.resume, map_location='cpu')
        utils.restore_args_from_checkpoint(args, resume_checkpoint)
        if 'best_threshold' in resume_checkpoint:
            args.inference_threshold = resume_checkpoint['best_threshold']

    if args.backbone in {'auto', 'auto_swin', 'auto_maxvit'}:
        args.backbone = resolve_timm_backbone_name(args.backbone)

    model, criterion = build_model(args)
    model.to(device)

    # Nạp trọng số pretrained để suy luận/đánh giá với mô hình đã huấn luyện.
    checkpoint = resume_checkpoint if resume_checkpoint is not None else utils.load_checkpoint(args.resume, map_location='cpu')
    utils.load_model_state(model, checkpoint['model'])
    
    # Thực hiện giai đoạn đánh giá: chỉ suy luận và đo chất lượng mà không cập nhật trọng số.
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    evaluate_single_image(model, args.img_path, device, inference_threshold=args.inference_threshold, vis_dir=vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
