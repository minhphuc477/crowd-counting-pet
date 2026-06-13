import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms

import util.misc as utils
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='initialize the backbone randomly instead of loading timm/ImageNet weights')
    parser.add_argument('--allow_random_backbone_fallback', action='store_true',
                        help='allow timm backbones to continue with random init if pretrained weights cannot load')
    parser.add_argument('--timm_adapter', default='lite_fpn', choices=('lite_fpn', 'direct', 'fpn'),
                        help='adapter used to map timm features into PET 4x/8x features')
    parser.add_argument('--fusion_mhf_mode', default='none', choices=('none', 'cem', 'cem_msem', 'full'))
    parser.add_argument('--fusion_mhf_heads', default=1, type=int)
    parser.add_argument('--fusion_mhf_position', default='before', choices=('before', 'post'))
    parser.add_argument('--fusion_mhf_strength', default=1.0, type=float)
    parser.add_argument('--fusion_mhf_activation', default='gelu', choices=('relu', 'gelu'))
    parser.add_argument('--fusion_mhf_impl', default='residual', choices=('residual', 'vmambacc'))
    parser.add_argument('--fusion_fpn_type', default='fpn', choices=('fpn', 'hs2fpn'))
    parser.add_argument('--fusion_mhf_reduction', default=4, type=int)
    parser.add_argument('--fusion_mhf_norm', default='none', choices=('none', 'bn', 'gn'))
    parser.add_argument('--fusion_mhf_spatial_kernel', default=7, type=int)
    parser.add_argument('--fusion_mhf_output_activation', default='none', choices=('none', 'sigmoid'))
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
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
    parser.add_argument('--transformer_activation', default='relu', choices=('relu', 'gelu'))
    parser.add_argument('--transformer_norm_style', default='post', choices=('post', 'pre'))
    parser.add_argument('--decoder_attention', default='softmax', choices=('softmax', 'linear'))
    parser.add_argument('--decoder_memory_halo', default=0, type=int)
    parser.add_argument('--enc_win_sizes', default='', type=str,
                        help='encoder window sizes as "w,h;w,h;..."; empty keeps paper PET defaults')
    parser.add_argument('--sparse_dec_win_size', default='', type=str,
                        help='sparse decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--dense_dec_win_size', default='', type=str,
                        help='dense decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--context_patch_size', default='', type=str,
                        help='quadtree splitter context patch size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--quad_context_mixer', default='none', choices=('none', 'lite'))
    parser.add_argument('--quad_context_levels', default=2, type=int)
    parser.add_argument('--quad_context_shift', default=1, type=int)
    parser.add_argument('--quad_context_mid_dim', default=128, type=int)
    parser.add_argument('--quad_context_activation', default='gelu', choices=('relu', 'gelu'))
    parser.add_argument('--splitter_head', default='pool', choices=('pool', 'conv'))
    parser.add_argument('--splitter_hidden_dim', default=128, type=int)
    parser.add_argument('--splitter_activation', default='gelu', choices=('relu', 'gelu'))
    
    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--count_loss_coef', default=0.0, type=float)
    parser.add_argument('--count_loss_gate', default='detach', choices=('detach', 'soft', 'hard'))
    parser.add_argument('--count_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'))
    parser.add_argument('--count_loss_start_epoch', default=-1, type=int)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights
    parser.add_argument('--pet_loss_variant', default='paper', choices=('paper', 'balanced'))
    parser.add_argument('--negative_loss_coef', default=0.1, type=float)
    parser.add_argument('--non_div_loss_coef', default=0.25, type=float)
    parser.add_argument('--quadtree_loss_coef', default=0.1, type=float)
    parser.add_argument('--quadtree_prior_coef', default=0.025, type=float)
    parser.add_argument('--split_count_threshold', default=2, type=int)
    parser.add_argument('--split_pos_weight', default=1.0, type=float)
    parser.add_argument('--split_threshold', default=0.5, type=float)
    parser.add_argument('--split_threshold_quantile', default=0.5, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)
    parser.add_argument('--eval_max_size', default=1536, type=int)

    # misc parameters
    parser.add_argument('--img_path', default='', help='image path to evaluate')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--override_score_threshold', default=None, type=float,
                        help='override the checkpoint score threshold at evaluation time')
    parser.add_argument('--override_split_threshold', default=None, type=float,
                        help='override the checkpoint split threshold at evaluation time')
    parser.add_argument('--override_split_threshold_quantile', default=None, type=float,
                        help='override the checkpoint split-threshold quantile at evaluation time')
    parser.add_argument('--checkpoint_model_key', default='auto',
                        choices=('auto', 'model', 'model_ema', 'model_raw'),
                        help='checkpoint state to evaluate; auto prefers model_ema when present')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def merge_checkpoint_args(args, checkpoint):
    checkpoint_args = checkpoint.get('args')
    if checkpoint_args is None:
        return args
    if isinstance(checkpoint_args, dict):
        checkpoint_args = argparse.Namespace(**checkpoint_args)

    merged = argparse.Namespace(**vars(checkpoint_args))
    runtime_keys = {
        'resume', 'device', 'vis_dir', 'img_path', 'data_path', 'dataset_file',
        'eval_max_size', 'num_workers', 'seed',
        'override_score_threshold', 'override_split_threshold', 'override_split_threshold_quantile',
        'checkpoint_model_key',
    }
    for key in runtime_keys:
        setattr(merged, key, getattr(args, key))
    return merged


def apply_eval_overrides(args):
    override_score_threshold = getattr(args, 'override_score_threshold', None)
    override_split_threshold = getattr(args, 'override_split_threshold', None)
    override_split_threshold_quantile = getattr(args, 'override_split_threshold_quantile', None)
    if override_score_threshold is not None:
        args.score_threshold = float(override_score_threshold)
    if override_split_threshold is not None:
        args.split_threshold = float(override_split_threshold)
    if override_split_threshold_quantile is not None:
        args.split_threshold_quantile = float(override_split_threshold_quantile)
    return args


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

        # draw predictions (green)
        size = 3
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = img_path.split('/')[-1].split('.')[0]
            img_save_path = os.path.join(vis_dir, '{}_pred{}.jpg'.format(name, len(pred[idx])))
            cv2.imwrite(img_save_path, sample_vis)
            print('image save to ', img_save_path)


@torch.no_grad()
def evaluate_single_image(model, img_path, device, vis_dir=None):
    model.eval()

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {img_path}')
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # transform image
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(device)
    img_h, img_w = samples.tensors.shape[-2:]
    valid_h, valid_w = img_h, img_w
    if samples.mask is not None:
        valid_area = torch.where(~samples.mask[0])
        if valid_area[0].numel() > 0:
            valid_h = int(valid_area[0][-1].item()) + 1
            valid_w = int(valid_area[1][-1].item()) + 1

    # inference
    outputs = model(samples, test=True)
    outputs_points = outputs['pred_points'][0]
    query_points = outputs.get('points_queries')
    prediction_cnt = outputs['pred_logits'].shape[1]
    if query_points is not None:
        if query_points.dim() == 3:
            query_points = query_points[0]
        if query_points.shape[0] == prediction_cnt:
            query_y = query_points[:, 0] * img_h
            query_x = query_points[:, 1] * img_w
            prediction_cnt = int(((query_y < valid_h) & (query_x < valid_w)).sum().item())
    print('prediction: ', prediction_cnt)
    
    # visualize predictions
    if vis_dir: 
        points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
        split_threshold = outputs.get('split_threshold', 0.5)
        if torch.is_tensor(split_threshold):
            split_threshold = float(split_threshold.detach().cpu().item())
        split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > split_threshold).float().numpy()
        visualization(samples, [points], vis_dir, img_path, split_map=split_map)
    

def main(args):
    if not args.img_path:
        raise ValueError('--img_path is required')
    if not args.resume:
        raise ValueError('--resume is required')

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    args = merge_checkpoint_args(args, checkpoint)
    args.no_pretrained_backbone = True
    args = apply_eval_overrides(args)

    # build model
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)

    # load pretrained model
    model_state, eval_model_key = utils.get_checkpoint_model_state(
        checkpoint,
        getattr(args, 'checkpoint_model_key', 'auto'),
    )
    model.load_state_dict(model_state)
    print(f'loaded checkpoint model state: {eval_model_key}')
    
    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    evaluate_single_image(model, args.img_path, device, vis_dir=vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
