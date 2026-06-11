"""
Transformer window-rize tools
"""

import torch
import torch.nn.functional as F


def _sanitize_window_mask(mask_win):
    """Avoid all-masked attention windows.

    PyTorch MultiheadAttention produces NaNs when a query has no valid key. PET
    can create fully padded windows after image padding. Those windows do not
    carry real image content, so unmasking their dummy tokens is preferable to
    masking NaNs after attention.
    """
    if mask_win.numel() == 0:
        return mask_win
    all_masked = mask_win.all(dim=1)
    if all_masked.any():
        mask_win = mask_win.clone()
        mask_win[all_masked] = False
    return mask_win


def enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w):
    """
    window-rize input for encoder
    """
    src_win = window_partition(src, window_size_h=enc_win_h, window_size_w=enc_win_w)
    pos_embed_win = window_partition(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w)
    mask_win = window_partition(mask.unsqueeze(1), window_size_h=enc_win_h, window_size_w=enc_win_w).squeeze(-1).permute(1,0)
    mask_win = _sanitize_window_mask(mask_win)
    return src_win, pos_embed_win, mask_win


def enc_win_partition_with_halo(src, pos_embed, mask, enc_win_h, enc_win_w, halo_h=0, halo_w=0):
    """
    Window-rize input for decoder cross-attention with extra memory context.

    The number and order of windows are unchanged from enc_win_partition, so
    query windows and active-window pruning still line up. Only the key/value
    memory tokens receive a clamped zero-padded halo around each base window.
    """
    halo_h = max(0, int(halo_h))
    halo_w = max(0, int(halo_w))
    if halo_h == 0 and halo_w == 0:
        return enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w)

    B, C, H, W = src.shape
    if H % enc_win_h != 0 or W % enc_win_w != 0:
        raise ValueError(
            f'Feature shape {(H, W)} must be divisible by window '
            f'size {(enc_win_h, enc_win_w)}.'
        )

    region_h = enc_win_h + 2 * halo_h
    region_w = enc_win_w + 2 * halo_w
    src_pad = F.pad(src, (halo_w, halo_w, halo_h, halo_h), value=0.0)
    pos_pad = F.pad(pos_embed, (halo_w, halo_w, halo_h, halo_h), value=0.0)
    mask_pad = F.pad(
        mask.unsqueeze(1).float(),
        (halo_w, halo_w, halo_h, halo_h),
        value=1.0,
    ).squeeze(1).to(torch.bool)

    src_windows = []
    pos_windows = []
    mask_windows = []
    for batch_idx in range(B):
        for y in range(0, H, enc_win_h):
            for x in range(0, W, enc_win_w):
                src_patch = src_pad[batch_idx, :, y:y + region_h, x:x + region_w]
                pos_patch = pos_pad[batch_idx, :, y:y + region_h, x:x + region_w]
                mask_patch = mask_pad[batch_idx, y:y + region_h, x:x + region_w]
                src_windows.append(src_patch.permute(1, 2, 0).reshape(region_h * region_w, C))
                pos_windows.append(pos_patch.permute(1, 2, 0).reshape(region_h * region_w, C))
                mask_windows.append(mask_patch.reshape(region_h * region_w))

    src_win = torch.stack(src_windows, dim=1)
    pos_embed_win = torch.stack(pos_windows, dim=1)
    mask_win = torch.stack(mask_windows, dim=0)
    mask_win = _sanitize_window_mask(mask_win)
    return src_win, pos_embed_win, mask_win


def enc_win_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input for encoder
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0,3,1,2)
    return x


def window_partition(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1)  # to (B, H, W, C)
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C)
    windows = windows.reshape(-1, window_size_h*window_size_w, C).permute(1,0,2) # window_size*window_size, num_windows*B, C
    return windows


def window_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.reshape(B, H*W, -1).permute(1,0,2)
    return x

