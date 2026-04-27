"""
Transformer window partition utilities.
"""

import torch


def _validate_window_shape(H, W, window_size_h, window_size_w):
    if H % window_size_h != 0 or W % window_size_w != 0:
        raise ValueError(
            f"Feature map shape {(H, W)} must be divisible by window size {(window_size_h, window_size_w)}."
        )


def _build_shift_slices(size, window_size, shift_size):
    if shift_size <= 0:
        return ((0, None),)
    return ((0, -window_size), (-window_size, -shift_size), (-shift_size, None))


def build_shifted_window_attention_mask(H, W, window_size_h, window_size_w, shift_h=0, shift_w=0, device=None):
    """
    Build the per-window block mask required by shifted-window attention.
    This follows the Swin masking scheme so wrapped tokens cannot attend
    across artificial cyclic-roll boundaries.
    """
    if not (shift_h or shift_w):
        return None

    _validate_window_shape(H, W, window_size_h, window_size_w)

    img_mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.int64)
    h_slices = _build_shift_slices(H, window_size_h, shift_h)
    w_slices = _build_shift_slices(W, window_size_w, shift_w)
    count = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h[0]:h[1], w[0]:w[1]] = count
            count += 1

    mask_windows = window_partition(img_mask, window_size_h=window_size_h, window_size_w=window_size_w)
    mask_windows = mask_windows.squeeze(-1).permute(1, 0)
    attn_mask = mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)
    return attn_mask


def expand_window_attention_mask(attn_mask, batch_size, num_heads):
    """
    Expand a per-window attention mask to the shape expected by
    nn.MultiheadAttention: (batch_size * num_heads, tgt_len, src_len).
    """
    if attn_mask is None:
        return None

    num_windows, window_area, _ = attn_mask.shape
    attn_mask = attn_mask.unsqueeze(0).expand(batch_size, num_windows, window_area, window_area)
    attn_mask = attn_mask.reshape(batch_size * num_windows, window_area, window_area)
    attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
    return attn_mask


def enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w, shift_h=0, shift_w=0, return_attn_mask=False):
    """
    window-rize input for encoder
    """
    _, _, H, W = src.shape
    _validate_window_shape(H, W, enc_win_h, enc_win_w)

    src_win = window_partition(src, window_size_h=enc_win_h, window_size_w=enc_win_w, shift_h=shift_h, shift_w=shift_w)
    pos_embed_win = window_partition(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w, shift_h=shift_h, shift_w=shift_w)
    mask_win = window_partition(
        mask.unsqueeze(1),
        window_size_h=enc_win_h,
        window_size_w=enc_win_w,
        shift_h=shift_h,
        shift_w=shift_w,
    ).squeeze(-1).permute(1, 0)
    if return_attn_mask:
        attn_mask = build_shifted_window_attention_mask(
            H,
            W,
            enc_win_h,
            enc_win_w,
            shift_h=shift_h,
            shift_w=shift_w,
            device=src.device,
        )
        return src_win, pos_embed_win, mask_win, attn_mask
    return src_win, pos_embed_win, mask_win


def enc_win_partition_reverse(windows, window_size_h, window_size_w, H, W, shift_h=0, shift_w=0):
    """
    reverse window-rized input for encoder
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0,3,1,2)
    if shift_h or shift_w:
        x = x.roll(shifts=(shift_h, shift_w), dims=(2, 3))
    return x


def window_partition(x, window_size_h, window_size_w, shift_h=0, shift_w=0):
    """
    window-rize input
    """
    B, C, H, W = x.shape
    _validate_window_shape(H, W, window_size_h, window_size_w)
    if shift_h or shift_w:
        x = x.roll(shifts=(-shift_h, -shift_w), dims=(2, 3))
    x = x.permute(0,2,3,1)  # to (B, H, W, C)
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C)
    windows = windows.reshape(-1, window_size_h*window_size_w, C).permute(1,0,2) # window_size*window_size, num_windows*B, C
    return windows


def window_partition_reverse(windows, window_size_h, window_size_w, H, W, shift_h=0, shift_w=0):
    """
    reverse window-rized input
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1, 0, 2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    if shift_h or shift_w:
        x = x.roll(shifts=(shift_h, shift_w), dims=(1, 2))
    x = x.reshape(B, H*W, -1).permute(1,0,2)
    return x

