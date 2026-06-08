"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .utils import *

class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 norm_style="post",
                 **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, norm_style=norm_style)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.enc_win_list = kwargs['enc_win_list']
        self.enc_shift_mode = kwargs.get('enc_shift_mode', 'none')
        if self.enc_shift_mode not in ('none', 'swin'):
            raise ValueError('enc_shift_mode must be one of "none" or "swin"')
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_shift_attn_mask(self, batch_size, height, width, win_h, win_w, shift_h, shift_w, device):
        if shift_h <= 0 and shift_w <= 0:
            return None
        grid_h = height // win_h
        grid_w = width // win_w
        window_ids = torch.arange(grid_h * grid_w, device=device, dtype=torch.float32)
        window_ids = window_ids.view(1, 1, grid_h, grid_w)
        window_ids = window_ids.repeat_interleave(win_h, dim=2).repeat_interleave(win_w, dim=3)
        window_ids = torch.roll(window_ids, shifts=(-shift_h, -shift_w), dims=(2, 3))
        id_windows = window_partition(window_ids, window_size_h=win_h, window_size_w=win_w)
        id_windows = id_windows.squeeze(-1).transpose(0, 1)
        attn_mask = id_windows.unsqueeze(1) != id_windows.unsqueeze(2)
        attn_mask = attn_mask.repeat(batch_size, 1, 1)
        return attn_mask.repeat_interleave(self.nhead, dim=0)
    
    def forward(self, src, pos_embed, mask):
        bs, c, h, w = src.shape
        
        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):
            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size
            shift_h, shift_w = 0, 0
            if self.enc_shift_mode == 'swin' and idx % 2 == 1:
                shift_h = enc_win_h // 2
                shift_w = enc_win_w // 2
            if shift_h > 0 or shift_w > 0:
                memeory_in = torch.roll(memeory, shifts=(-shift_h, -shift_w), dims=(2, 3))
                pos_embed_in = torch.roll(pos_embed, shifts=(-shift_h, -shift_w), dims=(2, 3))
                mask_in = torch.roll(mask, shifts=(-shift_h, -shift_w), dims=(1, 2))
            else:
                memeory_in, pos_embed_in, mask_in = memeory, pos_embed, mask
            memeory_win, pos_embed_win, mask_win  = enc_win_partition(memeory_in, pos_embed_in, mask_in, enc_win_h, enc_win_w)
            attn_mask = self._build_shift_attn_mask(
                bs,
                h,
                w,
                enc_win_h,
                enc_win_w,
                shift_h,
                shift_w,
                memeory.device,
            )

            # encoder forward
            output = self.encoder.single_forward(
                memeory_win,
                mask=attn_mask,
                src_key_padding_mask=mask_win,
                pos=pos_embed_win,
                layer_idx=idx,
            )

            # reverse encoder window
            memeory = enc_win_partition_reverse(output, enc_win_h, enc_win_w, h, w)
            if shift_h > 0 or shift_w > 0:
                memeory = torch.roll(memeory, shifts=(shift_h, shift_w), dims=(2, 3))
            if self.return_intermediate:
                memeory_list.append(memeory)        
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_


class WinDecoderTransformer(nn.Module):
    """
    Transformer Decoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=2, 
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 norm_style="post",
                 decoder_attention="softmax",
                 decoder_memory_halo=0,
                 decoder_global_context=False,
                 decoder_global_context_mode="residual",
                 return_intermediate_dec=False,
                 dec_win_w=16, dec_win_h=8,
                 ):
        super().__init__()
        if not decoder_global_context:
            decoder_global_context_mode = "none"
        if decoder_global_context_mode not in ("none", "residual", "token"):
            raise ValueError('decoder_global_context_mode must be one of "none", "residual", or "token"')

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, norm_style=norm_style,
                                                attention_type=decoder_attention)

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        self.global_context_mode = decoder_global_context_mode
        if self.global_context_mode == "residual":
            self.global_context_norm = nn.LayerNorm(d_model)
            self.global_context_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.global_context_scale = nn.Parameter(torch.zeros(1))
        else:
            self.global_context_norm = None
            self.global_context_proj = None
            self.register_parameter("global_context_scale", None)
        self._reset_parameters()

        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_decoder_layers
        self.memory_halo = max(0, int(decoder_memory_halo))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def decoder_forward(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during training
        """
        bs, c, h, w = src_shape
        qH, qW = query_feats.shape[-2:]

        # window-rize query input
        query_embed_ = query_embed.permute(1,2,0).reshape(bs, c, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=dec_win_h, window_size_w=dec_win_w)
        tgt = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # decoder attention
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                                                                        query_pos=query_embed_win, **kwargs)
        hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def decoder_forward_dynamic(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during inference
        """       
        # decoder attention
        tgt = query_feats
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                                                                        query_pos=query_embed, **kwargs)
        num_layer, num_elm, num_win, dim = hs_win.shape
        hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs

    def global_context_vector(self, src, mask):
        valid = (~mask).to(dtype=src.dtype).unsqueeze(1)
        denom = valid.sum(dim=(2, 3), keepdim=False).clamp_min(1.0)
        return (src * valid).sum(dim=(2, 3)) / denom

    def apply_global_context(self, src, pos_embed, mask, memory_win, pos_embed_win, mask_win):
        if self.global_context_mode == "none":
            return memory_win, pos_embed_win, mask_win

        bs = src.shape[0]
        windows_per_batch = memory_win.shape[1] // bs
        global_src = self.global_context_vector(src, mask)

        if self.global_context_mode == "residual":
            # Identity-initialized GCNet-style context. This preserves PET's
            # local cross-attention normalization while letting training learn a
            # small image-level bias for every local decoder memory window.
            context = self.global_context_proj(self.global_context_norm(global_src))
            context = context * self.global_context_scale
            context = context.repeat_interleave(windows_per_batch, dim=0).unsqueeze(0)
            return memory_win + context, pos_embed_win, mask_win

        valid = (~mask).to(dtype=pos_embed.dtype).unsqueeze(1)
        denom = valid.sum(dim=(2, 3), keepdim=False).clamp_min(1.0)
        global_pos = (pos_embed * valid).sum(dim=(2, 3)) / denom
        global_src = global_src.repeat_interleave(windows_per_batch, dim=0).unsqueeze(0)
        global_pos = global_pos.repeat_interleave(windows_per_batch, dim=0).unsqueeze(0)
        global_mask = torch.zeros(mask_win.shape[0], 1, dtype=torch.bool, device=mask_win.device)
        memory_win = torch.cat([memory_win, global_src], dim=0)
        pos_embed_win = torch.cat([pos_embed_win, global_pos], dim=0)
        mask_win = torch.cat([mask_win, global_mask], dim=1)
        return memory_win, pos_embed_win, mask_win
    
    def forward(self, src, pos_embed, mask, pqs, **kwargs):
        bs, c, h, w = src.shape
        query_embed, points_queries, query_feats, v_idx = pqs
        self.dec_win_w, self.dec_win_h = kwargs['dec_win_size']
        
        # Match the official PET decoder path: prune the encoder memory with the
        # same active-window index used for query generation.
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        memory_win, pos_embed_win, mask_win = enc_win_partition_with_halo(
            src,
            pos_embed,
            mask,
            int(self.dec_win_h / div_ratio),
            int(self.dec_win_w / div_ratio),
            self.memory_halo,
            self.memory_halo,
        )
        memory_win, pos_embed_win, mask_win = self.apply_global_context(
            src,
            pos_embed,
            mask,
            memory_win,
            pos_embed_win,
            mask_win,
        )
        
        # dynamic decoder forward
        if 'test' in kwargs:
            if v_idx is not None:
                memory_win = memory_win[:, v_idx]
                pos_embed_win = pos_embed_win[:, v_idx]
                mask_win = mask_win[v_idx]
            hs = self.decoder_forward_dynamic(query_feats, query_embed, 
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
            return hs
        else:
            hs = self.decoder_forward(query_feats, query_embed, 
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
            return hs.transpose(1, 2)
        

class TransformerEncoder(nn.Module):
    """
    Base Transformer Encoder
    """
    def __init__(self, encoder_layer, num_layers, **kwargs):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

        if 'return_intermediate' in kwargs:
            self.return_intermediate = kwargs['return_intermediate']
        else:
            self.return_intermediate = False
    
    def single_forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                layer_idx=0):
        
        output = src
        layer = self.layers[layer_idx]
        output = layer(output, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, pos=pos)        
        return output

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        intermediate = []
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return intermediate

        return output


class TransformerDecoder(nn.Module):
    """
    Base Transformer Decoder
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                **kwargs):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, query_pos=query_pos)
            
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu", norm_style="post"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = norm_style == "pre"

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src,
                     src_mask: Optional[Tensor],
                     src_key_padding_mask: Optional[Tensor],
                     pos: Optional[Tensor]):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = torch.nan_to_num(src2, nan=0.0)
        src = src + src2
        src = self.norm1(src)

        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor],
                    src_key_padding_mask: Optional[Tensor],
                    pos: Optional[Tensor]):
        src_norm = self.norm1(src)
        q = k = self.with_pos_embed(src_norm, pos)
        src2 = self.self_attn(q, k, value=src_norm, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = torch.nan_to_num(src2, nan=0.0)
        src = src + src2

        src_norm = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(src_norm)))
        src = src + src2
        return src

    def forward(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu", norm_style="post", attention_type="softmax"):
        super().__init__()
        self.self_attn = build_attention(attention_type, d_model, nhead, dropout)
        self.multihead_attn = build_attention(attention_type, d_model, nhead, dropout)

        # feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.nhead = nhead
        self.d_model = d_model
        self.normalize_before = norm_style == "pre"

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor],
                     memory_mask: Optional[Tensor],
                     tgt_key_padding_mask: Optional[Tensor],
                     memory_key_padding_mask: Optional[Tensor],
                     pos: Optional[Tensor],
                     query_pos: Optional[Tensor]):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt2 = torch.nan_to_num(tgt2, nan=0.0)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # decoder cross attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = torch.nan_to_num(tgt2, nan=0.0)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # feed-forward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor],
                    memory_mask: Optional[Tensor],
                    tgt_key_padding_mask: Optional[Tensor],
                    memory_key_padding_mask: Optional[Tensor],
                    pos: Optional[Tensor],
                    query_pos: Optional[Tensor]):
        tgt_norm = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt_norm, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt2 = torch.nan_to_num(tgt2, nan=0.0)
        tgt = tgt + tgt2

        tgt_norm = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt_norm, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = torch.nan_to_num(tgt2, nan=0.0)
        tgt = tgt + tgt2

        tgt_norm = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt_norm)))
        tgt = tgt + tgt2
        return tgt

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                memory_key_padding_mask, pos, query_pos,
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
            memory_key_padding_mask, pos, query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_encoder(args, **kwargs):
    return WinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        activation=getattr(args, 'transformer_activation', 'relu'),
        norm_style=getattr(args, 'transformer_norm_style', 'post'),
        enc_shift_mode=getattr(args, 'enc_shift_mode', 'none'),
        **kwargs,
    )


def build_decoder(args, **kwargs):
    return WinDecoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        activation=getattr(args, 'transformer_activation', 'relu'),
        norm_style=getattr(args, 'transformer_norm_style', 'post'),
        decoder_attention=getattr(args, 'decoder_attention', 'softmax'),
        decoder_memory_halo=getattr(args, 'decoder_memory_halo', 0),
        decoder_global_context=getattr(args, 'decoder_global_context', False),
        decoder_global_context_mode=getattr(args, 'decoder_global_context_mode', 'residual'),
        return_intermediate_dec=True,
    )


class LinearAttention(nn.Module):
    """Kernelized linear attention matching nn.MultiheadAttention's call shape."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, eps=1e-6):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim={embed_dim} must be divisible by num_heads={num_heads}')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def kernel(x):
        return F.elu(x) + 1.0

    def _project(self, proj, x):
        seq_len, batch_size, _ = x.shape
        x = proj(x)
        x = x.view(seq_len, batch_size, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        **kwargs,
    ):
        if attn_mask is not None:
            raise NotImplementedError('linear decoder attention does not support attn_mask')
        q = self.kernel(self._project(self.q_proj, query))
        k = self.kernel(self._project(self.k_proj, key))
        v = self._project(self.v_proj, value)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).to(dtype=k.dtype, device=k.device)
            valid = valid[:, None, :, None]
            k = k * valid
            v = v * valid

        kv = torch.einsum('bhsd,bhse->bhde', k, v)
        k_sum = k.sum(dim=2)
        denom = torch.einsum('bhld,bhd->bhl', q, k_sum).clamp_min(self.eps)
        out = torch.einsum('bhld,bhde,bhl->bhle', q, kv, denom.reciprocal())
        out = out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], self.embed_dim)
        out = self.out_proj(self.dropout(out))
        return out, None


def build_attention(attention_type, d_model, nhead, dropout):
    if attention_type == 'softmax':
        return nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    if attention_type == 'linear':
        return LinearAttention(d_model, nhead, dropout=dropout)
    raise ValueError(f'Unsupported decoder attention: {attention_type}. Use "softmax" or "linear".')


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
