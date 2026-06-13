# Giải thích hai cấu hình PET VGG16-BN tốt nhất hiện tại

Tài liệu này ghi lại cách hiểu code và kết quả của hai cấu hình:

- `vgg16_bn_drop700_apg_lc_seed42`
- `vgg16_bn_apglc_density_counthead_seed42`

Đồng thời, tài liệu này cũng ghi rõ vì sao biến thể `density_map_loss` mới không nên dùng cho kết quả chính: các log gần nhất cho thấy nó làm model over-count rất nặng.

## Kết luận ngắn gọn

Kết quả tốt nhất hiện tại **không** đến từ `count_head_topk`, và cũng **không** đến từ `density_map_loss`.

Kết quả tốt nhất đến từ PET inference kiểu threshold bình thường, nhưng trong training có thêm APG và scalar count-head auxiliary để ổn định score/count calibration.

| Cấu hình | Vai trò | Kết quả tốt nhất đã thấy |
| --- | --- | --- |
| `vgg16_bn_drop700_apg_lc_seed42` | VGG16-BN + lite FPN + APG, inference bằng PET threshold | `MAE=50.4341`, `MSE=79.2122` |
| `vgg16_bn_apglc_density_counthead_seed42` | APG+LC + scalar density-sum count head auxiliary, inference vẫn bằng threshold | `MAE=48.7967`, `MSE=76.7129` |
| `count_head_topk` eval | Dùng count head để chọn top-K prediction | Xấu, khoảng `MAE=56.0769` |
| `density_map_loss` mới | Spatial density-map auxiliary | Hỏng calibration, `pred_cnt > 1000`, `MAE > 500` |

Khi báo cáo kết quả tốt nhất, dùng:

```text
eval_count_mode=threshold
eval_nms_radius=0
eval_branch_gate=none
```

Không dùng `eval_count_mode=count_head_topk` cho kết quả chính.

## 1. Cấu hình `vgg16_bn_drop700_apg_lc_seed42`

### Mục tiêu

Cấu hình này là PET VGG16-BN với hai thay đổi hữu ích:

1. `timm_adapter=lite_fpn`: giữ đường feature/FPN nhẹ hơn và hợp với VGG16-BN trong repo hiện tại.
2. `apg_loss_coef=1.0`: thêm Auxiliary Point Guidance để giảm điểm yếu của Hungarian matching trong PET.

PET gốc chỉ gán positive cho một phần nhỏ point-query qua Hungarian matching. Trong crowd counting, nhiều query nằm gần người nhưng không được supervise trực tiếp, nên score calibration dễ bị lệch. APG sửa điểm này bằng cách supervise query gần GT point nhất.

### Lệnh train

File [train.sh](../train.sh) hiện đang chứa cấu hình này:

```bash
#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA="${DATA:-./data/ShanghaiTech/part_A}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
  --backbone vgg16_bn \
  --timm_adapter lite_fpn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/vgg16_bn_drop700_apg_lc_seed42 \
  --device cuda \
  --num_workers 2 \
  --batch_size 8 \
  --epochs 1500 \
  --eval_freq 5 \
  --lr_scheduler step \
  --lr_drop 700 \
  --lr_gamma 0.1 \
  --lr 0.0001 \
  --lr_backbone 0.00001 \
  --weight_decay 0.0001 \
  --clip_max_norm 0.1 \
  --patch_size 256 \
  --crop_attempts 1 \
  --min_crop_points 0 \
  --pet_loss_variant paper \
  --apg_loss_coef 1.0 \
  --apg_pos_k 1 \
  --apg_point_coef 5.0 \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --eval_nms_radius 0 \
  --eval_branch_gate none \
  --seed 42
```

### Kết quả

Checkpoint:

```text
outputs/SHA/vgg16_bn_drop700_apg_lc_seed42/best_checkpoint.pth
```

Kết quả trong `best_eval_results.json`:

```json
{
  "epoch": 585,
  "test_mae": 50.934065934065934,
  "test_mse": 79.4391051596846,
  "pred_cnt": 434.2637362637363,
  "gt_cnt": 433.3076923076923
}
```

Sau khi sweep threshold:

```text
MAE=50.4341
MSE=79.2122
score_threshold=0.59
split_threshold=0.45
eval_nms_radius=0.0
eval_branch_gate=none
```

`pred_cnt` gần với `gt_cnt`, nên cấu hình này không bị lỗi global under-count hoặc over-count. Phần sai số còn lại chủ yếu là lỗi theo ảnh/cục bộ.

### Code APG trong `models/pet.py`

Đoạn code chính:

```python
def compute_apg_loss(self, output, targets):
    """Auxiliary Point Guidance for PET point queries.

    APGCC's full method adds auxiliary proposal guidance to stabilize
    point-based matching. PET already owns a fixed point-query grid, so the
    compatible low-risk version is to directly supervise the nearest grid
    query/queries for each GT point as positive proposals.
    """
    logits = output['pred_logits']
    pred_points = output['pred_points']
    point_queries = output.get('points_queries')
    if point_queries is None:
        return logits.sum() * 0.0

    device = logits.device
    img_h, img_w = output['img_shape']
    query_abs = point_queries.to(device=device, dtype=pred_points.dtype).clone()
    query_abs[:, 0] *= img_h
    query_abs[:, 1] *= img_w

    cls_losses = []
    point_losses = []
    for batch_idx, target in enumerate(targets):
        gt_points = target['points'].to(device=device, dtype=pred_points.dtype)
        if gt_points.numel() == 0:
            continue
        query_dist = torch.cdist(gt_points, query_abs, p=2)
        k = min(self.apg_pos_k, query_abs.shape[0])
        nearest = query_dist.topk(k, largest=False).indices.reshape(-1)
        nearest = torch.unique(nearest)

        cls_target = torch.ones(nearest.shape[0], dtype=torch.long, device=device)
        cls_losses.append(self.point_classification_loss(logits[batch_idx, nearest], cls_target))

        gt_for_queries = gt_points[torch.cdist(query_abs[nearest], gt_points, p=2).argmin(dim=1)]
        gt_norm = gt_for_queries.clone()
        gt_norm[:, 0] /= img_h
        gt_norm[:, 1] /= img_w
        point_losses.append(
            F.smooth_l1_loss(pred_points[batch_idx, nearest], gt_norm, reduction='none').sum(dim=-1).mean()
        )
```

Ý nghĩa:

- `query_abs`: đổi point-query từ tọa độ normalized sang pixel.
- `torch.cdist(gt_points, query_abs)`: tính khoảng cách từ GT point đến từng query.
- `nearest`: query gần GT nhất.
- `cls_target=1`: ép query gần GT có class person.
- `point_losses`: ép predicted point của query gần GT nằm gần GT point.

APG được thêm vào cả hai nhánh sparse và dense:

```python
if self.apg_loss_coef > 0:
    if apg_active:
        loss_apg_sparse = self.compute_apg_loss(output_sparse, targets)
        loss_apg_dense = self.compute_apg_loss(output_dense, targets)
    else:
        loss_apg_sparse = output_sparse['pred_logits'].sum() * 0.0
        loss_apg_dense = output_dense['pred_logits'].sum() * 0.0
    loss_dict['loss_apg_sp'] = loss_apg_sparse
    loss_dict['loss_apg_ds'] = loss_apg_dense
    weight_dict['loss_apg_sp'] = apg_weight
    weight_dict['loss_apg_ds'] = apg_weight
    losses += (loss_apg_sparse + loss_apg_dense) * apg_weight
```

Vì vậy log training có:

```text
loss_apg_sp
loss_apg_ds
```

## 2. Cấu hình `vgg16_bn_apglc_density_counthead_seed42`

### Mục tiêu

Tên `density_counthead` ở đây nghĩa là count head được implement theo kiểu density-sum:

- head tạo một map density trên encoder feature;
- tổng của density map là predicted count;
- loss supervise scalar count;
- inference chính vẫn là PET threshold.

Nó **khác** với `density_map_loss`. `density_map_loss` là loss spatial mới thêm sau này và hiện đang thất bại.

### Kết quả tốt nhất

Checkpoint:

```text
outputs/SHA/vgg16_bn_apglc_density_counthead_seed42/best_checkpoint.pth
```

Kết quả tight sweep tốt nhất:

```text
MAE=48.7967
MSE=76.7129
score_threshold=0.575
split_threshold=0.47
eval_nms_radius=0.0
eval_branch_gate=none
eval_soft_split_gate=none
eval_count_mode=threshold
```

Kết quả epoch 32:

```text
epoch: 32
mae: 48.83516483516483
mse: 76.81532012947291
```

Threshold sweep:

```text
Best: mae=48.8132 mse=76.8167 score_threshold=0.58 split_threshold=0.45
```

Tight sweep:

```text
Best: mae=48.7967 mse=76.7129 score_threshold=0.575 split_threshold=0.47
```

### Vì sao train log nói best epoch 32 MAE 48, nhưng eval khác lại ra 56?

Vì eval mode khác nhau.

Đúng:

```text
eval_count_mode=threshold
```

Khi đó PET đếm bằng threshold trên person score. Đây là đường cho kết quả `48.8`.

Sai cho kết quả chính:

```text
eval_count_mode=count_head_topk
```

Top-K dùng scalar count head để chọn K prediction có score cao nhất. Thực nghiệm của bạn cho thấy top-K xấu:

```text
Best: mae=56.0769
MSE=87.6278
score_threshold=0.0
eval_count_mode=count_head_topk
```

Count head đang hữu ích khi làm auxiliary training, nhưng chưa tốt khi dùng trực tiếp làm inference selector.

### Code count head trong `models/pet.py`

```python
class GlobalCountHead(nn.Module):
    """Small density-sum count head used to calibrate PET point proposals."""

    def __init__(self, hidden_dim, init_count=40.0, init_cells=1024.0):
        super().__init__()
        init_density = max(float(init_count), 0.0) / max(float(init_cells), 1.0)
        init_density_logit = math.log(math.expm1(max(init_density, 1e-6)))
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_density_logit)

    def predict_density(self, x, mask=None):
        density = F.softplus(self.net(x.float()).squeeze(1))
        if mask is not None:
            density = density * (~mask).to(dtype=density.dtype, device=density.device)
        return density

    def forward(self, x, mask=None):
        density = self.predict_density(x, mask)
        return density.flatten(1).sum(dim=1)
```

Ý nghĩa:

- `Conv2d -> GELU -> Conv2d`: head nhỏ, không thay đổi PET decoder.
- `softplus`: đảm bảo density không âm.
- `mask`: không đếm padding.
- `flatten(1).sum(dim=1)`: tổng density trên mọi cell thành scalar count.

Trong `pet_forward`, count head được tính từ encoder feature:

```python
if self.count_head is not None:
    count_density = self.count_head.predict_density(self.count_head_features(encode_src), mask)
    outputs['count_density'] = count_density
    outputs['count_pred'] = count_density.flatten(1).sum(dim=1)
```

Loss count head:

```python
def compute_count_head_loss(self, outputs, targets):
    if self.count_head is None or 'count_pred' not in outputs:
        return outputs['split_map_raw'].sum() * 0.0
    pred_counts = outputs['count_pred'].to(dtype=outputs['split_map_raw'].dtype)
    target_counts = torch.as_tensor(
        [target['points'].shape[0] for target in targets],
        dtype=pred_counts.dtype,
        device=pred_counts.device,
    )
    if self.count_head_loss_type == 'l1':
        return F.l1_loss(pred_counts, target_counts)
    if self.count_head_loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_counts, target_counts)
    return F.l1_loss(torch.log1p(pred_counts), torch.log1p(target_counts))
```

Trong `compute_loss`, nó được cộng vào tổng loss:

```python
if self.count_head_loss_coef > 0:
    count_head_active = epoch >= self.count_head_start_epoch and (
        self.count_head_end_epoch < 0 or epoch <= self.count_head_end_epoch
    )
    if count_head_active:
        loss_count_head = self.compute_count_head_loss(outputs, targets)
    else:
        loss_count_head = outputs['split_map_raw'].sum() * 0.0
    loss_dict['loss_count_head'] = loss_count_head
    weight_dict['loss_count_head'] = self.count_head_loss_coef
    losses += loss_count_head * self.count_head_loss_coef
```

Ý nghĩa thực tế:

- `loss_count_head` ép encoder feature học global count.
- PET decoder vẫn học point classification/regression như cũ.
- Inference threshold vẫn dùng `pred_logits` của PET, không dùng `count_pred` để đếm.
- Vì vậy count head là auxiliary regularizer, không phải final counting mechanism.

### Lệnh eval đúng cho kết quả chính

```bash
export DATA=./data/ShanghaiTech/part_A

python eval.py \
  --resume outputs/SHA/vgg16_bn_apglc_density_counthead_seed42/best_checkpoint.pth \
  --dataset_file SHA \
  --data_path "$DATA" \
  --device cuda \
  --num_workers 2 \
  --score_threshold 0.575 \
  --split_threshold 0.47 \
  --eval_count_mode threshold \
  --eval_nms_radius 0 \
  --eval_branch_gate none \
  --results_file eval_results/SHA/final_vgg16_bn_apglc_density_counthead_48p7967.json
```

### Lệnh sweep đúng

```bash
python scripts/sweep_eval_thresholds.py \
  --resume outputs/SHA/vgg16_bn_apglc_density_counthead_seed42/best_checkpoint.pth \
  --dataset_file SHA \
  --data_path "$DATA" \
  --device cuda \
  --num_workers 2 \
  --output_dir eval_results/SHA/vgg16_bn_apglc_density_counthead_threshold_tight_sweep \
  --score_thresholds 0.56 0.565 0.57 0.575 0.58 0.585 0.59 \
  --split_thresholds 0.45 0.47 0.50 \
  --eval_count_modes threshold \
  --eval_nms_radii 0 \
  --eval_branch_gates none
```

## 3. Cảnh báo về `density_map_loss`

Run mới có `density_map_loss` đã thất bại.

Kết quả bạn gửi:

```text
epoch: 345
mae: 595.4560
mse: 671.7420
pred_cnt: 1028.7637
gt_cnt: 433.3077
best mae: 470.7637
best epoch: 210
```

Trước đó một run khác còn over-count nặng hơn:

```text
epoch: 165
mae: 1000.2857
pred_cnt: 1433.5934
gt_cnt: 433.3077
```

Nhận xét:

- `loss_density_map` đã về `0.0000` sau `density_map_end_epoch`, nhưng model đã bị đẩy sang calibration xấu từ trước.
- `loss_ce_sp` và `loss_ce_ds` cao hơn nhiều so với run tốt, cho thấy classifier của point-query bị hỏng.
- `pred_cnt` lớn hơn GT hơn 2 lần, nên đây không phải chỉ là lỗi threshold nhỏ. Đây là detector over-confident trên quá nhiều query.

Code density-map hiện tại:

```python
def build_density_map_targets(self, outputs, targets):
    if 'count_density' not in outputs:
        raise KeyError('count_density is required for density-map supervision')
    density = outputs['count_density']
    device = density.device
    dtype = density.dtype
    batch_size, map_h, map_w = density.shape
    target_density = torch.zeros(batch_size, map_h, map_w, dtype=dtype, device=device)
    img_h, img_w = outputs['sparse']['img_shape']
    img_h = max(float(img_h), 1.0)
    img_w = max(float(img_w), 1.0)

    for batch_idx, target in enumerate(targets):
        points = target['points'].to(device=device, dtype=torch.float32)
        if points.numel() == 0:
            continue
        y = torch.clamp((points[:, 0] / img_h * map_h).long(), 0, map_h - 1)
        x = torch.clamp((points[:, 1] / img_w * map_w).long(), 0, map_w - 1)
        linear_idx = y * map_w + x
        flat_target = target_density[batch_idx].flatten()
        flat_target.scatter_add_(0, linear_idx, torch.ones_like(linear_idx, dtype=dtype))
    return target_density

def compute_density_map_loss(self, outputs, targets):
    if self.count_head is None or 'count_density' not in outputs:
        return outputs['split_map_raw'].sum() * 0.0
    pred_density = outputs['count_density'].to(dtype=outputs['split_map_raw'].dtype)
    grad_scale = self.density_map_grad_scale
    if grad_scale < 1.0:
        pred_density = pred_density.detach() + grad_scale * (pred_density - pred_density.detach())
    target_density = self.build_density_map_targets(outputs, targets).to(dtype=pred_density.dtype)
    if self.density_map_loss_type == 'l1':
        raw_loss = F.l1_loss(pred_density, target_density, reduction='none')
    elif self.density_map_loss_type == 'smooth_l1':
        raw_loss = F.smooth_l1_loss(pred_density, target_density, reduction='none')
    else:
        raw_loss = F.smooth_l1_loss(torch.log1p(pred_density), torch.log1p(target_density), reduction='none')
    pos_weight = max(float(self.density_map_pos_weight), 0.0)
    weights = 1.0 + pos_weight * (target_density > 0).to(dtype=raw_loss.dtype)
    return (raw_loss * weights).sum() / weights.sum().clamp_min(1.0)
```

Vì sao thất bại:

- Target là sparse one-cell splat. Trong ảnh crowd dense, nó có thể ép encoder tạo peak cục bộ quá mạnh.
- Count head dùng chung `encode_src` với PET decoder, nên loss này có thể làm lệch feature mà classifier PET đang cần.
- Dù đã thêm `density_map_grad_scale`, kết quả thực tế vẫn cho thấy calibration bị hỏng.

Kết luận: hiện tại không dùng `--density_map_loss_coef` cho kết quả chính.

## 4. Inference path cần nhớ

Trong `test_forward`, nếu `eval_count_mode=threshold`, PET dùng score threshold như bình thường:

```python
count_topk = self.eval_count_mode == 'count_head_topk' and self.count_head is not None

if out_sparse is not None:
    out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
    out_sparse_eval_scores = self.apply_eval_soft_split_gate(
        out_sparse, outputs['split_map_raw'], 'sparse', out_sparse_scores
    )
    if count_topk:
        index_sparse = out_sparse_eval_scores >= self.eval_count_head_min_score
    else:
        index_sparse = self.get_score_mask(out_sparse_eval_scores).to(out_sparse['pred_logits'].device)
```

Nghĩa là:

- `threshold`: lấy query có person score vượt `score_threshold`.
- `count_head_topk`: lấy ứng viên rồi giữ top-K theo `count_pred`.

Kết quả của bạn cho thấy:

- `threshold` tốt: `MAE=48.7967`.
- `count_head_topk` xấu: `MAE=56.0769`.

Vì vậy count head nên tiếp tục là training auxiliary, không phải inference replacement.

## 5. Khuyến nghị hiện tại

Cho báo cáo và so sánh:

1. Dùng `vgg16_bn_drop700_apg_lc_seed42` làm baseline cải tiến đầu tiên:
   - `MAE=50.4341`
   - APG+LC, threshold inference.

2. Dùng `vgg16_bn_apglc_density_counthead_seed42` làm kết quả tốt nhất:
   - `MAE=48.7967`
   - APG+LC + scalar density-sum count head auxiliary.
   - Eval bằng threshold, không dùng top-K.

3. Không dùng các biến thể sau cho kết quả chính:
   - `density_map_loss`
   - `count_head_topk`
   - `global context`
   - `swin enc_shift`
   - `routed_apg`
   - `qd_apg`

4. Nếu tiếp tục nghiên cứu, hướng hợp lý hơn là:
   - giữ threshold inference;
   - giữ APG+LC;
   - giữ scalar count-head auxiliary;
   - thêm regularization nhẹ vào logits/score calibration, không thêm loss spatial mạnh vào shared encoder.

## 6. Phụ lục: giải thích code chi tiết hơn

Phần này giải thích code theo đúng luồng chạy của model. Mục tiêu là khi nhìn log hoặc sửa code, ta biết tensor nào đi qua đâu và loss nào đang tác động vào phần nào của PET.

### 6.1. Tổng quan kiến trúc PET trong repo

PET trong repo này hoạt động theo luồng:

```text
ảnh đầu vào
  -> backbone VGG16-BN
  -> lite FPN tạo feature 4x và 8x
  -> context encoder xử lý feature 8x
  -> quadtree splitter tạo split_map
  -> sparse branch dùng query stride 8
  -> dense branch dùng query stride 4
  -> mỗi branch xuất pred_logits và pred_points
  -> inference lọc prediction bằng score_threshold
```

Ý nghĩa các biến chính:

```text
samples              batch ảnh dạng NestedTensor
features['8x']       feature stride 8, dùng cho encoder và sparse branch
features['4x']       feature stride 4, dùng cho dense branch
encode_src           feature sau context_encoder
split_map            bản đồ quyết định vùng sparse/dense
outputs['sparse']    output của nhánh sparse
outputs['dense']     output của nhánh dense
pred_logits          logits class background/person
pred_points          tọa độ point dự đoán, normalized về [0, 1]
points_queries       tọa độ query gốc
```

PET không dự đoán density map để đếm như CSRNet. PET đếm bằng số point-query có score người vượt threshold. Vì vậy calibration của `pred_logits[..., 1]` rất quan trọng.

### 6.2. `pet_forward`: chạy encoder, splitter, sparse branch, dense branch

Đoạn code chính:

```python
src, mask = features[self.encode_feats].decompose()
src_pos_embed = pos[self.encode_feats]
assert mask is not None
encode_src = self.context_encoder(src, src_pos_embed, mask)
encode_src = self.quad_context_mixer(encode_src)
context_info = (encode_src, src_pos_embed, mask)
```

Giải thích:

- `self.encode_feats = '8x'`, nên `src` là feature stride 8.
- Với crop `256x256`, feature 8x thường là `32x32`.
- `mask` đánh dấu vùng padding. Nếu không dùng mask trong count head, model sẽ đếm cả vùng padding.
- `context_encoder` cho feature học ngữ cảnh cục bộ/toàn ảnh trong giới hạn window.
- `quad_context_mixer` nên giữ `none` trong kết quả chính, vì các thử nghiệm global/context mạnh đã gây under-count hoặc over-count.

Sau encoder, model tạo split map:

```python
bs, _, src_h, src_w = src.shape
sp_h, sp_w = src_h, src_w
ds_h, ds_w = int(src_h * 2), int(src_w * 2)
split_map = self.quadtree_splitter(encode_src)
split_map_raw_sparse = F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
split_map_raw_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
split_map_dense = split_map_raw_dense
split_map_sparse = 1 - split_map_raw_sparse
```

Nếu crop là `256x256`:

```text
src_h, src_w = 32, 32
sparse query count = 32 * 32 = 1024
dense query count = 64 * 64 = 4096
```

`split_map` không trực tiếp là kết quả count. Nó chỉ giúp PET quyết định vùng nào nên dùng sparse/dense query. Trong kết quả tốt nhất hiện tại, inference vẫn dùng:

```text
eval_branch_gate=none
```

Tức là không ép split map chọn branch cuối cùng. Model vẫn nối prediction từ sparse và dense rồi lọc bằng score threshold.

### 6.3. Sparse branch và dense branch khác nhau thế nào

Trong `pet_forward`:

```python
outputs_sparse = self.quadtree_sparse(samples, features, context_info, **sparse_kwargs)
outputs_dense = self.quadtree_dense(samples, features, context_info, **dense_kwargs)
```

Hai branch dùng cùng kiểu decoder nhưng khác stride:

```text
sparse branch: stride 8, ít query hơn, phù hợp vùng thưa
dense branch:  stride 4, nhiều query hơn, phù hợp vùng đông
```

Mỗi branch xuất:

```text
pred_logits   shape [B, N, 2]
pred_points   shape [B, N, 2]
pred_offsets  shape [B, N, 2]
points_queries shape [N, 2]
```

Trong đó:

- `N=1024` với sparse trên crop 256.
- `N=4096` với dense trên crop 256.
- class `0` là background.
- class `1` là person/head point.

### 6.4. PET loss gốc: Hungarian matching

Trong `compute_loss`, PET gọi `criterion` cho từng branch:

```python
if epoch >= warmup_ep:
    loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
    loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
else:
    loss_dict_sparse = criterion(output_sparse, targets)
    loss_dict_dense = criterion(output_dense, targets)
```

`criterion` dùng Hungarian matching để ghép prediction với GT point. Điểm yếu là: nếu có rất nhiều query nhưng chỉ có vài trăm GT point, chỉ một phần nhỏ query được match positive. Rất nhiều query còn lại bị xem như background.

Đó là lý do PET dễ bị score calibration không ổn:

```text
quá ít positive trực tiếp -> person score khó học ổn định
quá nhiều background query -> model dễ under-count hoặc over-count tùy threshold
```

APG được thêm để giảm vấn đề này.

### 6.5. APG hoạt động thế nào

APG trong repo này nằm ở `compute_apg_loss`.

Ý tưởng:

```text
với mỗi GT point
  -> tìm point-query gần nó nhất
  -> ép query đó là person
  -> ép pred_points của query đó gần GT point
```

Đoạn code:

```python
query_dist = torch.cdist(gt_points, query_abs, p=2)
k = min(self.apg_pos_k, query_abs.shape[0])
nearest = query_dist.topk(k, largest=False).indices.reshape(-1)
nearest = torch.unique(nearest)
```

Giải thích:

- `query_abs`: tọa độ query tính bằng pixel.
- `gt_points`: GT point tính bằng pixel.
- `torch.cdist`: ma trận khoảng cách `[num_gt, num_queries]`.
- `topk(..., largest=False)`: lấy query gần nhất.
- `torch.unique`: tránh duplicate nếu nhiều GT chọn cùng một query.

Classification APG:

```python
cls_target = torch.ones(nearest.shape[0], dtype=torch.long, device=device)
cls_losses.append(self.point_classification_loss(logits[batch_idx, nearest], cls_target))
```

Nghĩa là query gần GT phải học class person.

Regression APG:

```python
gt_for_queries = gt_points[torch.cdist(query_abs[nearest], gt_points, p=2).argmin(dim=1)]
gt_norm = gt_for_queries.clone()
gt_norm[:, 0] /= img_h
gt_norm[:, 1] /= img_w
point_losses.append(
    F.smooth_l1_loss(pred_points[batch_idx, nearest], gt_norm, reduction='none').sum(dim=-1).mean()
)
```

Nghĩa là predicted point của query gần GT phải dịch về đúng GT.

Trong `compute_loss`, APG được cộng cho cả sparse và dense:

```python
loss_apg_sparse = self.compute_apg_loss(output_sparse, targets)
loss_apg_dense = self.compute_apg_loss(output_dense, targets)
losses += (loss_apg_sparse + loss_apg_dense) * apg_weight
```

Vì vậy log có:

```text
loss_apg_sp
loss_apg_ds
```

Tại sao APG+LC tốt:

- APG thêm positive signal trực tiếp cho query gần GT.
- Nó không thay đổi inference.
- Nó làm person score của query gần người ổn định hơn.
- Vì inference của PET đếm bằng threshold, score calibration tốt hơn sẽ giảm MAE.

### 6.6. Count head trong `vgg16_bn_apglc_density_counthead_seed42`

Count head được tạo khi:

```text
--count_head_loss_coef > 0
```

Code:

```python
class GlobalCountHead(nn.Module):
    def __init__(self, hidden_dim, init_count=40.0, init_cells=1024.0):
        super().__init__()
        init_density = max(float(init_count), 0.0) / max(float(init_cells), 1.0)
        init_density_logit = math.log(math.expm1(max(init_density, 1e-6)))
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_density_logit)
```

Giải thích:

- Head này rất nhỏ, chỉ gồm `1x1 conv -> GELU -> 1x1 conv`.
- Nó không thay đổi sparse/dense decoder.
- `init_count=40` và `init_cells=1024` nghĩa là ban đầu mỗi cell có density nhỏ, tổng trên crop 256 khoảng 40.
- Khởi tạo như vậy tránh count head ban đầu output quá lớn hoặc quá nhỏ.

Forward:

```python
def predict_density(self, x, mask=None):
    density = F.softplus(self.net(x.float()).squeeze(1))
    if mask is not None:
        density = density * (~mask).to(dtype=density.dtype, device=density.device)
    return density

def forward(self, x, mask=None):
    density = self.predict_density(x, mask)
    return density.flatten(1).sum(dim=1)
```

Ý nghĩa:

- `softplus` đảm bảo density không âm.
- `mask` loại padding khỏi count.
- Tổng toàn density map thành `count_pred`.

Trong `pet_forward`:

```python
if self.count_head is not None:
    count_density = self.count_head.predict_density(self.count_head_features(encode_src), mask)
    outputs['count_density'] = count_density
    outputs['count_pred'] = count_density.flatten(1).sum(dim=1)
```

Điểm quan trọng: `count_pred` chỉ là auxiliary output. Kết quả 48.8 không dùng `count_pred` để đếm trực tiếp.

### 6.7. Count-head loss

Code:

```python
def compute_count_head_loss(self, outputs, targets):
    if self.count_head is None or 'count_pred' not in outputs:
        return outputs['split_map_raw'].sum() * 0.0
    pred_counts = outputs['count_pred'].to(dtype=outputs['split_map_raw'].dtype)
    target_counts = torch.as_tensor(
        [target['points'].shape[0] for target in targets],
        dtype=pred_counts.dtype,
        device=pred_counts.device,
    )
    if self.count_head_loss_type == 'l1':
        return F.l1_loss(pred_counts, target_counts)
    if self.count_head_loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_counts, target_counts)
    return F.l1_loss(torch.log1p(pred_counts), torch.log1p(target_counts))
```

Ý nghĩa:

- `target_counts` là số point annotation trong crop.
- `pred_counts` là tổng density của count head.
- `log_l1` giúp loss không quá lớn ở ảnh đông người.

Trong training:

```python
losses += loss_count_head * self.count_head_loss_coef
```

Vì vậy count head tác động vào encoder feature như một regularizer tổng count. Nó không ép từng query cụ thể phải bật/tắt, nên ít phá PET hơn các loss spatial mạnh.

### 6.8. Vì sao `count_head_topk` xấu dù count head giúp training

Trong `test_forward`:

```python
count_topk = self.eval_count_mode == 'count_head_topk' and self.count_head is not None
```

Nếu dùng threshold:

```python
index_sparse = self.get_score_mask(out_sparse_eval_scores).to(out_sparse['pred_logits'].device)
index_dense = self.get_score_mask(out_dense_eval_scores).to(out_dense['pred_logits'].device)
```

Nếu dùng top-K:

```python
k = int(outputs['count_pred'][0].detach().round().clamp(min=0, max=scores.numel()).item())
keep = scores.topk(k, largest=True).indices
pred_logits = pred_logits[keep]
pred_points = pred_points[keep]
```

Lý do top-K xấu:

- Count head có thể dự đoán tổng count tương đối ổn.
- Nhưng nó không biết prediction nào là đúng.
- Top-K phụ thuộc hoàn toàn vào thứ hạng score.
- Nếu score ranking có duplicate hoặc miss local point, top-K sẽ chọn sai.

Do đó:

```text
count head tốt khi làm training auxiliary
count head chưa tốt khi thay thế inference threshold
```

Kết quả thực nghiệm xác nhận:

```text
threshold:       MAE=48.7967
count_head_topk: MAE=56.0769
```

### 6.9. Vì sao `density_map_loss` thất bại

`density_map_loss` khác `count_head_loss`.

Count-head loss:

```text
chỉ ép tổng count đúng
```

Density-map loss:

```text
ép từng cell trong density map khớp target point-splat
```

Code target:

```python
y = torch.clamp((points[:, 0] / img_h * map_h).long(), 0, map_h - 1)
x = torch.clamp((points[:, 1] / img_w * map_w).long(), 0, map_w - 1)
linear_idx = y * map_w + x
flat_target.scatter_add_(0, linear_idx, torch.ones_like(linear_idx, dtype=dtype))
```

Nó tạo target rất nhọn: mỗi GT point cộng `1` vào một cell.

Trong crowd dense, target nhọn có thể làm encoder feature lệch:

```text
encoder bị ép tạo peak local mạnh
classifier PET bị over-confident
nhiều query vượt threshold
pred_cnt tăng rất lớn
```

Log bạn gửi xác nhận:

```text
epoch 345
pred_cnt = 1028.76
gt_cnt   = 433.30
MAE      = 595.45
```

Vì vậy hiện tại không dùng:

```text
--density_map_loss_coef
```

cho kết quả chính.

### 6.10. Cách đọc log training

Log tốt thường có:

```text
pred_cnt gần gt_cnt
loss_ce_sp/loss_ce_ds không tăng bất thường
MAE giảm dần hoặc dao động quanh vùng tốt
```

Với APG+LC tốt:

```text
pred_cnt khoảng 434
gt_cnt khoảng 433
MAE sweep khoảng 50.43
```

Với count-head tốt:

```text
epoch 32
MAE khoảng 48.83
sweep tốt nhất 48.79
eval_count_mode=threshold
```

Log xấu:

```text
pred_cnt > 700 trên SHA val
MAE > 200
loss_ce_sp/loss_ce_ds cao dần
best MAE không xuống gần 50 sau nhiều epoch
```

Nếu thấy:

```text
pred_cnt / gt_cnt > 2
```

thì không nên train tiếp cấu hình đó. Đây là lỗi calibration/architecture, không phải thiếu epoch.

Code hiện tại đã có guard trong `main.py` để tránh chạy tiếp các run kiểu này:

```python
def should_abort_for_bad_count(args, epoch, test_stats):
    if bool(getattr(args, 'no_abort_on_bad_count', False)):
        return False, ''
    if epoch < int(getattr(args, 'bad_count_start_epoch', 20)):
        return False, ''
    pred_cnt = float(test_stats.get('pred_cnt', 0.0))
    gt_cnt = float(test_stats.get('gt_cnt', 0.0))
    mae = float(test_stats.get('mae', 0.0))
    ...
    if mae >= mae_limit and bad_ratio >= ratio_limit:
        return True, message
```

Mặc định:

```text
bad_count_ratio_max=2.0
bad_count_mae_min=200.0
bad_count_start_epoch=20
```

Nghĩa là sau epoch 20, nếu validation cho thấy `pred_cnt/gt_cnt >= 2` hoặc `gt_cnt/pred_cnt >= 2` và MAE cũng lớn hơn 200, training sẽ tự dừng và ghi:

```text
abort_reason.json
```

Để tránh lặp lại lỗi density-map, code cũng tự tắt `density_map_loss` nếu command bật nó mà không có cờ xác nhận rủi ro:

```python
def sanitize_unstable_training_args(args):
    density_coef = float(getattr(args, 'density_map_loss_coef', 0.0))
    if density_coef > 0 and not bool(getattr(args, 'allow_unstable_density_map_loss', False)):
        print('WARNING: --density_map_loss_coef was requested but is disabled by default...')
        args.density_map_loss_coef = 0.0
    return args
```

Nếu thực sự muốn debug loss này, phải bật rõ:

```bash
--allow_unstable_density_map_loss
```

Không nên bật cờ này cho run kết quả chính.

### 6.11. Code nào thuộc cấu hình nào

`vgg16_bn_drop700_apg_lc_seed42` dùng:

```text
backbone vgg16_bn
timm_adapter lite_fpn
apg_loss_coef 1.0
count_head_loss_coef 0.0
density_map_loss_coef 0.0
eval_count_mode threshold
```

`vgg16_bn_apglc_density_counthead_seed42` dùng:

```text
backbone vgg16_bn
timm_adapter lite_fpn
apg_loss_coef 1.0
count_head_loss_coef > 0
density_map_loss_coef 0.0
eval_count_mode threshold
```

Không nhầm với cấu hình thất bại:

```text
density_map_loss_coef > 0
```

Tên `density_counthead` nghĩa là count head dùng density-sum architecture, không có nghĩa là bật `density_map_loss`.

### 6.12. Nguồn tham khảo

- PET official repository: https://github.com/cxliu0/PET
- PET paper/arXiv: https://arxiv.org/abs/2308.13814
- APGCC official repository: https://github.com/AaronCIH/APGCC
- APGCC paper/arXiv: https://arxiv.org/abs/2405.10589
