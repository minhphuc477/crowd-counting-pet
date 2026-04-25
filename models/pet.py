"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding


class BasePETCount(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'
    
    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # Tạo mã hóa vị trí dày đặc ở mọi pixel để truy vấn điểm giữ đúng thông tin tọa độ.
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # Lấy kích thước ảnh/feature map hiện tại để tính chỉ số và phép biến đổi không gian chính xác.
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride//2 -1) // stride

        # Sinh tập điểm truy vấn ban đầu theo lưới để quét không gian ảnh một cách hệ thống.
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # Trích xuất embedding vị trí tương ứng với các điểm truy vấn đã sinh.
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]
        query_embed = query_embed.view(bs, c, h, w)

        # Lấy đặc trưng tại điểm truy vấn theo cơ chế tương đương nội suy lân cận gần nhất.
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down,shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)

        return query_embed, points_queries, query_feats
    
    def points_queris_embed_inference(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during inference
        """
        # Tạo mã hóa vị trí dày đặc ở mọi pixel để truy vấn điểm giữ đúng thông tin tọa độ.
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # Lấy kích thước ảnh/feature map hiện tại để tính chỉ số và phép biến đổi không gian chính xác.
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride//2 -1) // stride

        # Sinh các điểm truy vấn theo bước stride hiện tại để chuẩn bị cho bước truy xuất đặc trưng.
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # Lấy embedding của các điểm truy vấn để cung cấp tín hiệu vị trí cho decoder.
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]

        # Truy xuất đặc trưng cho điểm truy vấn bằng cách ánh xạ gần nhất trên feature map.
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        
        # Chia tensor thành các cửa sổ cục bộ để giảm chi phí attention và tăng tính song song.
        query_embed = query_embed.reshape(bs, c, h, w)
        points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
        query_feats = query_feats.reshape(bs, c, h, w)

        dec_win_w, dec_win_h = kwargs['dec_win_size']
        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
        
        # Sinh động các điểm truy vấn mới dựa trên kết quả lớp trước để tinh chỉnh vị trí dự đoán.
        div = kwargs['div']
        div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
        valid_div = (div_win > 0.5).sum(dim=0)[:,0] 
        v_idx = valid_div > 0
        query_embed_win = query_embed_win[:, v_idx]
        query_feats_win = query_feats_win[:, v_idx]
        points_queries_win = points_queries_win.to(v_idx.device)[:, v_idx].reshape(-1, 2)
    
        return query_embed_win, points_queries_win, query_feats_win, v_idx
    
    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # Đồng thời tạo điểm truy vấn và embedding vị trí để chuẩn bị đầu vào cho transformer.
        if 'train' in kwargs:
            query_embed, points_queries, query_feats = self.points_queris_embed(samples, self.pq_stride, src, **kwargs)
            query_embed = query_embed.flatten(2).permute(2,0,1) # NxCxHxW --> (HW)xNxC
            v_idx = None
        else:
            query_embed, points_queries, query_feats, v_idx = self.points_queris_embed_inference(samples, self.pq_stride, src, **kwargs)

        out = (query_embed, points_queries, query_feats, v_idx)
        return out
    
    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = torch.nan_to_num(self.class_embed(hs), nan=0.0, posinf=20.0, neginf=-20.0)
        # Chuẩn hóa tọa độ/giá trị về khoảng [0, 1] để ổn định huấn luyện và dễ so sánh giữa ảnh.
        outputs_offsets = torch.nan_to_num(
            (self.coord_embed(hs).sigmoid() - 0.5) * 2.0,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        # Chuẩn hóa tọa độ điểm truy vấn theo kích thước ảnh để tránh phụ thuộc độ phân giải tuyệt đối.
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().to(hs.device)
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # Nội suy lại biên độ offset khi test để khớp thang đo dùng trong giai đoạn suy luận.
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)

        outputs_points = torch.nan_to_num(outputs_offsets[-1] + points_queries, nan=0.0, posinf=2.0, neginf=-1.0)
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets[-1]}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        # Chuẩn bị tensor truy vấn điểm theo định dạng mà transformer decoder yêu cầu.
        pqs = self.get_point_query(samples, features, **kwargs)
        
        # Truy vấn đặc trưng theo tọa độ điểm để decoder tập trung vào vùng có khả năng chứa đối tượng.
        kwargs['pq_stride'] = self.pq_stride
        hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)

        # Tạo đầu ra dự đoán cuối cùng từ đặc trưng đã được xử lý qua các khối mạng.
        points_queries = pqs[1]
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        return outputs
    

class PET(nn.Module):
    """ 
    Point quEry Transformer
    """
    def __init__(self, backbone, num_classes, args=None):
        super().__init__()
        self.backbone = backbone
        
        # Tạo positional embedding để mô hình nắm được thông tin vị trí không gian trên feature map.
        self.pos_embed = build_position_encoding(args)

        # Chiếu đặc trưng về cùng chiều kênh mong muốn để các khối phía sau xử lý nhất quán.
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            ]
        )

        # Mã hóa ngữ cảnh toàn cục để làm giàu thông tin trước khi dự đoán điểm đếm.
        self.encode_feats = '8x'
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]  # encoder window size
        args.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(args, enc_win_list=enc_win_list)

        # Dự đoán bản đồ tách vùng quadtree để quyết định nơi dùng nhánh sparse hoặc dense.
        context_patch = (128, 64)
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h ,context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # Tổ chức truy vấn điểm theo quadtree để phân bổ tài nguyên tính toán hiệu quả hơn.
        args.sparse_stride, args.dense_stride = 8, 4    # point-query stride
        transformer = build_decoder(args)
        self.quadtree_sparse = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer)
        self.quadtree_dense = BasePETCount(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)

    @staticmethod
    def _prepare_outputs_for_loss(outputs):
        prepared_outputs = dict(outputs)
        if 'pred_logits' in prepared_outputs:
            prepared_outputs['pred_logits'] = torch.nan_to_num(
                prepared_outputs['pred_logits'].float(),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        if 'pred_points' in prepared_outputs:
            prepared_outputs['pred_points'] = torch.nan_to_num(
                prepared_outputs['pred_points'].float(),
                nan=0.0,
                posinf=2.0,
                neginf=-1.0,
            )
        if 'pred_offsets' in prepared_outputs:
            prepared_outputs['pred_offsets'] = torch.nan_to_num(
                prepared_outputs['pred_offsets'].float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
        if 'points_queries' in prepared_outputs:
            prepared_outputs['points_queries'] = prepared_outputs['points_queries'].float()
        return prepared_outputs

    @staticmethod
    def _collect_target_density(targets, device):
        density_values = [target['density'].reshape(-1)[0] for target in targets]
        return torch.stack(density_values).to(device=device, dtype=torch.float32)

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse = self._prepare_outputs_for_loss(outputs['sparse'])
        output_dense = self._prepare_outputs_for_loss(outputs['dense'])
        weight_dict = criterion.weight_dict
        warmup_ep = 5
        split_map_sparse = outputs['split_map_sparse'].float()
        split_map_dense = outputs['split_map_dense'].float()

        # Tính tổng loss cho các nhánh để tối ưu mô hình theo mục tiêu huấn luyện đã định nghĩa.
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=split_map_sparse)
            loss_dict_dense = criterion(output_dense, targets, div=split_map_dense)
        else:
            loss_dict_sparse = criterion(output_sparse, targets)
            loss_dict_dense = criterion(output_dense, targets)

        # Tính loss cho nhánh sparse để tối ưu vùng mật độ thấp.
        loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # Tính loss cho nhánh dense để tối ưu vùng mật độ cao.
        loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # Tổng hợp loss của các truy vấn điểm để cập nhật tham số mô hình.
        losses = loss_pq_sparse + loss_pq_dense 

        # Cập nhật dict loss và dict trọng số để bộ criterion tính loss đúng theo cấu hình.
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # Tính loss cho bộ tách quadtree nhằm học quyết định chia vùng hiệu quả.
        den = self._collect_target_density(targets, outputs['split_map_raw'].device)   # crowd density
        bs = len(den)
        ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']

        # Áp ràng buộc cho vùng sparse để tránh dự đoán quá dày ở khu vực thưa.
        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()

        # Áp ràng buộc cho vùng dense để giữ dự đoán ổn định ở khu vực đông.
        if sum(ds_idx) > 0:
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        # Bổ sung loss của splitter vào loss tổng với hệ số trọng số tương ứng.
        loss_split = loss_split_sp + loss_split_ds
        weight_split = 0.1 if epoch >= warmup_ep else 0.0
        loss_dict['loss_split'] = loss_split
        weight_dict['loss_split'] = weight_split

        # Gom toàn bộ thành phần loss thành giá trị cuối cùng dùng cho bước lan truyền ngược.
        losses += loss_split * weight_split
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

    def forward(self, samples: NestedTensor, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # Sử dụng backbone để trích xuất đặc trưng mức cao từ ảnh đầu vào.
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # Tạo positional embedding để mô hình nắm được thông tin vị trí không gian trên feature map.
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed

        # Chiếu đặc trưng về cùng chiều kênh mong muốn để các khối phía sau xử lý nhất quán.
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # Thực hiện hàm forward: truyền dữ liệu qua các khối mạng để tạo ra đầu ra trung gian và đầu ra cuối.
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)   
        return out

    def pet_forward(self, samples, features, pos, **kwargs):
        # Thực hiện bước mã hóa ngữ cảnh để tăng khả năng mô hình phân biệt vùng đông/thưa.
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        context_info = (encode_src, src_pos_embed, mask)
        
        # Áp dụng splitter để phân vùng đặc trưng và điều hướng nhánh xử lý phù hợp.
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = torch.nan_to_num(self.quadtree_splitter(encode_src), nan=0.5, posinf=1.0, neginf=0.0)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
        
        # Chạy forward cho tầng quadtree mức 0 ở nhánh sparse để lấy dự đoán thưa.
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None
        
        # Chạy forward cho tầng quadtree mức 1 ở nhánh dense để tinh chỉnh vùng đông.
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4]
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        else:
            outputs_dense = None
        
        # Chuẩn hóa định dạng đầu ra thành cấu trúc thống nhất cho loss và evaluation.
        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs
    
    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # Tính tổng loss cho các nhánh để tối ưu mô hình theo mục tiêu huấn luyện đã định nghĩa.
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses
    
    def test_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        thrs = 0.5  # inference threshold        
        
        # Xử lý các truy vấn điểm nhánh sparse trước khi kết hợp vào đầu ra chung.
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # Xử lý các truy vấn điểm nhánh dense trước khi tổng hợp kết quả.
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # Đóng gói đầu ra theo định dạng chuẩn để các bước sau dùng trực tiếp.
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        div_out['split_map_raw'] = outputs['split_map_raw']
        return div_out


class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.div_thrs_dict = {8: 0.0, 4:0.5}

    @staticmethod
    def _collect_target_density(targets, device):
        density_values = [target['density'].reshape(-1)[0] for target in targets]
        return torch.stack(density_values).to(device=device, dtype=torch.float32)

    @staticmethod
    def _sanitize_outputs(outputs):
        sanitized_outputs = dict(outputs)
        if 'pred_logits' in sanitized_outputs:
            sanitized_outputs['pred_logits'] = torch.nan_to_num(
                sanitized_outputs['pred_logits'].float(),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        if 'pred_points' in sanitized_outputs:
            sanitized_outputs['pred_points'] = torch.nan_to_num(
                sanitized_outputs['pred_points'].float(),
                nan=0.0,
                posinf=2.0,
                neginf=-1.0,
            )
        if 'pred_offsets' in sanitized_outputs:
            sanitized_outputs['pred_offsets'] = torch.nan_to_num(
                sanitized_outputs['pred_offsets'].float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
        if 'points_queries' in sanitized_outputs:
            sanitized_outputs['points_queries'] = sanitized_outputs['points_queries'].float()
        return sanitized_outputs

    @staticmethod
    def _move_indices_to_device(indices, device):
        return [
            (
                src.to(device=device, dtype=torch.int64),
                tgt.to(device=device, dtype=torch.int64),
            )
            for src, tgt in indices
        ]
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # Tính thành phần loss phân lớp cho truy vấn điểm hoặc bản đồ tách vùng.
        if 'div' in kwargs:
            # Lấy chỉ số ảnh thuộc nhóm sparse/dense để áp dụng loss theo từng loại mẫu.
            den = self._collect_target_density(targets, src_logits.device)
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort)//2]
            sp_idx = den_sort[len(den_sort)//2:]
            eps = 1e-5

            # Tính cross-entropy thô trước khi áp trọng số hoặc mask ràng buộc.
            weights = target_classes.clone().float()
            weights[weights==0] = self.empty_weight[0]
            weights[weights==1] = self.empty_weight[1]
            raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')

            # Nhị phân hóa bản đồ tách để xác định rõ vùng sparse và dense.
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs

            # Áp dụng giám sát kép cho cả ảnh sparse và dense để mô hình học cân bằng.
            loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
            loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
            loss_ce = loss_ce_sp + loss_ce_ds

            # Tính loss cho vùng không chia tách để tránh tạo biên giả không cần thiết.
            non_div_mask = split_map <= div_thrs
            loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
            loss_ce = loss_ce + loss_ce_nondiv
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # Lấy cặp chỉ số matching giữa truy vấn dự đoán và mục tiêu ground-truth.
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Tính loss hồi quy tọa độ điểm dự đoán so với điểm đích.
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            # Tách chỉ số mẫu theo sparse/dense để xử lý loss theo từng nhánh.
            den = self._collect_target_density(targets, src_points.device)
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]

            def _cat_or_empty(index_groups):
                flat_indices = [torch.where(idx[0] == bs_id)[0] for bs_id in index_groups]
                if flat_indices:
                    return torch.cat(flat_indices)
                return idx[0].new_zeros((0,))

            pt_ds_idx = _cat_or_empty(img_ds_idx)
            pt_sp_idx = _cat_or_empty(img_sp_idx)

            # Áp dụng giám sát kép cho cả ảnh sparse và dense để mô hình học cân bằng.
            eps = 1e-5
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs
            loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
            loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
            loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)

            # Tính loss cho vùng không chia tách để tránh tạo biên giả không cần thiết.
            non_div_mask = split_map <= div_thrs
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)   

            # Tổng hợp thành phần point loss cuối cùng dùng cho lan truyền ngược.
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
        return losses

    def _get_src_permutation_idx(self, indices):
        # Hoán vị tensor dự đoán theo chỉ số matching để căn hàng với target tương ứng.
        src_tensors = [src for (src, _) in indices if src.numel() > 0]
        if not src_tensors:
            device = indices[0][0].device if indices else torch.device('cpu')
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices) if src.numel() > 0])
        src_idx = torch.cat(src_tensors)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # Hoán vị target theo chỉ số matching để so khớp đúng với dự đoán.
        tgt_tensors = [tgt for (_, tgt) in indices if tgt.numel() > 0]
        if not tgt_tensors:
            device = indices[0][1].device if indices else torch.device('cpu')
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices) if tgt.numel() > 0])
        tgt_idx = torch.cat(tgt_tensors)
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Lấy kết quả matching giữa output lớp cuối và ground-truth để tính loss chính xác.
        outputs = self._sanitize_outputs(outputs)
        match_device = next(iter(outputs.values())).device
        indices = self._move_indices_to_device(self.matcher(outputs, targets), match_device)

        # Tính số điểm mục tiêu trung bình trên toàn bộ tiến trình để chuẩn hóa thang loss.
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # Tính toàn bộ thành phần loss được yêu cầu trong cấu hình huấn luyện hiện tại.
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses


class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


def build_pet(args):
    device = torch.device(args.device)

    # Khởi tạo toàn bộ mô hình theo cấu hình hiện tại để sẵn sàng cho huấn luyện hoặc suy luận.
    num_classes = 1
    backbone = build_backbone(args)
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # Khởi tạo bộ tiêu chí loss để tính sai số cho từng nhánh dự đoán của mô hình.
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion
