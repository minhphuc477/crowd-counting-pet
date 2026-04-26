"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import copy
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# Đoạn này là workaround cho lỗi tensor rỗng trên một số phiên bản PyTorch/Torchvision cũ.
import torchvision
if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def _distributed_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CHECKPOINT_MODEL_ARG_FIELDS = (
    'backbone',
    'position_embedding',
    'dec_layers',
    'dim_feedforward',
    'hidden_dim',
    'dropout',
    'nheads',
    'lr_scheduler',
    'warmup_epochs',
    'min_lr',
    'set_cost_class',
    'set_cost_point',
    'ce_loss_coef',
    'point_loss_coef',
    'eos_coef',
    'patch_size',
    'split_warmup_epochs',
    'count_loss_coef',
    'enc_win_list',
    'dec_win_sizes',
    'use_shifted_windows',
    'enhanced_point_query',
    'query_context_kernel',
)

UPGRADE_COMPATIBLE_MISSING_KEY_PREFIXES = (
    'quadtree_sparse.query_context_conv.',
    'quadtree_sparse.query_content_fuse.',
    'quadtree_sparse.query_pos_fuse.',
    'quadtree_sparse.query_content_norm.',
    'quadtree_sparse.query_pos_norm.',
    'quadtree_sparse.branch_content_bias',
    'quadtree_sparse.branch_pos_bias',
    'quadtree_dense.query_context_conv.',
    'quadtree_dense.query_content_fuse.',
    'quadtree_dense.query_pos_fuse.',
    'quadtree_dense.query_content_norm.',
    'quadtree_dense.query_pos_norm.',
    'quadtree_dense.branch_content_bias',
    'quadtree_dense.branch_pos_bias',
)


def load_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_args_from_checkpoint(args, checkpoint, fields=None):
    checkpoint_args = checkpoint.get('args') if isinstance(checkpoint, dict) else None
    if checkpoint_args is None:
        return args

    if fields is None:
        fields = CHECKPOINT_MODEL_ARG_FIELDS

    for field in fields:
        if hasattr(checkpoint_args, field):
            setattr(args, field, copy.deepcopy(getattr(checkpoint_args, field)))
    return args


def load_model_state(model, state_dict, strict=True):
    if not strict:
        return model.load_state_dict(state_dict, strict=False)

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = [
        key for key in incompatible.missing_keys
        if not key.startswith(UPGRADE_COMPATIBLE_MISSING_KEY_PREFIXES)
    ]
    if missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            'Checkpoint/model mismatch. Missing keys: {} | Unexpected keys: {}'.format(
                missing_keys or incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        )
    return incompatible


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=_distributed_device())
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Tuần tự hóa dữ liệu về Tensor để truyền qua các tiến trình phân tán.
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    dist_device = _distributed_device()
    tensor = torch.ByteTensor(storage).to(dist_device)

    # Thu thập kích thước Tensor ở từng rank để chuẩn bị all_gather.
    local_size = torch.tensor([tensor.numel()], device=dist_device)
    size_list = [torch.tensor([0], device=dist_device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Nhận Tensor từ mọi rank và ghép lại để tổng hợp dữ liệu toàn cục.
    # Đệm Tensor về cùng kích thước vì torch.all_gather không hỗ trợ tensor khác shape.
    # Xử lý trường hợp cần gom các tensor có kích thước khác nhau giữa các tiến trình.
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=dist_device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=dist_device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # Sắp xếp key để thứ tự nhất quán giữa các tiến trình khi reduce dictionary.
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                meters = str(self).split('  ')[0] if 'mse' in str(self) else str(self)
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=meters,
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=meters,
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis_pad(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    block = 256
    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO: Tổng quát hóa đoạn xử lý này để hỗ trợ thêm nhiều kiểu dữ liệu đầu vào.
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() xuất sang ONNX chưa ổn định trong một số trường hợp.
            # Vì vậy cần gọi _onnx_nested_tensor_from_tensor_list() để tương thích ONNX tracing.
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO: Bổ sung hỗ trợ ảnh có kích thước khác nhau một cách tổng quát hơn.
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])

        # Dòng bên dưới là ví dụ cách tính kích thước nhỏ nhất theo từng chiều khi gom nhiều ảnh khác kích thước vào cùng một batch.
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() là phiên bản triển khai dành cho đường xuất ONNX.
# Phiên bản này tương thích với ONNX tracing tốt hơn hàm gốc.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # Workaround cho giới hạn hiện tại của exporter trong quá trình ONNX tracing.
    # Minh họa thao tác chép nội dung ảnh gốc vào tensor đã pad trước đó để giữ nguyên dữ liệu hợp lệ.
    # Minh họa thao tác cập nhật mask: đặt vùng ảnh hợp lệ thành False để phân biệt với vùng padding.
    # Phép gán theo slicing này hiện chưa được ONNX hỗ trợ đầy đủ.
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def cleanup_distributed_mode():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
