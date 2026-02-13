import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d
from flex_gemm.kernels.triton.utils import get_num_sm as _get_num_sm


# Monkey-patch SPLITK upper bound on the forward masked splitk autotuner.
# FlexGEMM hardcodes MAX_NUM_BLOCKS = 32 * get_num_sm(); we replace config_fn
# to use config.FLEX_GEMM_SPLITK_SM_MULTIPLIER instead.
def _patched_fwd_masked_splitk_configs(input, weight, bias, neighbor, sorted_idx, valid_kernel, valid_kernel_seg):
    N, Co = neighbor.shape[0], weight.shape[0]
    num_blocks = ((N + 127) // 128) * ((Co + 127) // 128)
    sm = _get_num_sm()
    multiplier = config.FLEX_GEMM_SPLITK_SM_MULTIPLIER
    min_log2 = max(0, int(math.log2(sm / num_blocks)))
    max_log2 = max(1, int(math.log2(multiplier * sm / num_blocks) + 1))
    return [{'SPLITK': 2 ** i} for i in range(min_log2, max_log2)]


from flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk import (
    sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk as _autotuned_fwd_masked,
)
_autotuned_fwd_masked.config_fn = _patched_fwd_masked_splitk_configs


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    flex_gemm.ops.spconv.set_algorithm(config.FLEX_GEMM_ALGO)
    flex_gemm.ops.spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)

    # check if neighbor map is already computed
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)
    
    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        self.weight,
        self.bias,
        neighbor_cache,
        self.dilation
    )
    
    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
    
    out = x.replace(out)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')
