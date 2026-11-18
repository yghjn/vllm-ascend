#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""HiF8 (HiFloat8) W8A8 Linear for DeepSeek-V4 family checkpoints.

Torch has no HiFloat8 dtype, so every HiF8 tensor is carried as ``torch.uint8``
(the raw 8-bit code) and the kernels are told how to read it through their
``*_dtype=torch_npu.hifloat8`` arguments:

* weight: per-channel (per output row) HiF8, produced at load time from the
  checkpoint's block-scaled FP8 matrix (dequantize -> ``npu_dynamic_quant``);
  the FP32 row scales go through ``npu_trans_quant_param`` once.
* activation: per-tensor HiF8 with unit scale, ``npu_dtype_cast`` per call.
  Inputs that are already HiF8 codes (uint8, e.g. the wo_a batch-matmul output
  feeding wo_b) are passed through. There is deliberately no activation
  calibration: a running max updated from Python cannot be captured into an ACL
  graph correctly, and we have no way to exercise the graph path here.

``wo_a`` is grouped (o_groups x o_lora_rank rows) and runs through
``npu_transpose_quant_batchmatmul``, which takes one per-channel scale vector
for all groups; its rows are therefore quantized per o_lora row *across* the
groups, and the weight is left in the same [groups, group_in, o_lora_rank]
layout as the MXFP8 path.

Selected instead of ``ds_linear`` by ``VLLM_ASCEND_DSV41_HIF8_LINEAR=1``;
shared experts stay on the MXFP8 ``ds_linear`` path and MoE is unchanged.

The scheme shares ``ds_linear``'s checkpoint schema and nothing else, so it holds a
``ds_linear`` instance for the parameter declaration rather than subclassing it; see
the class docstring.
"""

import math
from typing import Any

import torch
import torch_npu

from vllm_ascend.utils import FP8_METHOD, maybe_trans_nz

from ..base import AscendLinearScheme
from ..registry import register_scheme
from .w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod

HIF8_LINEAR_ENV = "VLLM_ASCEND_DSV41_HIF8_LINEAR"
_HIF8_FLOAT_INPUTS = (torch.bfloat16, torch.float16, torch.float32)

# Accuracy knobs:
#  VLLM_ASCEND_DSV41_HIF8_WEIGHT_RANGE=<M>  per-channel weight scale maps max|row| -> M.
#      Defaults to 16. "dyn" selects npu_dynamic_quant instead, which maps max|row| to
#      ~3.2e4, deep in HiF8's coarse large-magnitude region -- kept as an explicit ablation
#      value, not as the default, so a run that does not go through run_vllm.sh reproduces
#      the same numbers as one that does. Any other value has to be finite and positive;
#      hif8_weight_range says what 0 / inf / nan / negative would do instead.
#  VLLM_ASCEND_DSV41_HIF8_MX_LINEARS=wq_b,wo_a  Linear name suffixes kept on MXFP8. Matched
#      with endswith, so "wq_b" covers both the attention wq_b and indexer.wq_b, and "wkv"
#      covers both the attention wkv and engram.wkv; write the dotted form to pick just one.
#      wo_a and wo_b have to move together -- see ensure_wo_pair_consistent.


def _env(name, default):
    import os

    return os.environ.get(name, default)


def hif8_weight_range():
    """Parse the weight-range knob; refuse the values that silently corrupt the weights.

    ``_quantize_weight_rows`` divides ``max|row|`` by it, and torch raises on none of the
    degenerate cases: 0 makes every row scale ``inf`` and so every weight exactly 0 (the
    service still starts, every answer is wrong), ``inf`` makes the scale 0 and feeds
    +-inf to the cast, ``nan`` poisons both, and a negative value flips code and scale
    alike. Hence the check here, where the message can still name the variable.
    """
    name = "VLLM_ASCEND_DSV41_HIF8_WEIGHT_RANGE"
    value = _env(name, "16")
    if value == "dyn":
        return None
    try:
        weight_range = float(value)
    except ValueError:
        weight_range = float("nan")  # reported by the check below, which names the value
    if not math.isfinite(weight_range) or weight_range <= 0:
        raise ValueError(
            f'{name}="{value}" has to be a finite number greater than zero (the default '
            'is 16), or "dyn": it is the divisor of max|row| that forms the per-channel '
            "weight scale."
        )
    return weight_range


def hif8_mx_linear_suffixes() -> tuple:
    value = _env("VLLM_ASCEND_DSV41_HIF8_MX_LINEARS", "")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def keep_linear_on_mxfp8(prefix: str) -> bool:
    return any(prefix.endswith(suffix) for suffix in hif8_mx_linear_suffixes())


def ensure_cp_compatible() -> None:
    """Refuse HiF8 under context parallel rather than return wrong numbers.

    Two separate things break. ``wo_a`` is projected by ``AscendDSAImpl._wo_a_bmm``,
    which this scheme hooks, but the context-parallel implementations project it
    themselves (``attention/context_parallel/dsa_cp.py``:
    ``AscendDSACPImpl._forward_o_proj`` and ``AscendDSAPCPImpl``) and read
    ``layer.weight_scale`` as e8m0, while HiF8 stores the int64
    ``npu_trans_quant_param`` output there -- so that path would silently misread the
    scale. And ``AscendDSACPImpl._enable_o_proj_full_weight_switch`` switches full
    weights for ``wo_a`` *and* ``wo_b``, where ``_get_weight_switch_method`` rejects
    any scheme without ``supports_weight_switch``.

    So both have to be on MXFP8, not just ``wo_a``: that puts the whole o_proj path --
    projection and weight switch -- back on upstream's own ``ds_linear``, and leaves
    the remaining five Linears on HiF8. Note that this combination has not been run
    here (no CP-capable setup); the argument for it is only that those two Linears are
    then byte-identical to the baseline.
    """
    from vllm_ascend.attention.utils import enable_pcp
    from vllm_ascend.utils import enable_dsa_cp

    if not (enable_dsa_cp() or enable_pcp()):
        return
    if not (keep_linear_on_mxfp8("wo_a") and keep_linear_on_mxfp8("wo_b")):
        raise NotImplementedError(
            "VLLM_ASCEND_DSV41_HIF8_LINEAR=1 does not support context parallel: "
            "attention/context_parallel/dsa_cp.py projects wo_a without the HiF8 branch "
            "and would read its int64 weight scale as e8m0, and it switches full o_proj "
            "weights for both wo_a and wo_b, which this scheme does not support. Disable "
            "context parallel, or set VLLM_ASCEND_DSV41_HIF8_MX_LINEARS=wo_a,wo_b to keep "
            "the whole o_proj on MXFP8 (the other five Linears stay on HiF8)."
        )


def ensure_wo_pair_consistent(prefix: str) -> None:
    """A HiF8 ``wo_a`` requires a HiF8 ``wo_b``; the reverse pairing is fine.

    ``_wo_a_bmm`` returns HiF8 codes that ``wo_b`` consumes without a re-cast
    (``attention/dsa_v1.py``), so an MXFP8 ``wo_b`` would hand those uint8 codes to
    ``_quantize_input`` -> ``npu_dynamic_mx_quant``, which expects a float activation.
    Only this direction is rejected: an MXFP8 ``wo_a`` emits bf16, which ``apply``
    casts like any other float input, and that is the combination
    ``ensure_cp_compatible`` tells the user to set.
    """
    if not prefix.endswith("wo_a") or keep_linear_on_mxfp8(prefix):
        # Not wo_a, or wo_a is on MXFP8 too -- then the pair moves together already.
        return
    sibling = prefix[: -len("wo_a")] + "wo_b"
    if keep_linear_on_mxfp8(sibling):
        raise ValueError(
            f"VLLM_ASCEND_DSV41_HIF8_MX_LINEARS keeps {sibling} on MXFP8 while {prefix} "
            "stays on HiF8: wo_a emits HiF8 codes that wo_b consumes without a re-cast, so "
            "an MXFP8 wo_b would read them as a float activation. Keep both on MXFP8, or "
            "neither."
        )


def _quantize_weight_rows(dense: torch.Tensor):
    """Per-row HiF8 codes (uint8) + FP32 row scales for a [rows, cols] matrix."""
    weight_range = hif8_weight_range()
    if weight_range is None:
        quantized, row_scale = torch_npu.npu_dynamic_quant(dense.to(torch.bfloat16), dst_type=torch_npu.hifloat8)
        return _as_hif8_codes(quantized), row_scale.to(torch.float32).view(-1)
    row_scale = dense.float().abs().amax(dim=1).clamp_min(1e-12) / weight_range
    scaled = (dense.float() / row_scale.view(-1, 1)).to(torch.bfloat16)
    return _as_hif8_codes(torch_npu.npu_dtype_cast(scaled, dtype=torch_npu.hifloat8)), row_scale


def hif8_linear_enabled() -> bool:
    import os

    return os.environ.get(HIF8_LINEAR_ENV, "0") == "1"


def _dequantize_block_fp8(weight: torch.Tensor, block_scale: torch.Tensor, block: int) -> torch.Tensor:
    """[out, in] E4M3 with [ceil(out/block), ceil(in/block)] scales -> FP32 dense matrix."""
    if block_scale.dtype == torch.float8_e8m0fnu:
        block_scale = block_scale.to(torch.float32)  # numeric cast: exact powers of two
    out_features, in_features = weight.shape
    dense_scale = block_scale.to(torch.float32).repeat_interleave(block, dim=0)[:out_features]
    dense_scale = dense_scale.repeat_interleave(block, dim=1)[:, :in_features]
    return weight.to(torch.float32) * dense_scale


def _as_hif8_codes(quantized: torch.Tensor) -> torch.Tensor:
    return quantized if quantized.dtype == torch.uint8 else quantized.view(torch.uint8)


@register_scheme(FP8_METHOD, "ds_linear_hif8")
class AscendW8A8HIF8DSDynamicLinearMethod(AscendLinearScheme):
    """Loads exactly like ``ds_linear`` (same parameters / checkpoint tensors), runs as HiF8.

    ``ds_linear`` is held, not subclassed: this scheme reuses its checkpoint schema
    (``get_weight`` / ``get_pergroup_param``) and replaces every runtime behaviour.
    Subclassing would also inherit its *identity*, and ops/cv_linear.py's CVLinearWrapper
    dispatches on ``isinstance(scheme, AscendW8A8MXFP8DynamicLinearMethod)`` to reach an
    MXFP8 Cube path that reads layer.weight / layer.weight_scale with
    ``scale_dtype=float8_e8m0fnu``. These weights are HiF8 codes and this weight_scale is
    the int64 ``npu_trans_quant_param`` output, so that path fails on the first forward:

        Tensor dtype:long int itemsize:8, is not compatible with
        tensor_dtype:torch_npu.float8_e8m0fnu itemsize:1

    Not being an instance keeps wq_a / wkv / wq_b off it, at the cost of the Cube/Vector
    overlap for those three; ``apply`` below quantizes the activation itself. The same
    reasoning covers ``prepare_input_for_overlap`` (the base returns None, which is what
    this scheme wants), ``apply_single_group_gmm`` and ``restore_weights_for_rl_loading``:
    all three assume the MXFP8 weight / scale layout.
    """

    # Whatever upstream reads off the scheme by name rather than by type is declared here,
    # because not being an MXFP8 instance is exactly the point of this class:
    # method_adapters.py installs the MX scale weight loader for a RowParallelLinear
    # (wo_b) when supports_unaligned_tp_groups is set, and asserts group_size exists.
    # ds_linear inherits both from the MXFP8 method. The one answer that cannot be restored
    # by an attribute is is_mx_quant_type()'s; get_pergroup_param below settles that one.
    supports_unaligned_tp_groups = True
    model_dtype = None
    supports_weight_switch = False
    weight_switch_gather_specs = ()
    weight_switch_output_gather_specs = ()

    def __init__(self, weight_block_size):
        # Parse the knob here, at model build time, so a bad value stops the run now rather
        # than after the ten minutes it takes to load the weights -- the quantization itself
        # happens in process_weights_after_loading, i.e. once every tensor is already in.
        hif8_weight_range()
        self._ds = AscendW8A8MXFP8DSDynamicLinearMethod(weight_block_size)
        self.group_size = self._ds.group_size
        self.block_size = self._ds.block_size
        self.o_lora_rank = self._ds.o_lora_rank

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return self._ds.get_weight(input_size, output_size, params_dtype)

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params = self._ds.get_pergroup_param(input_size, output_size, params_dtype, layer_type)
        # ds_linear gets input_dim=1 on its scale from method_adapters.py's is_mx_quant_type()
        # fallback, which answers by isinstance and so would skip this scheme. Say it through
        # the _input_dim key the same site reads first ("Schemes whose scales are sharded along
        # the reduction dim can say so directly"), which is what keeps the scale sharded exactly
        # as ds_linear's at every TP size.
        params["_input_dim"] = 1
        return params

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dense = _dequantize_block_fp8(layer.weight.data, layer.weight_scale.data, self.block_size)
        prefix = getattr(layer, "prefix", "")
        if prefix.endswith("wo_a"):
            # The layer may use OTP instead of the model's ordinary TP group, so the group count
            # comes from the loaded output shard -- the same rule as the MXFP8 ds_linear path,
            # which reads it off the already-transposed weight. ``dense`` is still [out, in].
            rank = self.o_lora_rank
            rows, cols = dense.shape
            if rows % rank:
                raise ValueError(
                    f"{prefix}: wo_a output shard {rows} is not a multiple of o_lora_rank {rank}")
            groups = rows // rank
            grouped = dense.view(groups, rank, cols).transpose(0, 1).reshape(rank, groups * cols)
            codes, row_scale = _quantize_weight_rows(grouped)
            # [rank, groups, cols] -> [groups, cols (group_in), rank]: the MXFP8 wo_a layout.
            codes = codes.view(rank, groups, cols).permute(1, 2, 0).contiguous()
            layer.weight.data = codes
            layer.weight_scale.data = torch_npu.npu_trans_quant_param(row_scale.to(torch.float32)).view(-1)
            layer.hif8_bmm_x1_scale = torch_npu.npu_trans_quant_param(
                torch.ones(1, dtype=torch.float32, device=codes.device))
            layer.hif8_quant_scale = torch.ones(1, dtype=torch.bfloat16, device=codes.device)
        else:
            codes, row_scale = _quantize_weight_rows(dense)
            codes = codes.contiguous()
            # The NZ cast goes through maybe_trans_nz rather than npu_format_cast directly, so
            # this path keeps the two opt-outs the rest of the tree relies on: _should_trans_nz
            # declines k=1 / n=1, which aclnnMatmulWeightNZ cannot take, and a layer may set
            # skip_weight_nz_conversion to demand ND (ops/linear.py honours the same flag).
            # The cast stays *before* the transpose, unlike the MXFP8 path, which transposes in
            # the scheme and lets ops/linear.py do the NZ cast afterwards: that order is what
            # the HiF8 kernel was validated against and what README section 1's numbers were
            # produced with, and reordering it changes the buffer handed to npu_quant_matmul,
            # so it is left alone and asserted below instead.
            weight = codes if getattr(layer, "skip_weight_nz_conversion", False) else maybe_trans_nz(codes)
            if tuple(weight.shape) != tuple(dense.shape):
                raise ValueError(
                    f"{prefix}: NZ cast changed the logical shape, {tuple(dense.shape)} -> "
                    f"{tuple(weight.shape)}; the transposed view below assumes it does not")
            layer.weight.data = weight.transpose(0, 1)
            layer.weight_scale.data = torch_npu.npu_trans_quant_param(row_scale.to(torch.float32)).view(-1)
        del dense
        layer.is_hif8 = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            # PreparedLinearInput / (tensor, scale) are the MXFP8 fused-quant calling conventions:
            # they carry an MX per-token scale this kernel has no argument for.
            raise TypeError(
                f"{getattr(layer, 'prefix', 'linear')}: HiF8 Linear got a pre-quantized "
                f"{type(x).__name__} input, which is the MXFP8 fused-quant calling convention.")
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        if x.dtype in _HIF8_FLOAT_INPUTS:
            output_dtype = x.dtype
            x = torch_npu.npu_dtype_cast(x, dtype=torch_npu.hifloat8)
        else:
            # Already HiF8 codes -- the wo_a batch matmul feeds wo_b without a re-cast.
            output_dtype = torch.bfloat16
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        output = torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8,
            output_dtype=output_dtype,
        )
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)
        return output


def apply_hif8_wo_a(wo_a: torch.nn.Module, o: torch.Tensor, perm_x1, perm_x2, perm_y) -> torch.Tensor:
    """HiF8 grouped o_a projection; returns HiF8 codes that wo_b consumes without a re-cast."""
    o = torch_npu.npu_quantize(o, wo_a.hif8_quant_scale, None, torch_npu.hifloat8, -1, False)
    return torch_npu.npu_transpose_quant_batchmatmul(
        o,
        wo_a.weight,
        dtype=torch_npu.hifloat8,
        x1_scale=wo_a.hif8_bmm_x1_scale,
        x2_scale=wo_a.weight_scale,
        group_sizes=(0, 0, 0),
        perm_x1=perm_x1,
        perm_x2=perm_x2,
        perm_y=perm_y,
        batch_split_factor=1,
        x1_dtype=torch_npu.hifloat8,
        x2_dtype=torch_npu.hifloat8,
    )


__all__ = ["AscendW8A8HIF8DSDynamicLinearMethod", "apply_hif8_wo_a", "ensure_cp_compatible",
           "ensure_wo_pair_consistent", "hif8_linear_enabled", "keep_linear_on_mxfp8",
           "HIF8_LINEAR_ENV"]
