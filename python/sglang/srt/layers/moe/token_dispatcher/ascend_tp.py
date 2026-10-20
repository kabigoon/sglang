from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.srt.debug_utils import moe_trace
from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
    AllGatherFinalizeRoutingWrapper,
    NPUFinalizeRouting,
)
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    NPUMoEInitRouting_v2,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DispatcherOutputDtype,
    get_ascend_dispatcher_output_dtype,
)
from sglang.srt.runtime_context import get_parallel


class AscendTPDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_TP


class AscendTPCombineInput(NamedTuple):
    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_TP


class AscendTPDispatcher(BaseDispatcher):
    def __init__(self, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.num_experts = moe_runner_config.num_experts
        self.top_k = moe_runner_config.top_k
        self._dispatch_output: Optional[AscendTPDispatchOutput] = None

        self.quant_config: Optional[dict] = None
        self.layer_id = getattr(
            getattr(moe_runner_config, "layer", None), "layer_id", -1
        )

        # Initialise routing kernels with default (no quant config yet)
        self.set_ascend_dispatcher_output_dtype()
        moe_trace.trace(
            "ascend_tp_init",
            layer=self.layer_id,
            E=self.num_experts,
            K=self.top_k,
            dtype=self.ascend_dispatcher_output_dtype.value,
            init=type(self.init).__name__,
            finalize=type(self.finalize).__name__,
            group_list_type=self.group_list_type,
        )

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.set_ascend_dispatcher_output_dtype()
        moe_trace.trace(
            "ascend_tp_quant_config",
            layer=self.layer_id,
            cfg=quant_config,
            dtype=self.ascend_dispatcher_output_dtype.value,
            quant_mode=getattr(self.init, "quant_mode", None),
        )

        # If the quantisation is GGUF and TP is active, wrap the finalizer
        # with an all‑gather so that the dispatcher stays completely clean.
        if (
            isinstance(self.quant_config, dict)
            and self.quant_config.get("quant_type") == "gguf"
            and get_parallel().tp_size > 1
        ):
            self.finalize = AllGatherFinalizeRoutingWrapper(self.finalize, dim=-1)

    def set_ascend_dispatcher_output_dtype(self) -> None:
        """Choose init & finalize routing kernels based on quant config."""
        self.ascend_dispatcher_output_dtype = get_ascend_dispatcher_output_dtype(self)

        if self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16:
            self.init = NPUMoEInitRouting_v2(quant_mode=-1)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.INT8:
            self.init = NPUMoEInitRouting_v2(quant_mode=1)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.MXFP8:
            self.init = NPUMoEInitRouting_v2(quant_mode=MXFP8_QUANT_MODE)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        else:
            raise ValueError(
                f"Unsupported ascend_dispatcher_output_dtype: {self.ascend_dispatcher_output_dtype}"
            )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> AscendTPDispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        top_k = topk_weights.shape[-1]

        moe_trace.trace(
            "dispatch_in",
            layer=self.layer_id,
            x=hidden_states,
            ids=topk_ids,
            weights=topk_weights,
            quant_mode=getattr(self.init, "quant_mode", None),
        )

        (
            permuted_hidden_states,
            expanded_row_idx,
            expert_tokens,
            hidden_states_scale,
        ) = self.init._init_routing(
            hidden_states,
            topk_ids,
            self.num_experts,
            top_k,
        )

        moe_trace.trace(
            "dispatch_out",
            layer=self.layer_id,
            hs=permuted_hidden_states,
            scale=hidden_states_scale,
            row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            group_list_type=self.group_list_type,
        )
        moe_trace.dump(
            self.layer_id,
            "dispatch_out",
            hs=permuted_hidden_states,
            scale=hidden_states_scale,
            row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        self._dispatch_output = AscendTPDispatchOutput(
            hidden_states=permuted_hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expanded_row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            group_list_type=self.group_list_type,
        )
        return self._dispatch_output

    def combine(self, combine_input: AscendTPCombineInput) -> torch.Tensor:
        if self._dispatch_output is None:
            raise RuntimeError("combine() called before dispatch()")

        dispatch_out = self._dispatch_output

        # The finalizer (possibly wrapped with TP all‑gather) does all the work.
        final_hidden_states = self.finalize._finalize_routing(
            combine_input.hidden_states,
            topk_weights=dispatch_out.topk_weights,
            expanded_row_idx=dispatch_out.expanded_row_idx,
            topk_ids=dispatch_out.topk_ids,
        )

        moe_trace.trace(
            "combine_io",
            layer=self.layer_id,
            in_hs=combine_input.hidden_states,
            row_idx=dispatch_out.expanded_row_idx,
            out=final_hidden_states,
            finalize=type(self.finalize).__name__,
        )
        moe_trace.dump(self.layer_id, "combine_out", out=final_hidden_states)

        self._dispatch_output = None
        return final_hidden_states
