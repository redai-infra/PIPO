# Copyright (c) ModelScope Contributors. All rights reserved.
"""GRPOSglangEngine — SGLang counterpart of :class:`GRPOVllmEngine`.

Used by ``RolloutTrainerMixin`` when ``args.rollout_backend == 'sglang'`` (e.g. for
PIPO On-Policy Distillation, where vLLM cannot host the compressed-KV decoder
— see ``.agents/diary/2026-04-29-OPD-plan.md``).

Responsibilities
----------------
1. Subclass :class:`SglangEngine` (which already drives ``sgl.Engine.async_generate``).
2. Wrap each :class:`ChatCompletionResponse` returned by the parent into a
   :class:`RolloutOutput`, the protocol type expected by the rollout mixin.
3. Mirror :class:`GRPOVllmEngine`'s ``infer`` / ``async_infer`` entry-point shapes
   so :func:`RolloutTrainerMixin._engine_infer` can call either backend uniformly.

Out of scope (deferred per the OPD plan §11):
  * SGLang server (HTTP) mode for OPD — only colocate mode is acceptance-tested.
  * LoRA hot-swap on the SGLang side.
  * Streaming / async-generate path validation (only sync colocate is acceptance-tested).
"""

from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Optional, Union

from swift.metrics import Metric
from .protocol import ChatCompletionResponse, InferRequest, RequestConfig, RolloutOutput
from .sglang_engine import SglangEngine


class GRPOSglangEngine(SglangEngine):
    """Adapter around SglangEngine that returns RolloutOutput-typed results."""

    def infer(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[Any] = None,  # ignored (no LoRA hot-swap on SGLang)
    ) -> List[RolloutOutput]:
        # NB: SGLang doesn't expose an LoRA-via-adapter_request API in this code path;
        # we accept the kwarg for signature parity with GRPOVllmEngine but ignore it.
        res = super().infer(infer_requests, request_config, metrics, use_tqdm=use_tqdm)
        if not isinstance(res, list):
            res = [res]
        for i, result in enumerate(res):
            if isinstance(result, RolloutOutput):
                continue
            if not isinstance(result, ChatCompletionResponse):
                raise TypeError(
                    'GRPOSglangEngine: expected ChatCompletionResponse or RolloutOutput, '
                    f'got {type(result).__name__}.')
            res[i] = RolloutOutput(response=result)
        return res

    async def async_infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        use_tqdm: Optional[bool] = None,
        **kwargs,
    ) -> List[RolloutOutput]:
        if request_config is None:
            request_config = RequestConfig()
        assert request_config.n == 1, 'GRPOSglangEngine assumes n=1 per request.'

        tasks = [self.infer_async(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1

        prog_bar = tqdm_asyncio(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)

        async def _run(task):
            try:
                r = await task
            except Exception as e:
                if getattr(self, 'strict', True):
                    raise
                r = e
            prog_bar.update()
            self._update_metrics(r, metrics)
            return r

        # Reuse SglangEngine.batch_run helper if present; otherwise gather sequentially.
        wrapped = [_run(t) for t in tasks]
        if hasattr(self, 'batch_run'):
            res = await self.batch_run(wrapped)  # type: ignore[attr-defined]
        else:
            import asyncio
            res = await asyncio.gather(*wrapped)

        for i, r in enumerate(res):
            if isinstance(r, RolloutOutput):
                continue
            if isinstance(r, Exception):
                # Bubble up if strict; otherwise just keep the exception in the list.
                continue
            if not isinstance(r, ChatCompletionResponse):
                raise TypeError(
                    'GRPOSglangEngine.async_infer: expected ChatCompletionResponse or '
                    f'RolloutOutput, got {type(r).__name__}.')
            res[i] = RolloutOutput(response=r)
        return res
