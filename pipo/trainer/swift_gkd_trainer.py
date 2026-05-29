from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from transformers import PreTrainedModel

from swift.rlhf_trainers.gkd_trainer import GKDTrainer
from swift.trainers import disable_gradient_checkpointing
from swift.utils import get_logger
from swift.rlhf_trainers.utils import aggressive_empty_cache

logger = get_logger()


# ---------------------------------------------------------------------------
# Upstream Qwen3.5 FA2 illegal-memory-access workaround.
#
# The teacher in OPD (vanilla ``Qwen/Qwen3.5-9B``) is loaded from upstream
# ``transformers.models.qwen3_5.modeling_qwen3_5``. With ``attn_implementation=
# 'flash_attention_2'`` and ``per_device_train_batch_size=1``, the teacher's
# ``Qwen3_5TextModel.forward`` constructs an mRoPE 4D ``position_ids`` of shape
# ``[4, B, T]`` then slices to ``[3, B, T]`` and forwards it via ``**kwargs`` to
# every ``full_attention`` decoder layer. Upstream ``Qwen3_5Attention.forward``
# does NOT pop ``position_ids`` before calling the attention interface, so the
# malformed mRoPE tensor reaches ``_flash_attention_forward``. There,
# ``_is_packed_sequence(position_ids, batch_size=1)`` returns True (the mRoPE
# tensor does not match ``arange(B)``) and routes to the varlen path.
# ``_prepare_from_posids`` then computes ``cu_seqlens=[0, T, 2T, 3T]`` from
# the 3 mRoPE slices while the actual flattened query has only ``T`` tokens —
# the FA2 kernel reads past the end of the allocated GPU buffer and triggers
# ``cudaErrorIllegalAddress``. (For T<=64 the over-read still falls inside the
# allocated CUDA page and silently returns garbage, so the bug only surfaces
# at T >= ~256.)
#
# The local ``pipo/qwen3_5/modeling_qwen3_5.py::Qwen3_5Attention.forward``
# already has the fix (see its ``_attn_kw_drop`` set), so the *student* under
# FA2 works fine — only the upstream teacher needs patching. We mirror that
# fix here as a lightweight monkey-patch wrapping the upstream
# ``Qwen3_5Attention.forward`` once at trainer construction time.
#
# Reproduced and verified by ``tests/test_teacher_fa2_repro.py``.
# ---------------------------------------------------------------------------
_QWEN3_5_FA2_PATCH_ATTR = '_pipo_fa2_kwargs_drop_patched'

# Kwargs that must NEVER be forwarded to the FA2 backend; mirrors the set in
# ``pipo/qwen3_5/modeling_qwen3_5.py::Qwen3_5Attention.forward``. mRoPE
# ``position_ids`` / ``text_position_ids`` poison the ``_is_packed_sequence``
# heuristic; the ``cu_seq_lens_*`` / ``max_length_*`` keys would over-ride
# the kernel kwargs derived from the real ``attention_mask`` path.
_FA2_KWARGS_TO_DROP = frozenset({
    'position_ids',
    'text_position_ids',
    'cu_seq_lens_q',
    'cu_seq_lens_k',
    'max_length_q',
    'max_length_k',
})


def _patch_upstream_qwen3_5_attention_for_fa2() -> None:
    """Monkey-patch upstream ``Qwen3_5Attention.forward`` to drop mRoPE / varlen
    kwargs before they reach the FA2 backend. Idempotent and safe to call
    multiple times.

    No-ops cleanly if the upstream module is unavailable (e.g. user is on a
    transformers version that does not ship Qwen3.5).
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    except Exception as exc:  # pragma: no cover — older transformers
        logger.warning(
            f'[PIPOGKDTrainer] Could not import upstream Qwen3_5Attention '
            f'for FA2 monkey-patch ({exc}); teacher under FA2 may crash. '
            'If you hit "CUDA illegal memory access" inside the teacher '
            'forward, switch to ATTN_IMPL=sdpa.')
        return

    if getattr(Qwen3_5Attention, _QWEN3_5_FA2_PATCH_ATTR, False):
        return

    orig_forward = Qwen3_5Attention.forward

    def patched_forward(self, *args, **kwargs):
        for k in _FA2_KWARGS_TO_DROP:
            kwargs.pop(k, None)
        return orig_forward(self, *args, **kwargs)

    Qwen3_5Attention.forward = patched_forward
    setattr(Qwen3_5Attention, _QWEN3_5_FA2_PATCH_ATTR, True)
    logger.info(
        '[PIPOGKDTrainer] patched upstream Qwen3_5Attention.forward to '
        f'drop FA2-incompatible kwargs {sorted(_FA2_KWARGS_TO_DROP)}.')


class PIPOGKDTrainer(GKDTrainer):
    """GKD trainer specialised for PIPO students (SGLang/vLLM rollout only).

    Key differences vs. ``swift.rlhf_trainers.GKDTrainer``:

    1. ``_prepare_batch_inputs``: response-prefix fix for thinking templates,
       prompt-parity PAD insertion, even-length pad, PAD-position label masking.
    2. ``_student_compressed_logits``: trains the EXACT inference forward path
       (compressor + backbone + MTP head) so all trainable params receive grad.
    3. ``_teacher_*_pad_compacted``: vanilla teacher forward on a PAD-stripped
       sequence, with cumsum-based re-mapping back to original positions for
       JSD alignment.
    4. ``_prepare_sglang_engine`` (Phase 3): force ``enable_pipo=True`` /
       ``disable_radix_cache=True`` regardless of CLI args.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        # Apply the upstream-Qwen3.5 FA2 kwargs-drop patch BEFORE super().__init__,
        # because the parent constructs the teacher model and may eagerly run a
        # warmup forward; we want the patch in place by then.
        _patch_upstream_qwen3_5_attention_for_fa2()

        super().__init__(model, *_args, **kwargs)

        # Tokenizer's pad_token_id is the single source of truth (Qwen3.5 = 248044).
        # We assert here so a mis-configured tokenizer crashes at trainer init rather
        # than producing silently-wrong PAD-mask / compaction results downstream.
        pad_id = self.processing_class.pad_token_id
        assert pad_id is not None, (
            'PIPOGKDTrainer requires processing_class.pad_token_id to be set. '
            'For Qwen3.5 this should be 248044.')
        self._pipo_pad_id = int(pad_id)

        # Thinking templates (e.g. qwen3_5) auto-append ``response_prefix='<think>\n'``
        # to the prompt during inference. Our bug-fix path
        # (_encode_with_rollout_response) requires this prefix to be present so the
        # train-time prompt matches what SGLang fed the model during rollout. Fail fast
        # if the user wires up a non-thinking template by accident.
        response_prefix = self.template.response_prefix
        assert response_prefix, (
            'PIPOGKDTrainer requires Template.response_prefix to be non-empty '
            f'(qwen3_5 default = "<think>\\n"). Got {response_prefix!r}.')

        # Teacher must be a separate frozen model. OPSD self-distillation
        # (teacher = student-with-disabled-adapter) is NOT supported on this trainer.
        assert not self._is_self_distillation, (
            'PIPOGKDTrainer does not support self-distillation. '
            'Pass an explicit teacher_model (e.g. Qwen/Qwen3.5-9B).')

        # MTP-loss weight for the combined OPD loss (backbone JSD + w * MTP JSD).
        # Mirrors the SFT trainer's weighting (see swift_sft_trainer.py and
        # ``Qwen3_5ForCausalPIPO.mtp_loss_weight``).
        self._mtp_loss_weight = float(os.environ.get('MTP_LOSS_WEIGHT', '1.0'))

        # OPD KL-divergence mode.  When ``beta == 1.0`` we are doing reverse-KL
        # (minimise ``KL(p_student || p_teacher)`` per the OPD paper §2.2 —
        # note the OPD paper writes ``p_t`` for STUDENT and ``q_t`` for
        # TEACHER, where the subscript ``t`` is the timestep, NOT a model
        # name; we use ``s_lp`` / ``t_lp`` in code which means STUDENT /
        # TEACHER log-probs respectively). Three approximations are supported:
        #   * ``sampled`` (default) — Monte-Carlo estimate using the on-policy
        #     sampled tokens only (no full-softmax over the vocab); per-token
        #     loss ``l^sample = log p_s(y) − log p_t(y)`` (= ``s_lp − t_lp``);
        #     mean → ``KL(p_s || p_t)``.
        #   * ``topk`` — restrict the comparison to a per-position union of
        #     student top-k, teacher top-k, and the sampled label index;
        #     ``log_softmax`` is renormalised over that union. The conf-head
        #     BCE target on this path uses the SAMPLED-mode formula (ratio
        #     at the rolled-out label), not ``Σ min(p_s, p_t)``.
        #   * ``full`` — full-vocab reverse KL (``generalized_jsd_loss`` with
        #     ``topk=None``).
        # When ``beta != 1.0`` the mode is ignored and standard JSD is used.
        self._opd_kl_mode = os.environ.get('OPD_KL_MODE', 'sampled').lower()
        if self._opd_kl_mode not in ('sampled', 'topk', 'full'):
            raise ValueError(f'OPD_KL_MODE must be sampled/topk/full, got {self._opd_kl_mode}')

        # NOTE: ``OPD_TOPK_SOURCE`` is now IGNORED. ``OPD_KL_MODE=topk`` always
        # uses the UNION of student top-k, teacher top-k, and the sampled
        # label index as the per-position candidate set (with duplicate
        # slots masked to ``-inf`` before softmax). See the topk branch in
        # ``_compute_chunked_jsd_loss`` for details. The env var is still
        # read & validated for backward-compat (so existing launch scripts
        # don't crash) but its value does not affect training.
        self._opd_topk_source = os.environ.get('OPD_TOPK_SOURCE', 'teacher').lower()
        if self._opd_topk_source not in ('teacher', 'student'):
            raise ValueError(f'OPD_TOPK_SOURCE must be teacher/student, got {self._opd_topk_source}')

        self._opd_conf_loss_weight = float(os.environ.get('OPD_CONF_LOSS_WEIGHT', '1'))
        self._opd_conf_detach_inputs = os.environ.get('OPD_CONF_DETACH_INPUTS', '1') == '1'

        # When True (default), rollout-PAD label positions are filled with the
        # teacher's argmax (computed for free inside the chunked OPD loop from
        # the t_logits that's already there), so MTP / conf head no longer lose
        # supervision density to the PAD-mask. See ``_compute_loss_single``.
        self._opd_fill_pad_label = os.environ.get('OPD_FILL_PAD_LABEL', '1') == '1'

        self._teacher_load_context_depth = 0

        self._pipo_empty_cache_steps = int(
            os.environ.get('PIPO_EMPTY_CACHE_STEPS', '1'))

        # Diagnostic log so users can spot config errors quickly.
        backend = getattr(self.args, 'rollout_backend', 'vllm')
        logger.info(
            f'[PIPOGKDTrainer] initialised. pad_token_id={self._pipo_pad_id}, '
            f'rollout_backend={backend}, use_vllm={self.args.use_vllm}, lmbda={self.lmbda}, '
            f'mtp_loss_weight={self._mtp_loss_weight}, '
            f'opd_kl_mode={self._opd_kl_mode}, '
            f'opd_topk_source={self._opd_topk_source}, '
            f'opd_conf_loss_weight={self._opd_conf_loss_weight}, '
            f'opd_conf_detach_inputs={self._opd_conf_detach_inputs}, '
            f'opd_fill_pad_label={self._opd_fill_pad_label}, '
            f'empty_cache_steps={self._pipo_empty_cache_steps}.')

    @contextmanager
    def load_teacher_model_context(self):
        """Re-entrant teacher load/offload context.

        The upstream context always offloads on exit. PIPO OPD can nest the
        context because ``_compute_loss_single`` keeps the teacher ``lm_head`` for
        chunked loss computation while ``_teacher_full_logits`` also protects the
        teacher forward. Only the outermost context should move the teacher.
        """
        if not self.args.offload_teacher_model:
            yield
            return

        if self._teacher_load_context_depth == 0:
            self.load_model(self.accelerator.unwrap_model(self.teacher_model))
        self._teacher_load_context_depth += 1
        try:
            yield
        finally:
            self._teacher_load_context_depth -= 1
            if self._teacher_load_context_depth == 0:
                self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

    @contextmanager
    def _teacher_lm_head_context(self, teacher_lm_head: nn.Module, device: torch.device):
        """Keep only teacher ``lm_head`` on ``device`` for one projection.

        ``load_teacher_model_context`` moves the entire teacher backbone + head,
        which defeats the point of teacher offload during chunked OPD loss.  Once
        teacher hidden states have been materialised, the chunks only need the
        output embedding matrix, so move just that submodule to GPU and offload it
        immediately after ``t_logits`` has been produced.
        """
        if not self.args.offload_teacher_model:
            yield
            return

        teacher_lm_head.to(device)
        try:
            yield
        finally:
            teacher_lm_head.to('cpu')

    # ------------------------------------------------------------------
    # Even-length helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_to_even_length(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pad_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Pad a 2-D batch tensor to even length on the time axis.

        Appends the tokenizer's ``pad_id`` on the right.  The new position carries
        ``attention_mask=0`` and ``label=-100``, so it does not contribute to loss.
        Inputs are returned as-is when ``T`` is already even, so call sites can
        unconditionally call this.
        """
        if input_ids.size(-1) % 2 == 0:
            out = {'input_ids': input_ids}
            if attention_mask is not None:
                out['attention_mask'] = attention_mask
            if labels is not None:
                out['labels'] = labels
            return out

        device = input_ids.device
        pad_col = torch.full((input_ids.size(0), 1), pad_id, dtype=input_ids.dtype, device=device)
        new_input_ids = torch.cat([input_ids, pad_col], dim=-1)
        out: Dict[str, torch.Tensor] = {'input_ids': new_input_ids}

        if attention_mask is not None:
            zero_col = torch.zeros((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=device)
            out['attention_mask'] = torch.cat([attention_mask, zero_col], dim=-1)
        if labels is not None:
            ignore_col = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device)
            out['labels'] = torch.cat([labels, ignore_col], dim=-1)
        return out

    # ------------------------------------------------------------------
    # Override: even-length pad + PAD-position label masking
    # ------------------------------------------------------------------
    def _prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
        """Encode an on-policy training batch with strict per-sample isolation.

        Hard contract (asserted, no fallback):

        * On the on-policy training-encode path (``encode_prompt_only=False`` and
          at least one sample carries ``response_token_ids``), if the caller
          passes ``B == len(inputs) > 1`` samples we **always** encode each
          sample in its own ``B=1`` chunk and stash the per-chunk dicts under
          ``_per_micro_chunks``. This guarantees zero cross-sample padding and
          unambiguous PAD bookkeeping (every PAD in a chunk's input_ids comes
          from either the SGLang prefill pair or the prompt-parity adjustment —
          NEVER from batch alignment).
        * For ``B == 1`` or ``encode_prompt_only=True``, no chunking is needed
          (a single sample produces no cross-sample padding).

        Carrier dict layout when chunking fires:
          - ``input_ids`` / ``attention_mask`` / ``labels``: a copy of chunk[0]
            so any callers that read these (e.g. ``forward_context``,
            ``_prepare_inputs`` device transfer) see a valid single-sample batch.
          - ``_per_micro_chunks``: ``List[Dict[str, torch.Tensor]]`` of length B,
            consumed by ``compute_loss`` -> ``_compute_loss_micro_chunks``.
        """
        n = len(inputs)
        if (not encode_prompt_only) and n > 1 and any(
                d.get('response_token_ids') for d in inputs):
            chunks: List[Dict[str, torch.Tensor]] = [
                self._prepare_batch_inputs_single([sample], encode_prompt_only=False)
                for sample in inputs
            ]
            for ci, chunk in enumerate(chunks):
                assert chunk['input_ids'].size(0) == 1, (
                    f'per-sample chunk {ci} has B={chunk["input_ids"].size(0)}, expected 1.')
            carrier: Dict[str, torch.Tensor] = dict(chunks[0])
            carrier['_per_micro_chunks'] = chunks
            return carrier

        return self._prepare_batch_inputs_single(inputs, encode_prompt_only=encode_prompt_only)

    def _prepare_batch_inputs_single(
        self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
        """Single-chunk encoder: response-prefix fix + even-pad + PAD-label masking.

        Three PIPO-specific behaviors over the parent encoder:

        1. **Response-prefix fix (on-policy rollouts)**: For thinking templates
           (e.g. ``qwen3_5`` with ``response_prefix='<think>\\n'``), the SGLang/vLLM
           rollout prompt sent to the model already ends with ``<think>\\n`` (auto-
           appended by ``Template`` during inference, see
           ``base.py::_swift_encode``::``elif self.response_prefix:``). The returned
           ``response_token_ids`` therefore does **not** contain the ``<think>\\n``
           tokens — the model continues generation *after* them.

           However, ``GKDTrainer._prepare_batch_inputs`` calls
           ``replace_assistant_response_with_ids`` which throws away the
           ``message['content']`` string (which had ``<think>\\n``) and injects the
           bare ``response_token_ids`` directly after ``<|im_start|>assistant\\n``,
           skipping the ``elif self.response_prefix:`` branch (only fires when
           ``response is None``). Result: the encoded sequence ends ``...assistant\\n
           [response]`` instead of ``...assistant\\n<think>\\n[response]``, causing
           a 2-token shift between rollout-time and training-time prefixes.

           Fix: when on-policy ``response_token_ids`` is present, encode prompt-only
           first (which auto-appends ``<think>\\n``) and then concatenate the rollout
           ``response_token_ids`` ourselves. See ``_encode_with_rollout_response``.

        2. **Even-length padding**: the PIPO backbone consumes pairs of tokens,
           so ``T`` must be even. The pad token is appended (right side).
        """
        needs_response_prefix_fix = (
            not encode_prompt_only
            and any(d['response_token_ids'] for d in inputs if 'response_token_ids' in d))

        if needs_response_prefix_fix:
            encoded = self._encode_with_rollout_response(inputs)
        else:
            encoded = super()._prepare_batch_inputs(inputs, encode_prompt_only=encode_prompt_only)

        input_ids = encoded['input_ids']
        assert input_ids.dim() == 2, (
            f'Expected 2-D input_ids, got shape {tuple(input_ids.shape)}. '
            'PIPO OPD is text-only.')

        if not encode_prompt_only:
            labels = encoded['labels']
            pad_mask = (input_ids == self._pipo_pad_id)
            if int(pad_mask.sum().item()) > 0:
                encoded['labels'] = torch.where(
                    pad_mask, torch.full_like(labels, -100), labels)

        if input_ids.size(-1) % 2 != 0:
            padded = self._pad_to_even_length(
                input_ids=input_ids,
                attention_mask=encoded['attention_mask'],
                labels=encoded.get('labels'),
                pad_id=self._pipo_pad_id,
            )
            encoded.update(padded)

        assert 'position_ids' not in encoded, (
            'position_ids must not be in the encoded batch — the PIPO HF model '
            'derives them internally and our even-length pad would invalidate any '
            'pre-computed positions.')
        return encoded

    # ------------------------------------------------------------------
    # Response-prefix fix helpers
    # ------------------------------------------------------------------
    def _get_suffix_token_ids(self) -> List[int]:
        """Tokenize ``template_meta.suffix`` (str pieces) into a flat list of token ids.

        Memoized after first call. Qwen3.5's suffix is ``['<|im_end|>\\n']``; if a
        different template registers non-string suffix pieces this will assert.
        """
        cached = getattr(self, '_cached_suffix_ids', None)
        if cached is not None:
            return cached
        suffix_ids: List[int] = []
        for piece in (self.template.template_meta.suffix or []):
            assert isinstance(piece, str), (
                f'_get_suffix_token_ids only supports str suffix pieces; got {type(piece).__name__}: {piece!r}')
            suffix_ids.extend(self.template.tokenizer.encode(piece, add_special_tokens=False))
        self._cached_suffix_ids = suffix_ids
        return suffix_ids

    def _encode_with_rollout_response(self, inputs: list) -> Dict[str, torch.Tensor]:
        """Encode (prompt + response_prefix + response_token_ids [+ suffix]) per sample,
        then right-pad and stack into a batch tensor dict.

        Why we go through ``encode_prompt_only=True`` first instead of constructing
        the prompt ourselves: the parent's prompt-only path already handles BOS,
        system, multi-turn history, and the inference-only ``response_prefix`` append
        in a single, well-tested pass. We only need to do the response concatenation,
        which the parent's full-encode path gets wrong for thinking templates.

        Notes:
        * ``inputs`` is **not** mutated (we deepcopy before calling the parent).
        * SGLang/vLLM single-turn rollouts always produce ``response_token_ids =
          List[int]``. Multi-turn (``List[List[int]]``) is not supported here.
        * ``add_eos`` is read per-sample (defaults to ``True`` to match
          ``_swift_encode``). For SGLang/vLLM rollouts it is always ``False``
          (see ``rollout_mixin.py::merge_output_input_data`` setting
          ``input_data['add_eos'] = False``).
        * **Prompt parity**: if ``prompt_ids`` (after the auto-appended
          ``response_prefix='<think>\\n'``) has odd length, we append one PAD token
          to make it even, mirroring SGLang's
          ``tokenizer_manager.py::_tokenize_one_request`` L769-770 — which is what
          SGLang did to the prompt during rollout. Without this match, the PIPO
          compressor would pair the last prompt token with the first generated token
          (``response_token_ids[0]`` = the prefill's ``token1``), which violates the
          pair-boundary alignment that the SFT-trained student expects (mirrors
          ``swift_sft_trainer.py::_build_random_padded_inputs`` L155-159).
        """
        # One-time diagnostic so we can confirm in training logs that the bug-fix
        # path is actually firing for on-policy SGLang/vLLM rollouts.
        if not getattr(self, '_logged_response_prefix_fix', False):
            logger.info(
                f'[PIPOGKDTrainer] response-prefix fix active: '
                f'response_prefix={self.template.response_prefix!r}, '
                f'suffix_token_ids={self._get_suffix_token_ids()}, '
                f'pad_id={self._pipo_pad_id}.')
            self._logged_response_prefix_fix = True

        # Step 1: prompt-only encode (auto-appends ``response_prefix`` like
        # ``<think>\n``). Deepcopy because the parent mutates ``messages`` in place.
        prompt_inputs = deepcopy(inputs)
        prompt_encoded = super()._prepare_batch_inputs(prompt_inputs, encode_prompt_only=True)
        prompt_input_ids = prompt_encoded['input_ids']
        prompt_attn = prompt_encoded['attention_mask']
        assert prompt_input_ids.dim() == 2, (
            f'prompt-only encode returned non-2D input_ids: {tuple(prompt_input_ids.shape)}')

        suffix_ids = self._get_suffix_token_ids()
        pad_id = self._pipo_pad_id
        device = prompt_input_ids.device
        dtype_ids = prompt_input_ids.dtype
        dtype_attn = prompt_attn.dtype

        # Step 2: per-sample concat (prompt + parity-PAD + response + optional suffix).
        # IMPORTANT: prompt-only encoding goes through the parent's INFERENCE-mode
        # template path (``mode='transformers'``), which **left-pads** thinking-style
        # chat templates so the rightmost slot aligns with the next-token-to-generate.
        # We therefore must NOT slice the first ``valid_len`` tokens (those would be
        # the leading PADs); use a boolean mask drawn from ``attention_mask`` so the
        # extraction is correct under either padding side.
        # See user-reported bug 2026-05-13: B>=2 with left-padded prompts produced
        # mid-sequence ``PAD PAD ... <|im_start|>user ...`` inputs and bogus PAD
        # counts (e.g. 33 instead of 4 with B=4) because the leading PADs got pulled
        # into ``prompt_ids`` here, then the trailing real tokens got truncated.
        bs = len(inputs)
        sequences: List[Tuple[List[int], List[int], List[int]]] = []
        for i, data in enumerate(inputs):
            valid_mask = prompt_attn[i].bool()
            prompt_ids = prompt_input_ids[i][valid_mask].tolist()

            if len(prompt_ids) % 2 != 0:
                prompt_ids = prompt_ids + [pad_id]

            resp_ids = list(data['response_token_ids'])
            assert resp_ids and isinstance(resp_ids[0], int), (
                f'Expected single-turn response_token_ids (List[int]), got '
                f'{type(resp_ids[0]).__name__ if resp_ids else "empty"}.')

            add_eos = data.get('add_eos', True)
            eos = list(suffix_ids) if add_eos else []

            full_ids = prompt_ids + resp_ids + eos
            full_labels = [-100] * len(prompt_ids) + resp_ids + eos
            full_attn = [1] * len(full_ids)

            sequences.append((full_ids, full_attn, full_labels))

        # Step 3: right-pad to batch max length (matches default Template padding_side).
        max_len = max(len(s[0]) for s in sequences)
        input_ids = torch.full((bs, max_len), pad_id, dtype=dtype_ids, device=device)
        attention_mask = torch.zeros((bs, max_len), dtype=dtype_attn, device=device)
        labels = torch.full((bs, max_len), -100, dtype=torch.long, device=device)

        for i, (ids, am, lb) in enumerate(sequences):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=dtype_ids, device=device)
            attention_mask[i, :L] = torch.tensor(am, dtype=dtype_attn, device=device)
            labels[i, :L] = torch.tensor(lb, dtype=torch.long, device=device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    # ------------------------------------------------------------------
    # compute_loss override — PIPO needs full per-token student logits
    # ------------------------------------------------------------------
    def _student_full_logits(self, model: nn.Module, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the student backbone in *uncompressed* mode and emit full ``[B, T, V]`` logits.

        Why we cannot just call ``model(**model_inputs)``: when ``labels=None`` the
        PIPO HF forward (``Qwen3_5ForCausalPIPO.forward``) takes the
        *inference* branch and returns ``logits[:, -1:]`` (the next-token-only logits),
        which doesn't match GKDTrainer's per-position JSD-loss expectations. We bypass
        that by calling ``backbone -> lm_head`` ourselves with
        ``use_compressed_prompt=False`` semantics.

        Returned tensor has shape ``[B, T, V]`` and a real grad path back to the
        student parameters (LoRA + compressor are unaffected because we don't touch
        the compressor on this code path; only the backbone parameters drive the
        gradient — that's intentional, see the OPD plan §4.1 ``compute_loss`` note
        and the design discussion in this file's docstring).
        """
        unwrapped = self.accelerator.unwrap_model(model)
        # PEFT wraps the actual model under base_model.model. We assume student is
        # always LoRA-tuned per the OPD config.
        assert is_peft_model(unwrapped), 'PIPO OPD student must be a PEFT model.'
        base_model = unwrapped.base_model.model
        backbone_out = base_model.model(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            use_cache=False,
        )
        return base_model.lm_head(backbone_out.last_hidden_state)  # [B, T, V]

    # ------------------------------------------------------------------
    # Compressed-mode student forward — train the EXACT inference path
    # ------------------------------------------------------------------
    def _student_compressed_logits(
        self,
        model: nn.Module,
        model_inputs: Dict[str, torch.Tensor],
        return_hidden_states: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int, nn.Module]]:
        """Run student in COMPRESSED mode (matches SGLang serving exactly).

        Returns
        -------
        (backbone_logits, mtp_logits, T_padded) where:
            backbone_logits: ``[B, L-1, V]`` — backbone's per-pair predictions of
                even-position target tokens ``(t_2, t_4, ..., t_{T-2})``. Receives
                gradient through ``compressor`` + 32 backbone layers (incl. LoRA on
                ``q/k/v/o/gate/up/down_proj``) + ``lm_head`` (frozen, tied with
                ``embed_tokens``).
            mtp_logits: ``[B, L-1, V]`` — MTP head's per-pair predictions of
                odd-position target tokens ``(t_3, t_5, ..., t_{T-1})``. Receives
                gradient through ``mtp_fc`` + ``mtp_pre_fc_norm_*`` + 1 MTP decoder
                layer (incl. LoRA on its ``q/k/v/o/gate/up/down_proj``) +
                ``mtp.norm`` + ``lm_head``.
            T_padded: int — total token count after even-length padding (always
                even). Caller uses this to slice labels and teacher logits.

        When ``return_hidden_states=True``, returns
        ``(backbone_hidden, mtp_hidden, T_padded, lm_head)`` so the caller can
        perform chunked lm-head projection (avoids materialising the full
        ``[B, L-1, V]`` logits tensor at long context).

        Why compressed mode for OPD
        ---------------------------
        In SGLang serving, every token pair flows through the compressor → backbone
        path; the MTP head produces ``token2`` for each pair. Training the student
        with the SAME compressed forward (vs. an uncompressed AR forward) ensures:

        1. **Parameter coverage**: every trainable parameter (compressor +
           ``modules_to_save`` MTP norms/fc + LoRA on backbone & MTP) is in the
           gradient path of the JSD loss. Uncompressed mode trains only backbone
           LoRA + (frozen) lm_head — compressor and MTP head receive NO OPD signal.
        2. **Distribution match**: backbone weights see compressed-pair latents at
           training time, the same as at inference. No train/inference gap.
        3. **PAD-skip semantics**: with PAD tokens in the rolled-out sequence,
           pair (t_p, PAD) compressed to one latent → backbone predicts the next
           non-PAD token, exactly mirroring the runtime "MTP-skip" behaviour.

        Implementation mirrors ``Qwen3_5ForCausalPIPO.forward()`` lines 437-498
        (the ``can_compress + labels is not None`` training path) but materialises
        full logits via ``lm_head`` instead of going through chunked CE — JSD needs
        per-position distributions. Memory: ``2 * (B * (L-1) * V)`` for the two
        logit tensors + the teacher's ``B * T * V``. At ``max_length=4096``, ``V≈248K``,
        ``bf16``, this is ~6 GB total — fits comfortably alongside training.
        At ``max_length=131072`` this becomes ~130 GB and OOMs; use
        ``return_hidden_states=True`` + chunked lm-head to cap peak memory.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        assert is_peft_model(unwrapped), 'PIPO OPD student must be a PEFT model.'
        base = unwrapped.base_model.model

        # Direct attribute access — PIPO HF model defines these unconditionally
        # (see ``Qwen3_5ForCausalPIPO``). An AttributeError here means the model
        # plugin is broken; let it crash loudly rather than silently fall back.
        backbone = base.model
        lm_head = base.lm_head
        mtp = base.mtp
        embed_pad_compress = base._embed_pad_and_compress

        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        # Caller (`_prepare_batch_inputs`) is responsible for even-length padding.
        # Crash loudly if that contract is violated.
        T = input_ids.size(1)
        assert T % 2 == 0, (
            f'PIPO compressed forward requires even T; got T={T}. '
            'This is a `_prepare_batch_inputs` bug.')
        # L = T // 2  (number of pairs / latents)

        # --- Compressor + backbone (32 layers, with LoRA) ---
        compressed_embeds, pair_mask = embed_pad_compress(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        backbone_out = backbone(
            inputs_embeds=compressed_embeds,
            attention_mask=pair_mask,
            use_cache=False,
        )
        backbone_hidden = backbone_out.last_hidden_state  # [B, L, H]
        # Drop the last pair's hidden — there's no even-position target after it.
        backbone_hidden_in = backbone_hidden[:, :-1]      # [B, L-1, H]

        # --- MTP head (1 decoder layer + fc + norms, with LoRA) ---
        # Teacher-forcing token1 from the rolled-out sequence (same as the SFT path
        # in modeling_qwen3_5_mtp.py:482).
        sampled_token1 = input_ids[:, 2::2]               # [B, L-1]
        mtp_hidden = mtp(
            inputs_embeds=backbone.embed_tokens(sampled_token1),
            hidden_states=backbone_hidden_in,
            attention_mask=torch.ones_like(sampled_token1, dtype=torch.bool),
            use_cache=False,
        )                                                  # [B, L-1, H]

        # --- Project to vocab via the (tied, frozen) lm_head ---
        if return_hidden_states:
            return backbone_hidden_in, mtp_hidden, T, lm_head
        backbone_logits = lm_head(backbone_hidden_in)      # [B, L-1, V]
        mtp_logits = lm_head(mtp_hidden)                   # [B, L-1, V]
        return backbone_logits, mtp_logits, T

    # ------------------------------------------------------------------
    # PAD-compaction helper for teacher forward
    # ------------------------------------------------------------------
    def _teacher_logits_pad_compacted(
        self,
        teacher_model: nn.Module,
        model_inputs: Dict[str, torch.Tensor],
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run teacher on PAD-stripped inputs and re-pad logits to original positions.

        Why: SGLang's PIPO serving inserts PAD tokens at rejected positions.
        The vanilla Qwen3.5 teacher has never seen PAD in mid-sequence, so feeding it
        PAD makes ALL its predictions in the post-PAD prefix mildly OOD.
        We instead feed the teacher a clean (PAD-stripped) sequence and use its logits
        to supervise the student via a positional re-mapping.

        Re-mapping rule (uses ``cumsum(non_pad_mask) - 1`` as the gather index):
          * non-PAD position p → teacher logit at compacted_idx = (number of non-PADs
            up to and including p) - 1.
          * PAD position p → teacher logit at the *previous* non-PAD's compacted_idx
            (i.e. teacher's prediction conditioned on the prefix-without-PAD ending
            just before p). This makes student.logits[PAD position] (which predicts the
            next non-PAD token) align with the same target tokenwise.
          * Position right BEFORE a PAD → automatically excluded from JSD loss because
            ``shifted_labels[p] = labels[p+1] = -100`` (PAD's label was masked to -100
            in ``_prepare_batch_inputs``). So the slight mismatch at that position is
            harmless.

        Implementation is fully vectorised across batch via ``cumsum`` + ``gather``;
        for B>1 with variable PAD count per sample, the compacted sequences are
        right-padded to ``max_keep_len`` for batching (the right-pad logits are never
        gathered because gather indices stay within each sample's valid range).

        Args:
            teacher_model: the frozen teacher (vanilla Qwen3.5).
            model_inputs: ``{'input_ids', 'attention_mask', ...}`` from the encoded batch.
            pad_mask: ``[B, T]`` bool tensor — True where input_ids == pad_id.

        Returns:
            ``[B, T, V]`` teacher logits aligned to the original (PAD-containing)
            position indexing. Detached / no-grad.
        """
        input_ids = model_inputs['input_ids']  # [B, T]
        attn = model_inputs['attention_mask']  # [B, T]
        device = input_ids.device
        non_pad = ~pad_mask  # [B, T]
        n_keep = non_pad.sum(dim=1)  # [B]
        max_keep = int(n_keep.max().item())
        # Caller guarantees n_pad > 0 in `_teacher_full_logits`, so max_keep > 0
        # iff at least one row has a non-PAD token. An all-PAD batch would be a bug
        # upstream (`_prepare_batch_inputs` always emits a real prompt); fail loud.
        assert max_keep > 0, 'PAD-compaction received an all-PAD batch.'

        # --- Build compacted batch via gather ---
        # Per-row sort: stable argsort of non_pad (descending) keeps the original
        # order of non-PAD positions and pushes PAD positions to the right.
        # ``stable=True`` is required so we preserve in-order positions.
        sort_idx = torch.argsort(non_pad.long(), dim=1, descending=True, stable=True)  # [B, T]
        sort_idx_keep = sort_idx[:, :max_keep]  # [B, max_keep]

        compacted_ids = torch.gather(input_ids, dim=1, index=sort_idx_keep)
        # Build attention_mask for the compacted batch: 1 for valid (within n_keep[b]),
        # 0 for the right-padded slots. Multiply by gathered original attn so any
        # already-zero positions stay zero.
        col = torch.arange(max_keep, device=device).unsqueeze(0)  # [1, max_keep]
        compacted_attn = torch.gather(attn.long(), dim=1, index=sort_idx_keep) * (
            col < n_keep.unsqueeze(1)).long()  # [B, max_keep]

        # --- Teacher forward ---
        with torch.no_grad():
            teacher_out = teacher_model(
                input_ids=compacted_ids,
                attention_mask=compacted_attn,
                use_cache=False,
            )
        teacher_logits_compact = teacher_out.logits  # [B, max_keep, V]
        V = teacher_logits_compact.size(-1)

        # --- Re-map back to original positions ---
        # gather_idx[b, p] = compacted index whose teacher prediction aligns with
        # original position p. For non-PAD pos: cumsum(non_pad)-1. For PAD pos: same
        # formula gives the PREVIOUS non-PAD's compacted idx (because cumsum doesn't
        # advance at PAD). We clamp(min=0) to handle the rare case of leading PAD.
        gather_idx = (non_pad.long().cumsum(dim=1) - 1).clamp(min=0)  # [B, T]
        gather_idx_v = gather_idx.unsqueeze(-1).expand(-1, -1, V)  # [B, T, V]
        teacher_logits_repad = torch.gather(teacher_logits_compact, dim=1, index=gather_idx_v)
        return teacher_logits_repad

    def _teacher_hidden_pad_compacted(
        self,
        teacher_model: nn.Module,
        model_inputs: Dict[str, torch.Tensor],
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, nn.Module]:
        """Hidden-states variant of :meth:`_teacher_logits_pad_compacted`.

        Runs the teacher's **backbone** (no lm_head) on the PAD-stripped sequence
        and re-maps the resulting hidden states back to the original (PAD-containing)
        positions. Returns ``(hidden_repad, teacher_lm_head)`` so the caller can chunk
        the lm-head projection downstream — mirroring the non-compacted hidden-states
        path in :meth:`_teacher_full_logits` when ``return_hidden_states=True``.

        Why this mirror exists: the chunked lm-head path (controlled by
        ``PIPO_OPD_CHUNK_SIZE``) is the default for long-context OPD, but the
        original PAD-compaction code only existed for the full-logits path. Without
        this method, the chunked path would have to fall back to the non-chunked
        logits path whenever PADs are present — i.e. always for SGLang rollouts
        (prefill always emits one PAD per sample). That fallback would materialise
        ``[B, T, V]`` teacher logits and could OOM at 128K context.
        """
        # Direct attribute access — vanilla Qwen3.5 always exposes ``.model``
        # (text backbone) and ``.lm_head``. Multimodal shells are out of scope.
        teacher_unwrapped = self.accelerator.unwrap_model(teacher_model)
        teacher_backbone = teacher_unwrapped.model
        teacher_lm_head = teacher_unwrapped.lm_head

        input_ids = model_inputs['input_ids']  # [B, T]
        attn = model_inputs['attention_mask']  # [B, T]
        device = input_ids.device
        non_pad = ~pad_mask  # [B, T]
        n_keep = non_pad.sum(dim=1)  # [B]
        max_keep = int(n_keep.max().item())
        assert max_keep > 0, 'PAD-compaction received an all-PAD batch.'

        # --- Build compacted batch via stable argsort + gather ---
        # ``stable=True`` preserves in-order non-PAD positions; PADs are pushed right
        # and dropped by the [:, :max_keep] slice below.
        sort_idx = torch.argsort(non_pad.long(), dim=1, descending=True, stable=True)  # [B, T]
        sort_idx_keep = sort_idx[:, :max_keep]  # [B, max_keep]

        compacted_ids = torch.gather(input_ids, dim=1, index=sort_idx_keep)
        col = torch.arange(max_keep, device=device).unsqueeze(0)  # [1, max_keep]
        compacted_attn = torch.gather(attn.long(), dim=1, index=sort_idx_keep) * (
            col < n_keep.unsqueeze(1)).long()  # [B, max_keep]

        # --- Teacher BACKBONE-only forward (no lm_head; caller chunks it later) ---
        with torch.no_grad():
            backbone_out = teacher_backbone(
                input_ids=compacted_ids,
                attention_mask=compacted_attn,
                use_cache=False,
            )
        hidden_compact = backbone_out.last_hidden_state  # [B, max_keep, H]
        H = hidden_compact.size(-1)

        # --- Re-map to original positions ---
        # See ``_teacher_logits_pad_compacted`` for the full re-mapping rationale.
        # Briefly: gather_idx[b, p] = (cumsum(non_pad[b])[p] - 1).clamp(0). For
        # non-PAD position p, this points to the SAME token in compacted indexing.
        # For PAD position p, it points to the PREVIOUS non-PAD token (because
        # cumsum doesn't advance at PAD), which gives the teacher hidden conditioned
        # on the prefix-without-PAD — exactly aligned with what the PIPO
        # student saw at the compressed pair containing this PAD.
        gather_idx = (non_pad.long().cumsum(dim=1) - 1).clamp(min=0)  # [B, T]
        gather_idx_h = gather_idx.unsqueeze(-1).expand(-1, -1, H)  # [B, T, H]
        hidden_repad = torch.gather(hidden_compact, dim=1, index=gather_idx_h)  # [B, T, H]
        return hidden_repad, teacher_lm_head

    # ------------------------------------------------------------------
    # Sampled-token Reverse KL helper (dead-code; main path uses
    # `_compute_chunked_sampled_kl`. Kept as a clean reference impl.)
    # ------------------------------------------------------------------
    def _compute_reverse_kl_sampled_token(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sampled-token reverse KL ``KL(p_s ‖ p_t)`` (OPD paper §2.2).

        For each valid position (``labels != -100``) we compute the per-token
        Monte-Carlo estimator from the OPD paper:

            l^sample_t  =  log p_s(ŷ_t) − log p_t(ŷ_t)

        and return the mean. ``E_{ŷ_t∼p_s}[l^sample_t] = D_KL(p_s ‖ p_t)``,
        so minimising the mean pulls the student towards the teacher.

        Args:
            student_logits: ``[B, S, V]`` logits from the student (p_s).
            teacher_logits: ``[B, S, V]`` logits from the frozen teacher (p_t).
            labels: ``[B, S]`` label tensor; positions with ``-100`` are ignored.
            temperature: softmax temperature (shared with the parent trainer).

        Returns:
            Scalar loss tensor (mean over all valid positions).
        """
        mask = labels != -100
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        sampled_token_ids = labels[mask]

        if student_logits.numel() == 0:
            return student_logits.new_zeros(())

        student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_logprobs = F.log_softmax(teacher_logits / temperature, dim=-1)

        sampled_token_ids = sampled_token_ids.unsqueeze(-1)  # [N, 1]
        student_token_lp = torch.gather(student_logprobs, -1, sampled_token_ids).squeeze(-1)
        teacher_token_lp = torch.gather(teacher_logprobs, -1, sampled_token_ids).squeeze(-1)

        # OPD paper §2.2: l^sample = log p_s(y) − log p_t(y); mean → KL(p_s ‖ p_t).
        loss = (student_token_lp - teacher_token_lp).mean()
        return loss

    # ------------------------------------------------------------------
    # Teacher logits helper — single entry point with optional PAD-compaction
    # ------------------------------------------------------------------
    def _teacher_full_logits(
        self,
        model_inputs: Dict[str, torch.Tensor],
        pad_mask: torch.Tensor,
        n_pad: int,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, nn.Module]]:
        """Run vanilla teacher (Qwen3.5) and return logits or (hidden, lm_head).

        Two paths:
          1. PAD-compaction + non-zero PAD count → run on compacted seq, re-map
             back to original positions for JSD alignment.
          2. Otherwise → straight teacher forward.

        When ``return_hidden_states=True``, returns ``(last_hidden_state, lm_head)``
        so the caller can chunk the lm-head projection and avoid materialising the
        full ``[B, T, V]`` tensor (critical for long-context OPD).
        """
        load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
        t_fwd = {k: v for k, v in model_inputs.items() if k != 'labels'}

        # PAD-compaction path: the gold rule for SGLang rollout (PAD always present
        # in response_token_ids[1] from the prefill).
        if n_pad > 0:
            with load_context, disable_gradient_checkpointing(
                    self.teacher_model, self.args.gradient_checkpointing_kwargs):
                if return_hidden_states:
                    return self._teacher_hidden_pad_compacted(
                        self.teacher_model, model_inputs, pad_mask)
                return self._teacher_logits_pad_compacted(
                    self.teacher_model, model_inputs, pad_mask)

        # No-compaction path. PAD-free input (off-policy or pad_compaction=0).
        with torch.no_grad(), load_context, disable_gradient_checkpointing(
                self.teacher_model, self.args.gradient_checkpointing_kwargs):
            if return_hidden_states:
                teacher_unwrapped = self.accelerator.unwrap_model(self.teacher_model)
                # Vanilla Qwen3.5 teacher: ``.model`` is the backbone, ``.lm_head``
                # the output head. Direct attribute access — fail loudly if either
                # is missing (means the user passed a multimodal shell or a non-
                # Qwen3.5 teacher we can't introspect).
                teacher_out = teacher_unwrapped.model(
                    input_ids=t_fwd['input_ids'],
                    attention_mask=t_fwd['attention_mask'],
                    use_cache=False,
                )
                return teacher_out.last_hidden_state, teacher_unwrapped.lm_head
            teacher_out = self.teacher_model(**t_fwd)
            return teacher_out.logits  # [B, T_orig, V]

    # ------------------------------------------------------------------
    # Chunked JSD / sampled-KL helpers (avoid materialising full [B, T, V])
    # ------------------------------------------------------------------
    def _accumulate_conf_chunk(
        self,
        confidence_head: nn.Module,
        backbone_hidden: torch.Tensor,
        mtp_hidden: torch.Tensor,
        conf_target: torch.Tensor,
        mask: torch.Tensor,
        detach_inputs: bool,
        accum: Dict[str, float],
        loss_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run conf head + BCE on one chunk; mutate ``accum`` with diagnostics.

        ``conf_target`` is the per-position **EAGLE-style accept-rate** target
        (``[B, chunk]``, values in ``[0, 1]``, already detached). The caller
        is responsible for constructing it according to the OPD mode:

          * sampled-KL path → per-token accept rate
            ``α(y) = min(1, p_t(y) / p_s(y))`` evaluated at the sampled label.
          * full-vocab / topk JSD path → expected accept rate at this position
            ``Σ_y min(p_s(y), p_t(y)) = 1 - TV(p_s, p_t)`` (restricted to the
            top-k subset under topk mode, which matches the serving-time
            top-k draft).

        Both forms are direct EAGLE-accept-rate proxies: the inference-time
        SGLang Phase-2 commit-vs-PAD gate consumes ``sigmoid(ConfHead)`` as a
        per-pair accept probability, and we want training/inference semantics
        to agree without any extra calibration.

        ``mask`` selects the valid (non-PAD, label != -100) positions; only
        those contribute to BCE and the metric counters.

        The returned tensor is the chunk's BCE *sum* (NOT mean) — the caller
        normalises by ``num_valid_conf`` after the loop, mirroring the OPD
        loss aggregation pattern.
        """
        # Conf head input — detach() guards against the degenerate solution
        # where MTP sacrifices token2 quality to make itself easier to predict.
        if detach_inputs:
            s_back_h_chunk = backbone_hidden.detach()
            s_mtp_h_chunk = mtp_hidden.detach()
        else:
            s_back_h_chunk = backbone_hidden
            s_mtp_h_chunk = mtp_hidden

        conf_logit = confidence_head(s_back_h_chunk, s_mtp_h_chunk)  # [B, chunk]

        conf_logit_flat = conf_logit[mask]
        target_flat = conf_target[mask].to(conf_logit_flat.dtype)
        chunk_loss = F.binary_cross_entropy_with_logits(
            conf_logit_flat, target_flat, reduction='sum'
        ).to(loss_dtype)

        with torch.no_grad():
            pred_p = torch.sigmoid(conf_logit_flat)
            accum['_pred_sum'] += float(pred_p.sum().item())
            accum['_target_sum'] += float(target_flat.sum().item())
            # accum['_pred_pos_count'] += int((pred_p > 0.5).sum().item())
            # accum['_target_pos_count'] += int((target_flat > 0.5).sum().item())
            accum['_num_valid'] += int(mask.sum().item())

        return chunk_loss

    @staticmethod
    def _finalise_conf(
        total_conf_loss: torch.Tensor,
        accum: Dict[str, float],
        out_dtype: torch.dtype,
        anchor_for_zero_grad: torch.Tensor,
        confidence_head: Optional[nn.Module] = None,
        ghost_backbone_hidden: Optional[torch.Tensor] = None,
        ghost_mtp_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Convert chunked accumulators into a (loss, metrics) pair.

        When zero valid positions exist (e.g. all-PAD mtp targets — common
        with high ``PIPO_CONF_THRESHOLD`` when many ``token2`` slots
        were already PAD-replaced during rollout), the chunk loop never
        called ``confidence_head``. Returning a 0-grad scalar anchored only
        to ``anchor_for_zero_grad`` (the student backbone hidden) leaves
        ``confidence_head.{norm, fc1, fc2}`` OUTSIDE the autograd graph on
        this rank — but other ranks (whose batch did contain valid
        positions) put those same params INTO the graph. DeepSpeed ZeRO-2
        then waits forever for the missing reduce-scatter contributions on
        the unlucky ranks → all ranks hang at the gradient ALLREDUCE for
        the conf-head bucket → eventual NCCL timeout (or peer-memory
        corruption under heavy memory pressure at long context).

        Fix: when ``confidence_head`` and the ``ghost_*`` tensors are
        supplied (i.e. the caller intended to train the head this step) AND
        ``n == 0``, run a tiny detached ghost forward through the head and
        scale by ``0`` so the loss VALUE is unchanged but every conf-head
        parameter ends up with a zero (rather than ``None``) gradient,
        keeping the bucket aligned across ranks.
        """
        n = accum['_num_valid']
        if n == 0:
            base_zero = anchor_for_zero_grad.sum() * 0.0
            if (confidence_head is not None
                    and ghost_backbone_hidden is not None
                    and ghost_mtp_hidden is not None):
                # ``[:, :1]`` keeps the leading dims of the head's expected
                # input shape ``[B, S, H]`` while costing only 1 token of
                # compute. ``.detach()`` mirrors the standard conf-input
                # detach contract; multiplying by ``0.0`` zeroes out any
                # NaN/Inf in ``ghost_*`` and contributes nothing to the
                # loss value. The result is a real autograd path through
                # ``confidence_head.{norm, fc1, fc2}`` so DDP/ZeRO-2 sees
                # a (zero-valued) gradient for every conf-head param.
                dummy_back = ghost_backbone_hidden[:, :1].detach() * 0.0
                dummy_mtp = ghost_mtp_hidden[:, :1].detach() * 0.0
                ghost = (confidence_head(dummy_back, dummy_mtp).sum()
                         * 0.0).to(base_zero.dtype)
                base_zero = base_zero + ghost
            return base_zero.to(out_dtype), {
                'opd_conf_loss': 0.0,
                'opd_conf_mean_pred': 0.0,
                'opd_conf_mean_target': 0.0,
                'opd_conf_pred_pos_rate': 0.0,
                'opd_conf_target_pos_rate': 0.0,
                'opd_conf_num_valid': 0.0,
            }
        loss = (total_conf_loss / n).to(out_dtype)
        return loss, {
            'opd_conf_loss': float(loss.detach().item()),
            'opd_conf_mean_pred': accum['_pred_sum'] / n,
            'opd_conf_mean_target': accum['_target_sum'] / n,
            # 'opd_conf_pred_pos_rate': accum['_pred_pos_count'] / n,
            # 'opd_conf_target_pos_rate': accum['_target_pos_count'] / n,
            'opd_conf_num_valid': float(n),
        }

    def _compute_chunked_jsd_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_lm_head: nn.Module,
        teacher_lm_head: nn.Module,
        labels: torch.Tensor,
        chunk_size: int,
        temperature: float,
        beta: float,
        topk: Optional[int] = None,
        topk_source: str = 'student',  # noqa: ARG002 — accepted for backward-compat only; union-topk path now ignores it (see body comments).
        # ── Confidence head co-supervision (optional) ──
        # When ``confidence_head`` is given, we compute per-position OPD loss
        # (instead of a flattened-then-sum reduction) so the SAME log-prob
        # tensors feed both the OPD loss aggregator and the conf-head BCE
        # target. This avoids a second lm-head forward (which is ~10× more
        # expensive than the rest of the chunk for V≈248K, so the saving is
        # substantial). Caller passes ``backbone_hidden_for_conf`` aligned to
        # ``student_hidden`` (both ``[B, S, H]``); the head sees pairs
        # ``(backbone_pos, student_pos)`` per chunk.
        confidence_head: Optional[nn.Module] = None,
        backbone_hidden_for_conf: Optional[torch.Tensor] = None,
        detach_conf_inputs: bool = True,
        # When provided, positions where ``fill_label_mask`` is True have their
        # ``-100`` label replaced with ``s_logits.argmax(-1)`` per chunk — i.e.
        # a deterministic proxy of the student-drafted token (``ŷ ~ p_s``), so
        # both the reverse-KL estimator and the conf-head BCE target stay on
        # the same student-sample distribution used at EAGLE serving time.
        fill_label_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """Compute JSD / reverse-KL loss by chunking the lm-head projection.

        Peak memory is ``O(chunk_size * V)`` instead of ``O(S * V)``. The
        per-chunk reduction is ``per_pos[mask].sum()`` so the SAME log-prob
        tensors feed both the OPD aggregator and the conf-head BCE target
        (avoiding a second lm-head forward; lm-head is the dominant per-chunk
        cost at V≈248K).

        Candidate set selection (per chunk, per position):

          * ``topk is None``       — full vocab.
          * ``topk is not None``   — **union** of student top-k, teacher top-k,
                                     and the sampled label index (B=1 only).
                                     Duplicate slots are masked with ``-inf``
                                     so the softmax sees only unique candidates.
                                     ``topk_source`` is accepted for backward
                                     compat but IGNORED in this path.

        Conf-head BCE target (per chunk, per position) — derived from the
        same log-prob tensors as the KL loss:

          * full vocab           → ``Σ_y min(p_s(y), p_t(y)) = 1 − TV(p_s, p_t)``.
          * union-topk           → SAMPLED-mode formula (matches
            :meth:`_compute_chunked_sampled_kl`):
            ``α(y) = exp(-max(0, log p_s(y) − log p_t(y)))`` where ``y`` is
            the rolled-out label and ``log p_*`` are the union-renormalised
            log-probs read at the label's slot inside ``union_idx``.

        Returns
        -------
        ``(opd_loss, conf_loss_or_None, conf_metrics)``
            * ``opd_loss``         — scalar OPD loss.
            * ``conf_loss_or_None``— scalar BCE if ``confidence_head`` was given,
                                     else ``None`` (caller should not add it to
                                     the total).
            * ``conf_metrics``     — diagnostic dict; empty when conf is off.
        """
        B, S, _ = student_hidden.shape
        assert B == teacher_hidden.size(0) == labels.size(0)
        assert S == teacher_hidden.size(1) == labels.size(1)
        if confidence_head is not None:
            assert backbone_hidden_for_conf is not None, (
                '_compute_chunked_jsd_loss: backbone_hidden_for_conf must be '
                'provided alongside confidence_head.')
            assert backbone_hidden_for_conf.shape == student_hidden.shape, (
                f'backbone_hidden_for_conf shape {tuple(backbone_hidden_for_conf.shape)} '
                f'must match student_hidden {tuple(student_hidden.shape)}.')

        total_loss = student_hidden.new_zeros((), dtype=torch.float32)
        num_valid = 0
        # Conf accumulators — see ``_accumulate_conf_chunk`` for fields.
        total_conf_loss = student_hidden.new_zeros((), dtype=torch.float32)
        conf_accum: Dict[str, float] = {
            '_pred_sum': 0.0, '_target_sum': 0.0,
            # '_pred_pos_count': 0, '_target_pos_count': 0,
            '_num_valid': 0,
        }

        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            s_h = student_hidden[:, start:end]
            t_h = teacher_hidden[:, start:end]
            lbl = labels[:, start:end]

            s_logits = student_lm_head(s_h)
            with self._teacher_lm_head_context(teacher_lm_head, t_h.device):
                t_logits = teacher_lm_head(t_h)

            if fill_label_mask is not None:
                fm = fill_label_mask[:, start:end]
                if fm.any():
                    lbl = torch.where(fm, s_logits.argmax(dim=-1), lbl)

            # ── Build the candidate index set for this chunk ──
            # In ``topk`` mode we now take the UNION of (a) student top-k,
            # (b) teacher top-k, and (c) the sampled label index, then
            # renormalise ``log_softmax`` over that union. ``topk_source``
            # is therefore IGNORED here (kept in the signature only for
            # backward-compatible call sites). Rationale:
            #
            #   * union(s, t) gives a richer candidate set than either side
            #     alone — high-disagreement positions have ~2K candidates,
            #     well-agreed positions collapse back near K (via dedup).
            #   * Including the sampled label guarantees the conf-head BCE
            #     target (computed below from the SAME log-prob tensors as
            #     the KL loss) can always read s_lp / t_lp at the actually
            #     rolled-out token, with no fallback / no second lm-head
            #     forward.
            #
            # Restricted to B=1 because the per-position equality matrix
            # for dedup is ``[B, chunk, M, M]``; with B=1, M ≤ 2K+1 ≈ 65,
            # this is a few MB per chunk. ``_compute_loss_single`` always
            # invokes us with B=1 (per-sample micro-chunking guarantees it).
            if topk is not None:
                assert s_logits.size(0) == 1, (
                    '_compute_chunked_jsd_loss: union-topk path requires B=1 '
                    '(_compute_loss_single guarantees this).')
                _, s_top_idx = torch.topk(s_logits, k=topk, dim=-1)  # [1, chunk, K]
                _, t_top_idx = torch.topk(t_logits, k=topk, dim=-1)  # [1, chunk, K]
                # Append the sampled label index so conf-head BCE can read
                # s_lp / t_lp at the rolled-out token even when it lies
                # outside the natural top-k of both sides. ``clamp(min=0)``
                # turns ``-100`` (masked positions) into a valid gather idx;
                # those positions are filtered by ``mask`` downstream, so the
                # garbage value is harmless.
                label_idx = lbl.clamp(min=0).unsqueeze(-1).to(s_top_idx.dtype)
                union_idx = torch.cat([s_top_idx, t_top_idx, label_idx], dim=-1)  # [1, chunk, 2K+1]
                M = union_idx.size(-1)

                # Per-position dedup: mark each repeat after its first
                # occurrence with True. We then ``masked_fill`` those slots
                # with ``-inf`` in the gathered logits so they contribute
                # zero mass under softmax and the renormalisation reflects
                # only the unique candidate set.
                eq = union_idx.unsqueeze(-1) == union_idx.unsqueeze(-2)  # [1, chunk, M, M]
                tri = torch.tril(
                    torch.ones(M, M, dtype=torch.bool, device=union_idx.device),
                    diagonal=-1)
                is_dup = (eq & tri).any(dim=-1)  # [1, chunk, M]

                s_topk = torch.gather(s_logits, dim=-1, index=union_idx) / temperature
                t_topk = torch.gather(t_logits, dim=-1, index=union_idx) / temperature
                neg_inf = torch.finfo(s_topk.dtype).min
                s_topk = s_topk.masked_fill(is_dup, neg_inf)
                t_topk = t_topk.masked_fill(is_dup, neg_inf)
            else:
                union_idx = None
                s_topk = s_logits / temperature
                t_topk = t_logits / temperature

            mask = lbl != -100
            if not mask.any():
                del s_logits, t_logits, s_topk, t_topk
                continue

            s_log_probs = F.log_softmax(s_topk, dim=-1)  # [B, chunk, V or M]
            t_log_probs = F.log_softmax(t_topk, dim=-1)

            # ── Per-position OPD loss (numerically equivalent to old F.kl_div) ──
            # Original semantics preserved exactly:
            #   beta == 1.0  ⇒ ``F.kl_div(t_lp, s_lp, log_target=True).sum()``
            #                   = ``s_lp.exp() * (s_lp - t_lp)`` summed over V.
            #   beta == 0.0  ⇒ ``F.kl_div(s_lp, t_lp, log_target=True).sum()``
            #                   = ``t_lp.exp() * (t_lp - s_lp)`` summed over V.
            #   else (beta-JSD) ⇒ mixture-based decomposition; matches the prior
            #                   ``beta_t * kl_teacher + (1 - beta_t) * kl_student``.
            if beta == 1.0:
                per_pos_loss = (s_log_probs.exp() * (s_log_probs - t_log_probs)).sum(-1)
            elif beta == 0.0:
                per_pos_loss = (t_log_probs.exp() * (t_log_probs - s_log_probs)).sum(-1)
            else:
                beta_t = torch.tensor(beta, dtype=s_log_probs.dtype, device=s_log_probs.device)
                log_beta = torch.log(beta_t)
                log_1_minus_beta = torch.log1p(-beta_t)
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )  # [B, chunk, V/K]
                kl_teacher_per = (t_log_probs.exp() * (t_log_probs - mixture_log_probs)).sum(-1)
                kl_student_per = (s_log_probs.exp() * (s_log_probs - mixture_log_probs)).sum(-1)
                per_pos_loss = beta_t * kl_teacher_per + (1 - beta_t) * kl_student_per

            chunk_loss = per_pos_loss[mask].sum()
            total_loss = total_loss + chunk_loss
            num_valid += int(mask.sum().item())

            # ── Conf head BCE (if requested) — EAGLE-AR target derived per chunk ──
            # Two branches, matching the OPD KL flavour:
            #
            #   * ``topk`` (union-topk mode) → use the SAMPLED-token ratio
            #     formula (same as ``_compute_chunked_sampled_kl``):
            #         α(y) = min(1, p_t(y) / p_s(y))
            #              = exp(-max(0, log p_s(y) − log p_t(y)))
            #     where ``y`` is the rolled-out label. ``s_log_probs`` /
            #     ``t_log_probs`` are read at the label's position INSIDE
            #     ``union_idx`` — i.e. they are the topk-renormalised log
            #     probs at the sampled token, matching the user-supplied
            #     pseudo-code (``s_lp_sampled = s_lp[topk_indices == y]``).
            #     Because we appended the label index to ``union_idx``, the
            #     label is guaranteed present; if it ALSO appears in the
            #     student/teacher topk slice (the common case) ``argmax``
            #     on the equality mask returns the FIRST (non-duplicate)
            #     occurrence, so the gathered log-prob is the live one
            #     rather than the masked ``-inf`` duplicate.
            #
            #   * full-vocab (``topk is None``) → unchanged: expected
            #     accept rate ``Σ_y min(p_s(y), p_t(y)) = 1 − TV(p_s, p_t)``.
            #
            # Both forms are detached so BCE never leaks gradient back into
            # the student logits via the target.
            if confidence_head is not None:
                if topk is not None:
                    match = (union_idx == lbl.clamp(min=0).unsqueeze(-1))  # [1, chunk, M]
                    first_match = match.float().argmax(dim=-1, keepdim=True)  # [1, chunk, 1]
                    s_lp_sampled = s_log_probs.gather(-1, first_match).squeeze(-1)  # [1, chunk]
                    t_lp_sampled = t_log_probs.gather(-1, first_match).squeeze(-1)  # [1, chunk]
                    conf_target = torch.exp(
                        -(s_lp_sampled - t_lp_sampled).clamp(min=0.0)
                    ).detach()                                              # [1, chunk]
                else:
                    conf_target = torch.minimum(
                        s_log_probs.exp(), t_log_probs.exp()
                    ).sum(-1).clamp(0.0, 1.0).detach()                       # [B, chunk]
                conf_chunk = self._accumulate_conf_chunk(
                    confidence_head=confidence_head,
                    backbone_hidden=backbone_hidden_for_conf[:, start:end],
                    mtp_hidden=s_h,
                    conf_target=conf_target,
                    mask=mask,
                    detach_inputs=detach_conf_inputs,
                    accum=conf_accum,
                    loss_dtype=total_conf_loss.dtype,
                )
                total_conf_loss = total_conf_loss + conf_chunk

            del s_logits, t_logits, s_topk, t_topk, s_log_probs, t_log_probs, per_pos_loss

        if num_valid == 0:
            # Stay connected to the student computation graph so the returned 0
            # carries a valid (zero-valued) gradient path. ``new_zeros(())`` would
            # produce a leaf tensor with no grad — when paired with a 0-grad sibling
            # in ``loss = loss_back + w * loss_mtp``, DeepSpeed's
            # ``engine.backward(loss)`` raises "loss must be a scalar tensor" because
            # the autograd graph degenerates.
            opd_loss = (student_hidden.sum() * 0.0).to(student_hidden.dtype)
        else:
            opd_loss = (total_loss / num_valid).to(student_hidden.dtype)

        if confidence_head is None:
            return opd_loss, None, {}
        # Pass the head + 1-token ghost inputs into ``_finalise_conf`` so the
        # ``n == 0`` branch can run a no-op forward through the head and
        # keep its params in the autograd graph — see the docstring there
        # for the full rationale (DDP/ZeRO-2 grad-bucket alignment).
        conf_loss, conf_metrics = self._finalise_conf(
            total_conf_loss=total_conf_loss,
            accum=conf_accum,
            out_dtype=student_hidden.dtype,
            anchor_for_zero_grad=student_hidden,
            confidence_head=confidence_head,
            ghost_backbone_hidden=backbone_hidden_for_conf,
            ghost_mtp_hidden=student_hidden,
        )
        return opd_loss, conf_loss, conf_metrics

    def _compute_chunked_sampled_kl(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_lm_head: nn.Module,
        teacher_lm_head: nn.Module,
        labels: torch.Tensor,
        chunk_size: int,
        temperature: float,
        # ── Confidence head co-supervision (optional; same contract as JSD) ──
        confidence_head: Optional[nn.Module] = None,
        backbone_hidden_for_conf: Optional[torch.Tensor] = None,
        detach_conf_inputs: bool = True,
        # Same contract as in ``_compute_chunked_jsd_loss``: when provided,
        # rollout-PAD label positions are replaced with ``s_logits.argmax(-1)``
        # so the reverse-KL estimator stays on the student-sample distribution
        # (using ``y ~ p_t`` here would flip the gradient sign of the MC
        # estimator: E_{y~p_t}[log p_s − log p_t] = −KL(p_t ‖ p_s)).
        fill_label_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """Compute sampled-token reverse KL by chunking the lm-head projection.

        Per-token unbiased Monte-Carlo estimator of reverse KL (matching the
        formulation in the OPD paper, §2.2 Sampled-Token OPD):

            l^sample_t  ≜  log p_s(ŷ_t) − log p_t(ŷ_t)
                        =  s_lp − t_lp                          # = per_pos_loss

            E_{ŷ_t∼p_s}[l^sample_t]  =  D_KL(p_s ‖ p_t)

        Minimising the mean of ``per_pos_loss`` over sampled tokens is an
        unbiased estimator of the per-position reverse KL — pulling the
        student towards the teacher.

        Same return contract as :meth:`_compute_chunked_jsd_loss` — when
        ``confidence_head`` is given, we derive the EAGLE per-token accept
        rate at the sampled label as the BCE target:

            α(y)  =  min(1, p_t(y) / p_s(y))
                  =  exp( min(0, log p_t(y) − log p_s(y)) )
                  =  exp( -max(0, per_pos_loss) )
                  =  exp( -per_pos_loss.clamp(min=0) )

        Clamp is applied BEFORE exp for bf16 stability (avoids exp of any
        large-magnitude operand when student strongly prefers y, i.e. large
        positive ``per_pos_loss``).
        """
        B, S, _ = student_hidden.shape
        if confidence_head is not None:
            assert backbone_hidden_for_conf is not None, (
                '_compute_chunked_sampled_kl: backbone_hidden_for_conf must be '
                'provided alongside confidence_head.')
            assert backbone_hidden_for_conf.shape == student_hidden.shape, (
                f'backbone_hidden_for_conf shape {tuple(backbone_hidden_for_conf.shape)} '
                f'must match student_hidden {tuple(student_hidden.shape)}.')

        total_loss = student_hidden.new_zeros((), dtype=torch.float32)
        num_valid = 0
        total_conf_loss = student_hidden.new_zeros((), dtype=torch.float32)
        conf_accum: Dict[str, float] = {
            '_pred_sum': 0.0, '_target_sum': 0.0,
            # '_pred_pos_count': 0, '_target_pos_count': 0,
            '_num_valid': 0,
        }

        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            s_h = student_hidden[:, start:end]
            t_h = teacher_hidden[:, start:end]
            lbl = labels[:, start:end]

            s_logits = student_lm_head(s_h)

            # Fill rollout-PAD labels with the student argmax (deterministic
            # proxy of ``ŷ ~ p_s``) — no teacher forward needed here; s_logits
            # is already in hand and we recover the original lazy-teacher
            # memory order (s_logits / t_logits never coexist).
            if fill_label_mask is not None:
                fm = fill_label_mask[:, start:end]
                if fm.any():
                    lbl = torch.where(fm, s_logits.argmax(dim=-1), lbl)

            mask = lbl != -100
            if not mask.any():
                del s_logits
                continue

            # Sampled-token OPD only needs log p(y) at the sampled label. Avoid
            # materialising full ``log_softmax`` tensors (another [B, chunk, V]
            # each) by computing ``logit_y - logsumexp(logits)`` directly. This
            # is exactly equivalent to gather(log_softmax(...), y), but removes
            # the OOM-prone allocation that used to fail at ``F.log_softmax``.
            sampled_safe = lbl.clamp(min=0).unsqueeze(-1)  # [B, chunk, 1]
            s_scaled = s_logits / temperature
            s_lp = torch.gather(s_scaled, -1, sampled_safe).squeeze(-1) - torch.logsumexp(s_scaled, dim=-1)
            del s_logits, s_scaled

            with self._teacher_lm_head_context(teacher_lm_head, t_h.device):
                t_logits = teacher_lm_head(t_h)
            t_scaled = t_logits / temperature
            t_lp = torch.gather(t_scaled, -1, sampled_safe).squeeze(-1) - torch.logsumexp(t_scaled, dim=-1)
            del t_logits, t_scaled
            # OPD paper §2.2 Sampled-Token: l^sample_t = log p_s(y) − log p_t(y).
            # E_{y~p_s}[l^sample] = KL(p_s ‖ p_t); minimising it pulls student
            # towards teacher (reverse KL). PRIOR BUG: the operand order was
            # swapped (t_lp − s_lp), which optimised −KL and pushed the
            # student AWAY from the teacher.
            per_pos_loss = (s_lp - t_lp)  # [B, chunk]; -100 positions are garbage

            chunk_loss = per_pos_loss[mask].sum()
            total_loss = total_loss + chunk_loss
            num_valid += int(mask.sum().item())

            if confidence_head is not None:
                # EAGLE per-token accept rate at the sampled label y:
                #   α(y) = min(1, p_t(y)/p_s(y))
                #        = exp(min(0, log p_t(y) − log p_s(y)))
                #        = exp(-max(0, per_pos_loss))
                # Clamp BEFORE exp for bf16 stability (avoids exp of any
                # large-magnitude operand when student strongly prefers y).
                # Detached so BCE never leaks gradient back into student
                # logits via the target.
                conf_target = torch.exp(-per_pos_loss.clamp(min=0.0)).detach()
                conf_chunk = self._accumulate_conf_chunk(
                    confidence_head=confidence_head,
                    backbone_hidden=backbone_hidden_for_conf[:, start:end],
                    mtp_hidden=s_h,
                    conf_target=conf_target,
                    mask=mask,
                    detach_inputs=detach_conf_inputs,
                    accum=conf_accum,
                    loss_dtype=total_conf_loss.dtype,
                )
                total_conf_loss = total_conf_loss + conf_chunk

            del per_pos_loss

        if num_valid == 0:
            # See note in `_compute_chunked_jsd_loss` — keep the returned 0 inside
            # the autograd graph so DeepSpeed's backward doesn't reject the loss.
            opd_loss = (student_hidden.sum() * 0.0).to(student_hidden.dtype)
        else:
            opd_loss = (total_loss / num_valid).to(student_hidden.dtype)

        if confidence_head is None:
            return opd_loss, None, {}
        # See `_compute_chunked_jsd_loss` for the rationale on passing the
        # head + ghost inputs (ZeRO-2 grad-bucket alignment when no chunk
        # contained any valid mtp position on this rank).
        conf_loss, conf_metrics = self._finalise_conf(
            total_conf_loss=total_conf_loss,
            accum=conf_accum,
            out_dtype=student_hidden.dtype,
            anchor_for_zero_grad=student_hidden,
            confidence_head=confidence_head,
            ghost_backbone_hidden=backbone_hidden_for_conf,
            ghost_mtp_hidden=student_hidden,
        )
        return opd_loss, conf_loss, conf_metrics

    # ------------------------------------------------------------------
    # Main compute_loss — branches on compressed vs. uncompressed student
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """PIPO OPD loss.

        Two paths, selected by the carrier dict produced upstream by
        ``_prepare_batch_inputs``:

        * **Per-sample chunks path** (``B > 1``): ``inputs`` carries
          ``_per_micro_chunks`` — a list of ``B`` independently encoded
          ``B=1`` chunk dicts. We forward each chunk through
          ``_compute_loss_single`` and **immediately** call
          ``accelerator.backward(sub_loss / N)`` so its full autograd graph
          (32-layer backbone + MTP + lm-head intermediates, tens of GB per
          chunk at long context) is released before the next chunk's forward
          begins. Peak activation memory drops from ``sum_i (chunk_i fwd)``
          to ``max_i (chunk_i fwd)``. The returned scalar is a zero-valued
          tensor with grad path through a trainable param so HF Trainer's
          outer ``accelerator.backward(loss)`` runs without error and lets
          DeepSpeed perform its standard per-step gradient-accumulation
          boundary detection. See ``_compute_loss_micro_chunks`` for the
          DeepSpeed ZeRO-2 ``set_gradient_accumulation_boundary(False)``
          dance that makes in-loop backwards safe.

        * **Single path** (``B == 1`` or ``return_outputs=True``): direct
          ``_compute_loss_single`` call.

        KL-divergence flavour (``OPD_KL_MODE``):
          * ``sampled`` — MC reverse-KL on sampled tokens (default for ``beta=1``).
          * ``topk`` — restrict to ``gkd_logits_topk`` candidates.
          * ``full`` — full-vocab.
        Ignored when ``beta != 1.0`` (symmetric JSD).

        Long-context: ``PIPO_OPD_CHUNK_SIZE`` chunks the lm-head projection.

        Teacher: vanilla Qwen3.5 with PAD-compaction (always on; see ``__init__``).

        Mathematical equivalence of the chunks path:
          The N in-loop backwards each contribute ``sub_loss_b / N`` of
          gradient, summed via ZeRO-2's ``all_grad_tensors`` accumulator into
          the optimizer's view of ``(1/N) * Σ_b sub_loss_b`` — i.e.
          sample-mean across chunks. Net per-step gradient = sample-balanced
          mean — equivalent to ``per_device_train_batch_size=1`` with
          ``gradient_accumulation_steps`` scaled by ``B``. Not identical to a
          token-balanced B-batch forward (long samples no longer get higher
          token-weight), but matches the ``B=1`` baseline with the natural
          B-fold GA scaling.
        """
        chunks = inputs.pop('_per_micro_chunks', None) if isinstance(inputs, dict) else None
        if chunks is not None and not return_outputs:
            return self._compute_loss_micro_chunks(model, chunks, num_items_in_batch)

        # Fast-fail: when B>1 reaches here it means _prepare_batch_inputs did not
        # produce chunks (e.g. encode_prompt_only path or off-policy dataset
        # response). Off-policy isn't tested in the OPD config (lmbda=1.0); fall
        # back to the single-shot forward.
        return self._compute_loss_single(model, inputs, return_outputs, num_items_in_batch)

    def _compute_loss_single(self, model, inputs, return_outputs, num_items_in_batch):
        """Original single-shot compute_loss body (no micro-batching)."""
        # 1. Strip control fields used internally by parent (we don't consume them).
        inputs.pop('_data_source', None)
        inputs.pop('_teacher_api_logprobs', None)
        inputs.pop('_teacher_api_indices', None)
        inputs.pop('_opsd_teacher_inputs', None)

        assert not self.use_teacher_api, (
            'Teacher API mode is not supported by PIPOGKDTrainer.')
        assert not self.use_liger_gkd_loss, (
            'Liger fused JSD loss is not integrated with PIPO HF backbone. '
            'Disable --use_liger_kernel for OPD on PIPO.')

        # 2. Build clean model_inputs
        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
        for k in ('logits_to_keep', 'channel', 'compute_loss_func', 'loss_scale', 'text_position_ids'):
            model_inputs.pop(k, None)

        # 3. PAD-compaction setup. ``_prepare_batch_inputs`` always provides input_ids
        # and labels with matching shape; if not, fail loudly downstream.
        pad_id = self._pipo_pad_id
        input_ids = model_inputs['input_ids']
        labels_full = inputs['labels']
        pad_mask = (input_ids == pad_id)
        n_pad = int(pad_mask.sum().item())

        log_metrics: Dict[str, float] = {}
        if n_pad > 0:
            log_metrics['pipo_pad_count'] = float(n_pad)

        # Build the per-position "fill this PAD label with the student's argmax"
        # mask for the MTP-loss call. We exclude batch right-padding via
        # ``attention_mask``. All PADs in PIPO rollouts land on ODD input
        # positions (prompt parity-PAD at position ``nq`` where ``nq`` is odd,
        # response rollout PAD at odd relative index with prompt_len even, and
        # the trailing parity-to-even PAD at position ``L_old`` odd), so the
        # backbone-side (even-position) slice is always all-False and the
        # backbone call below leaves ``fill_label_mask=None``. The assert
        # guards against a future rollout policy that violates this invariant.
        fill_mtp_mask: Optional[torch.Tensor] = None
        if self._opd_fill_pad_label and n_pad > 0:
            assert not pad_mask[:, 2::2].any(), (
                'fill_back_mask should be all-False under current PIPO '
                'rollout policy (all PADs land on ODD positions). Got True '
                'entries in even-index slice — rollout policy changed?')
            fill_mtp_mask = pad_mask[:, 3::2]
            log_metrics['pipo_pad_fill_count'] = float(fill_mtp_mask.sum().item())

        use_topk = (self.beta == 1.0 and self._opd_kl_mode == 'topk')
        use_sampled = (self.beta == 1.0 and self._opd_kl_mode == 'sampled')

        # Teacher hidden states are no-grad tensors.  Keep the full teacher on
        # GPU only for this backbone forward; after ``_teacher_full_logits``
        # returns, ``load_teacher_model_context`` offloads the full teacher.  The
        # chunked loss below will temporarily move only teacher ``lm_head`` to GPU.
        teacher_hidden, teacher_lm_head = self._teacher_full_logits(
            model_inputs, pad_mask, n_pad, return_hidden_states=True)

        backbone_hidden, mtp_hidden, T, student_lm_head = self._student_compressed_logits(
            model, model_inputs, return_hidden_states=True)
        L = T // 2

        # All shapes are guaranteed by the upstream contracts:
        #   - input_ids has even length T (asserted in `_student_compressed_logits`).
        #   - labels has the same T (built alongside in `_prepare_batch_inputs`).
        #   - teacher_hidden has the same T (compaction re-pads to T; non-
        #     compaction forward preserves length).
        assert teacher_hidden.size(1) == T, (
            f'teacher_hidden T mismatch: {teacher_hidden.size(1)} vs student {T}')
        assert labels_full.size(1) == T, (
            f'labels T mismatch: {labels_full.size(1)} vs student {T}')

        # Slice teacher hidden for backbone / MTP alignment.
        teacher_back = teacher_hidden[:, 1:T - 1:2]   # positions [1, 3, ..., T-3]
        teacher_mtp = teacher_hidden[:, 2:T:2]        # positions [2, 4, ..., T-2]

        labels_back = labels_full[:, 2::2]            # [B, L-1] — targets for backbone
        labels_mtp = labels_full[:, 3::2]             # [B, L-1] — targets for MTP

        chunk_size = int(os.environ.get('PIPO_OPD_CHUNK_SIZE', 1024))
        topk_arg = self.gkd_logits_topk if use_topk else None

        # ── Confidence head wiring (optional, post-hoc commit predictor) ──
        # We unwrap PEFT once here so the SAME ``confidence_head`` reference
        # can be passed into the MTP-loss chunked function below. The head
        # is supervised inside that loop (no extra lm-head forward) using the
        # EAGLE per-pair accept rate as BCE target — see
        # ``_accumulate_conf_chunk`` for the per-mode target construction
        # (sampled: ``min(1, p_t/p_s)``; full/topk: ``Σ min(p_s, p_t)``).
        unwrapped = self.accelerator.unwrap_model(model)
        base_model = unwrapped.base_model.model if is_peft_model(unwrapped) else unwrapped
        confidence_head = getattr(base_model, 'confidence_head', None)
        # Pass head into MTP-loss call iff both (a) instantiated and
        # (b) loss-weight > 0; otherwise leave conf path inactive in the chunked
        # function (no-op return, no extra compute).
        conf_head_for_mtp = (
            confidence_head if confidence_head is not None
            and self._opd_conf_loss_weight > 0 else None
        )

        if use_sampled:
            loss_back, _, _ = self._compute_chunked_sampled_kl(
                student_hidden=backbone_hidden,
                teacher_hidden=teacher_back,
                student_lm_head=student_lm_head,
                teacher_lm_head=teacher_lm_head,
                labels=labels_back,
                chunk_size=chunk_size,
                temperature=self.temperature,
            )
            loss_mtp, loss_conf, conf_metrics = self._compute_chunked_sampled_kl(
                student_hidden=mtp_hidden,
                teacher_hidden=teacher_mtp,
                student_lm_head=student_lm_head,
                teacher_lm_head=teacher_lm_head,
                labels=labels_mtp,
                chunk_size=chunk_size,
                temperature=self.temperature,
                confidence_head=conf_head_for_mtp,
                backbone_hidden_for_conf=backbone_hidden if conf_head_for_mtp is not None else None,
                detach_conf_inputs=self._opd_conf_detach_inputs,
                fill_label_mask=fill_mtp_mask,
            )
        else:
            loss_back, _, _ = self._compute_chunked_jsd_loss(
                student_hidden=backbone_hidden,
                teacher_hidden=teacher_back,
                student_lm_head=student_lm_head,
                teacher_lm_head=teacher_lm_head,
                labels=labels_back,
                chunk_size=chunk_size,
                temperature=self.temperature,
                beta=self.beta,
                topk=topk_arg,
                topk_source=self._opd_topk_source,
            )
            loss_mtp, loss_conf, conf_metrics = self._compute_chunked_jsd_loss(
                student_hidden=mtp_hidden,
                teacher_hidden=teacher_mtp,
                student_lm_head=student_lm_head,
                teacher_lm_head=teacher_lm_head,
                labels=labels_mtp,
                chunk_size=chunk_size,
                temperature=self.temperature,
                beta=self.beta,
                topk=topk_arg,
                topk_source=self._opd_topk_source,
                confidence_head=conf_head_for_mtp,
                backbone_hidden_for_conf=backbone_hidden if conf_head_for_mtp is not None else None,
                detach_conf_inputs=self._opd_conf_detach_inputs,
                fill_label_mask=fill_mtp_mask,
            )

        loss = loss_back + self._mtp_loss_weight * loss_mtp

        log_metrics['opd_backbone_loss'] = loss_back.detach().item()
        log_metrics['opd_mtp_loss'] = loss_mtp.detach().item()
        log_metrics['pipo_pair_count'] = float(L)

        # ── Conf loss + diagnostics (already computed inside MTP chunk loop) ──
        if conf_head_for_mtp is not None and loss_conf is not None:
            loss = loss + self._opd_conf_loss_weight * loss_conf
            log_metrics.update(conf_metrics)
            log_metrics['opd_conf_loss_weight'] = self._opd_conf_loss_weight
        elif confidence_head is not None and self._opd_conf_loss_weight == 0:
            # Head exists but no supervision — keep its parameters in the
            # autograd graph with a 0-grad ghost forward, otherwise DeepSpeed
            # Zero2 will complain about params with no grad on this rank.
            dummy_back = backbone_hidden[:, :1].detach() * 0.0
            dummy_mtp = mtp_hidden[:, :1].detach() * 0.0
            ghost_loss = confidence_head(dummy_back, dummy_mtp).sum() * 0.0
            loss = loss + ghost_loss.to(loss.dtype)
            if self.state.global_step == 0:
                logger.warning(
                    '[PIPOGKDTrainer] confidence_head is instantiated but '
                    'OPD_CONF_LOSS_WEIGHT=0; head will receive ghost forward only '
                    '(no real supervision). Set OPD_CONF_LOSS_WEIGHT>0 to train it.')
        elif confidence_head is None and self._opd_conf_loss_weight > 0 and self.state.global_step == 0:
            logger.warning(
                f'[PIPOGKDTrainer] OPD_CONF_LOSS_WEIGHT={self._opd_conf_loss_weight} '
                'is set but confidence_head was not instantiated on the model. ')

        if self.state.global_step % max(self.args.logging_steps, 1) == 0 and log_metrics:
            self.log(log_metrics)

        # Return a dummy outputs_student with backbone hidden so the parent
        # caller has a tensor to inspect if needed.
        outputs_student = SimpleNamespace(logits=backbone_hidden, loss=None)
        if return_outputs:
            return loss, outputs_student
        return loss

    # ------------------------------------------------------------------
    # Per-sample (B=1 chunk) micro-batching
    # ------------------------------------------------------------------
    def _maybe_get_deepspeed_engine(self, model) -> Optional[Any]:
        """Return the DeepSpeed engine wrapping ``model`` if present, else ``None``.

        DeepSpeed ZeRO-2/3 surface the ``set_gradient_accumulation_boundary``
        and ``_is_gradient_accumulation_boundary`` attributes — those are the
        only things we need from the engine. We probe three candidates in
        order: the ``model`` argument (Accelerate hands the engine directly
        through this slot when DeepSpeed is active), ``self.model_wrapped``
        (HF Trainer's stored DeepSpeed engine), and ``self.deepspeed``
        (DeepSpeed plugin attribute path).
        """
        for cand in (model,
                     getattr(self, 'model_wrapped', None),
                     getattr(self, 'deepspeed', None)):
            if cand is not None and hasattr(cand, 'set_gradient_accumulation_boundary'):
                return cand
        return None

    def _compute_loss_micro_chunks(self, model, chunks: List[Dict[str, torch.Tensor]], num_items_in_batch):
        """Per-chunk forward + IMMEDIATE backward to cap peak activation memory.

        Why
        ---
        The original implementation accumulated ``N`` chunks' forward graphs
        and let HF Trainer's outer ``accelerator.backward(total)`` consume all
        of them in a single backward. Peak activation memory was
        ``sum_i (chunk_i forward)``, which OOM'd on long-context OPD with
        variable-length samples (e.g. 12K + 25K + 48K + 64K chunks pushing
        130+ GB just for activations on B200, since each chunk's autograd
        graph held the full 32-layer backbone + MTP + lm-head intermediates
        until backward).

        New behaviour: each chunk's ``forward + scaled backward`` runs in the
        same iteration, releasing its full autograd graph (and ~tens of GB of
        activations) before the next chunk's forward begins. Peak shrinks to
        ``max_i (chunk_i forward)``. ``aggressive_empty_cache()`` after the
        backward then truly recovers the freed memory, unlike before (where
        the live autograd graph kept everything alive).

        DeepSpeed ZeRO-2 grad-accumulation contract
        --------------------------------------------
        ZeRO-2 reduce-scatters on EVERY backward (``no_sync()`` is unsupported
        for ZeRO-2 — see ``deepspeed/runtime/engine.py::no_sync`` assert).
        That's fine: each backward's reduced partition is accumulated into
        ``optimizer.all_grad_tensors`` inside
        ``independent_gradient_partition_epilogue``, and only **flushed** into
        ``optimizer.averaged_gradients`` (the buffer the optimizer actually
        consumes at ``step``) when ``is_gradient_accumulation_boundary()`` is
        True.

        So the recipe is:
          1. Force ``boundary=False`` for ALL N in-loop backwards → the
             reductions accumulate but never flush mid-loop.
          2. Reset boundary to its prior state (typically ``None`` =
             auto-detect by ``micro_steps % GA == GA-1``) before returning.
          3. Return a zero scalar with a grad path through a trainable param.
             HF Trainer's outer ``accelerator.backward(zero_loss)`` then runs
             one extra backward — but with zero gradients, so it merely adds
             0 to ``all_grad_tensors`` and lets DeepSpeed perform its standard
             per-step boundary check (deciding whether THIS outer step is
             where to flush + step, exactly as in the single-shot path).

        Net effect: gradients accumulated across N chunks behave EXACTLY like
        a single backward of ``(1/N) * Σ sub_loss`` on the optimizer's books,
        while peak activation memory drops from ``sum_i`` to ``max_i``.

        Hard contracts (asserted, fast-fail):
          * Each chunk has ``input_ids.size(0) == 1`` (one sample).
          * Each per-chunk ``sub_loss`` is a 0-d scalar tensor with grad.
          * Returned scalar is a zero-valued tensor with grad path through a
            trainable parameter, so HF Trainer's outer backward succeeds.
        """
        from copy import copy as _shallow_copy

        N = len(chunks)
        assert N >= 2, f'_compute_loss_micro_chunks expects N>=2 chunks, got {N}.'

        ds_engine = self._maybe_get_deepspeed_engine(model)
        # Save current boundary state so we can restore it after the loop. For
        # ZeRO-2 this is normally ``None`` (auto-detect) but some upstream
        # caller could have overridden it; we want to be a polite citizen.
        saved_boundary = (
            ds_engine._is_gradient_accumulation_boundary if ds_engine is not None else None)

        total_loss_val = 0.0
        for ci, chunk in enumerate(chunks):
            assert chunk['input_ids'].size(0) == 1, (
                f'chunk {ci} has B={chunk["input_ids"].size(0)}, expected exactly 1.')

            # Force NON-boundary for every in-loop backward so DeepSpeed's
            # ZeRO-2 epilogue accumulates into ``all_grad_tensors`` without
            # flushing into ``averaged_gradients``. The outer trainer's
            # backward will trigger the (real) boundary check post-loop.
            if ds_engine is not None:
                ds_engine.set_gradient_accumulation_boundary(False)

            # _compute_loss_single mutates the dict (pops control fields) —
            # pass a shallow copy so the chunk stays intact for downstream
            # debugging / reentry.
            sub_loss = self._compute_loss_single(
                model, _shallow_copy(chunk), False, num_items_in_batch)
            assert sub_loss.dim() == 0 and sub_loss.numel() == 1, (
                f'chunk {ci} sub_loss is not a 0-d scalar: shape={tuple(sub_loss.shape)}, '
                f'numel={sub_loss.numel()}.')
            assert sub_loss.requires_grad, (
                f'chunk {ci} sub_loss has no grad path — student forward broken.')

            # Scale by 1/N so the SUM over N in-loop backwards equals the
            # mean — matching the previous semantics
            # ``total = (1/N) * Σ_i sub_loss_i``.
            scaled = sub_loss / float(N)
            self.accelerator.backward(scaled)
            total_loss_val += float(scaled.detach().item())

            # Drop references to the chunk's forward graph so the next
            # iteration's ``empty_cache`` can truly reclaim its activations.
            del sub_loss, scaled
            if self._pipo_empty_cache_steps > 0:
                aggressive_empty_cache()

        # Restore DeepSpeed boundary state (let outer backward go through the
        # normal auto-detect path — DeepSpeed counts ``micro_steps`` only at
        # ``step()`` time, so our in-loop backwards never advanced it).
        if ds_engine is not None:
            ds_engine.set_gradient_accumulation_boundary(saved_boundary)

        # Return a zero-valued scalar with a real grad path. We thread the
        # grad through a single trainable parameter element (``flatten()[0]``)
        # so DeepSpeed's outer backward has something to traverse. The
        # ``total_loss_val`` is added as a detached scalar so HF Trainer's
        # loss logging (``loss.detach() / GA_steps``) shows the actual value
        # we backpropagated, not zero.
        trainable = next(p for p in model.parameters() if p.requires_grad)
        zero_anchor = trainable.flatten()[0] * 0.0
        loss_value_tensor = torch.tensor(
            total_loss_val,
            dtype=zero_anchor.dtype,
            device=zero_anchor.device,
        )
        final_loss = zero_anchor + loss_value_tensor

        assert final_loss.dim() == 0 and final_loss.requires_grad, (
            f'micro_chunks final_loss malformed: shape={tuple(final_loss.shape)}, '
            f'requires_grad={final_loss.requires_grad}.')

        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({'pipo_micro_chunks': float(N)})

        return final_loss

    # ------------------------------------------------------------------
    # Phase 3: SGLang engine override (force PIPO flags + memory saver)
    # ------------------------------------------------------------------
    def _prepare_sglang_engine(self):
        """Force-enable PIPO flags on the SGLang engine before delegation.

        Also wires the ``enable_memory_saver`` opt-in (the SGLang server arg that
        actually makes ``sleep_level > 0`` free GPU memory). See the
        ``PIPO_OPD_SGLANG_MEMORY_SAVER*`` env vars below.
        """
        assert self.args.rollout_backend == 'sglang', (
            f'_prepare_sglang_engine called with rollout_backend='
            f'{self.args.rollout_backend!r} (expected "sglang").')

        # Force PIPO serving path — we never want a silent fallback to
        # vanilla Qwen3.5 SGLang when the user is training a PIPO student.
        self.args.sglang_enable_pipo = True
        self.args.sglang_disable_radix_cache = True

        # ── enable_memory_saver opt-in ────────────────────────────────
        # Why this is needed at all:
        # The parent's ``--sleep_level 1/2`` only frees real GPU memory when
        # SGLang's ``ServerArgs.enable_memory_saver=True``. Otherwise the
        # adapter is a Noop (`pause()=pass`) and ``release_memory_occupation()``
        # silently does nothing — the trainer reports "sleeping" while VRAM
        # stays pinned (see ``.agents/SLEEP_LEVEL_SUMMARY.md``). ms-swift's
        # ``_prepare_sglang_engine`` never sets this flag, so we inject it here.
        #
        # Why it is OFF by default:
        # On the ms-swift colocate stack the very first SGLang prefill has
        # historically crashed with ``CUDAError: illegal memory access`` inside
        # ``HybridReqToTokenPool.alloc`` when the saver is on (see the
        # 2026-05-12 diary). A standalone harness in
        # ``tests/test_sglang_memory_saver.py`` cannot reproduce that crash,
        # so the root cause is suspected to be the ``hook_mode='preload'``
        # default of ``torch_memory_saver``: it requires ``LD_PRELOAD`` to
        # propagate to the SGLang scheduler subprocess, which is fragile when
        # ms-swift + DeepSpeed + torchrun have already wrapped the launch.
        #
        # What this opt-in does:
        # * ``PIPO_OPD_SGLANG_MEMORY_SAVER=1`` injects
        #   ``enable_memory_saver=True`` into ``sglang_engine_kwargs`` so the
        #   real adapter is constructed instead of the Noop.
        # * ``PIPO_OPD_SGLANG_MEMORY_SAVER_HOOK_MODE`` (default
        #   ``"torch"``) configures the global ``torch_memory_saver`` singleton
        #   BEFORE SGLang touches it. ``"torch"`` uses ``CUDAPluggableAllocator``
        #   and does NOT need ``LD_PRELOAD`` — this sidesteps the propagation
        #   problem that ``"preload"`` runs into under ms-swift, and is the
        #   reason we expose the knob.
        #
        # If ``"torch"`` mode still crashes, the user can fall back to
        # ``"preload"`` (the upstream default) by setting the env var
        # explicitly — the saver will then assert on ``LD_PRELOAD`` being
        # present, which is at least a loud failure mode.
        saver_enabled = os.environ.get(
            'PIPO_OPD_SGLANG_MEMORY_SAVER', '0') == '1'
        # Default is now ``preload`` because ``torch`` mode allocates KV cache
        # via ``CUDAPluggableAllocator``, whose pointers Triton's nvidia
        # driver cannot resolve — the first triton kernel that touches
        # ``req_to_token_pool.req_to_token`` (e.g.
        # ``write_req_to_token_pool_triton`` in
        # ``sglang/srt/mem_cache/common.py``) fails with
        # ``ValueError: Pointer argument (at 0) cannot be accessed from
        # Triton (cpu tensor?)``.
        #
        # ``preload`` mode hooks the global cuda malloc/free via
        # ``LD_PRELOAD``, so allocations inside ``region(...)`` still come
        # from the default cuda allocator — Triton sees them as normal CUDA
        # pointers. The catch is that every process that touches the saver
        # (including the spawn'd SGLang scheduler subprocess) must have
        # ``LD_PRELOAD`` set; we handle that below.
        saver_hook_mode = os.environ.get(
            'PIPO_OPD_SGLANG_MEMORY_SAVER_HOOK_MODE', 'preload').strip().lower()
        if saver_hook_mode not in ('torch', 'preload'):
            logger.warning(
                '[PIPOGKDTrainer] PIPO_OPD_SGLANG_MEMORY_SAVER_HOOK_MODE='
                f'{saver_hook_mode!r} is not in {{torch, preload}}; falling back '
                "to 'preload'.")
            saver_hook_mode = 'preload'

        if saver_enabled:
            # Quick import check — fail-fast in the parent if the wheel is
            # missing, before we spawn the SGLang scheduler subprocesses.
            try:
                import torch_memory_saver  # noqa: F401  type: ignore
            except ImportError:
                logger.warning(
                    '[PIPOGKDTrainer] PIPO_OPD_SGLANG_MEMORY_SAVER=1 '
                    'but torch_memory_saver is not installed; the saver will '
                    'be disabled and sleep_level will be a no-op.')
                saver_enabled = False

            if saver_enabled:
                # SGLang's scheduler is spawned (mp.set_start_method('spawn',
                # force=True) in entrypoints/engine.py), so configuring the
                # parent process's singleton would NOT propagate. We export the
                # mode through an env var that our patched
                # ``torch_memory_saver_adapter.py`` reads exactly once per
                # process, right before constructing the real adapter (and so
                # before the first ``region(...)`` call). The env is inherited
                # by every spawn'd subprocess.
                os.environ['TORCH_MEMORY_SAVER_HOOK_MODE'] = saver_hook_mode
                logger.info(
                    '[PIPOGKDTrainer] exporting TORCH_MEMORY_SAVER_HOOK_MODE='
                    f'{saver_hook_mode!r} for the SGLang scheduler subprocess.')

                # ``preload`` mode requires ``LD_PRELOAD`` to point at the
                # torch_memory_saver hook .so in every process that touches
                # the saver (including the spawn'd SGLang scheduler).
                # ``configure_subprocess()`` only sets it inside a narrow
                # context manager around ``proc.start()``; that is enough
                # for the scheduler subprocess (which inherits the parent's
                # env at fork), but we still need this trainer process to
                # have ``LD_PRELOAD`` set BEFORE the spawn so the inherited
                # env contains it (and so any further re-exec inside SGLang
                # also sees it). Set it idempotently here.
                if saver_hook_mode == 'preload':
                    try:
                        from torch_memory_saver.utils import (  # type: ignore
                            get_binary_path_from_package,
                        )
                        preload_so = str(get_binary_path_from_package(
                            'torch_memory_saver_hook_mode_preload'))
                        existing = os.environ.get('LD_PRELOAD', '')
                        if preload_so not in existing.split(':'):
                            os.environ['LD_PRELOAD'] = (
                                f'{preload_so}:{existing}' if existing else preload_so)
                            logger.info(
                                '[PIPOGKDTrainer] prepended '
                                f'{preload_so} to LD_PRELOAD so the SGLang '
                                'scheduler subprocess inherits the saver '
                                'hook. New LD_PRELOAD='
                                f'{os.environ["LD_PRELOAD"]!r}.')
                        else:
                            logger.info(
                                '[PIPOGKDTrainer] LD_PRELOAD already '
                                f'contains {preload_so!r}; leaving as-is.')
                    except Exception as exc:
                        logger.warning(
                            '[PIPOGKDTrainer] failed to discover the '
                            'torch_memory_saver preload .so '
                            f'({type(exc).__name__}: {exc}). preload mode '
                            'will likely assert in the SGLang scheduler. '
                            'Fall back to '
                            'PIPO_OPD_SGLANG_MEMORY_SAVER_HOOK_MODE='
                            'torch (note: incompatible with Triton kernels) '
                            'or set LD_PRELOAD manually in swift_opd.sh.')

                # Under Qwen3.5's hybrid (linear + full) attention, the
                # ``MambaPool`` ('linear-attention state' cache) is the buffer
                # that has historically tripped the first-prefill
                # ``cudaErrorIllegalAddress`` when wrapped in
                # ``memory_saver.region(KV_CACHE)`` (see the SLEEP_LEVEL diary
                # / our 2026-05-17 retest). We default to keeping the mamba
                # state OUTSIDE the saver region — the attention KV cache
                # (the much larger buffer) is still saver-tracked and will
                # be released by ``pause(kv_cache)``; only the linear-state
                # cache (~a few GB) stays GPU-resident across rollouts. The
                # explicit override ``SGLANG_DISABLE_MEMORY_SAVER_MAMBA=0``
                # re-enables the region wrap for users who never hit the
                # crash and want every byte freed.
                os.environ.setdefault(
                    'SGLANG_DISABLE_MEMORY_SAVER_MAMBA', '1')
                logger.info(
                    '[PIPOGKDTrainer] SGLANG_DISABLE_MEMORY_SAVER_MAMBA='
                    f"{os.environ['SGLANG_DISABLE_MEMORY_SAVER_MAMBA']!r} "
                    '(default 1 with saver on — avoids the PIPO × '
                    'ms-swift first-prefill crash at the cost of a few GB of '
                    'mamba state staying on the GPU across rollouts).')

        # Inject (or remove) the flag in ``sglang_engine_kwargs`` so it reaches
        # ``ServerArgs(enable_memory_saver=...)``. Default-Noop is sufficient
        # when the opt-in is off, so we don't override the user's explicit
        # ``--sglang_engine_kwargs '{"enable_memory_saver": ...}'`` if any.
        if self.args.sglang_engine_kwargs is None:
            self.args.sglang_engine_kwargs = {}
        if saver_enabled:
            self.args.sglang_engine_kwargs['enable_memory_saver'] = True

        # Honest diagnostics — the most common confusion is "I set
        # sleep_level=1, why is VRAM still 45 GB after rollout?"
        sleep_level = int(getattr(self.args, 'sleep_level', 0) or 0)
        effective_saver = bool(
            self.args.sglang_engine_kwargs.get('enable_memory_saver', False))
        if sleep_level > 0 and not effective_saver:
            logger.warning(
                f'[PIPOGKDTrainer] sleep_level={sleep_level} but '
                'enable_memory_saver=False on the SGLang engine. '
                '`_engine_sleep()` will call `release_memory_occupation()` but '
                'the noop torch_memory_saver adapter `pause()` is a no-op, so '
                'GPU memory will NOT actually be freed. To enable, set '
                'PIPO_OPD_SGLANG_MEMORY_SAVER=1 (and optionally pick a '
                'hook mode via PIPO_OPD_SGLANG_MEMORY_SAVER_HOOK_MODE='
                'torch|preload, default torch). Memory-frugal workarounds '
                'without the saver: --offload_teacher_model true, lower '
                'SGLANG_MEM_FRACTION_STATIC, smaller PIPO_OPD_CHUNK_SIZE, '
                'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, '
                '--deepspeed zero3, or SGLANG_TP_SIZE=2.')

        logger.info(
            '[PIPOGKDTrainer] forcing SGLang engine kwargs: '
            'enable_pipo=True, disable_radix_cache=True, '
            f'enable_memory_saver={effective_saver}'
            + (f' (hook_mode={saver_hook_mode!r})' if effective_saver else '')
            + '.')
        return super()._prepare_sglang_engine()

    # ------------------------------------------------------------------
    # First-time weight sync correctness check (gated by env var)
    # ------------------------------------------------------------------
    def _move_model_to_vllm(self, skip_async_check: bool = False):
        """Wrap parent's weight sync to validate compressor key parity on first call.

        Activate by setting ``PIPO_DEBUG_WEIGHT_SYNC=1`` in the environment.
        Fast-fail: any introspection failure inside the debug block raises.
        """
        debug = os.environ.get('PIPO_DEBUG_WEIGHT_SYNC', '0') == '1'
        backend = self.args.rollout_backend
        first_call = not getattr(self, '_pipo_first_sync_checked', False)

        # Pre-sync: log HF compressor norms once.
        if debug and first_call and backend == 'sglang':
            unwrapped = self.accelerator.unwrap_model(self.model)
            assert is_peft_model(unwrapped), 'PIPO OPD student must be a PEFT model.'
            base = unwrapped.base_model.model
            pre_norms = {
                n: float(p.detach().float().norm().item()) for n, p in base.compressor.named_parameters()
            }
            logger.info(f'[PIPOGKDTrainer] HF compressor L2 norms (pre-sync): {pre_norms}')

            # Key parity (HF state_dict cleaned-name set vs. SGLang params).
            hf_state = self._collect_state_dict_for_vllm()
            hf_keys = set(hf_state.keys())
            inner = self.engine.engine.tokenizer_manager.model_runner.model
            sglang_keys = set(n for n, _ in inner.named_parameters())
            missing = sorted(k for k in hf_keys if k not in sglang_keys)
            if missing:
                logger.warning(
                    f'[PIPOGKDTrainer] {len(missing)} HF state_dict keys not directly '
                    f'present in SGLang model.named_parameters() (most are remapped by '
                    f'load_weights — first 10 shown): {missing[:10]}')
            else:
                logger.info('[PIPOGKDTrainer] HF state_dict keys all present on SGLang side.')

        ret = super()._move_model_to_vllm(skip_async_check=skip_async_check)

        # Post-sync: snapshot SGLang compressor weight via get_weights_by_name.
        if debug and first_call and backend == 'sglang':
            names = ('compressor.linear.weight', 'compressor.alpha.weight', 'compressor.delta.weight')
            got = {n: self.engine.engine.get_weights_by_name(n, truncate_size=4) for n in names}
            logger.info(f'[PIPOGKDTrainer] SGLang compressor weight head (post-sync): {got}')

        if first_call:
            self._pipo_first_sync_checked = True
        return ret


__all__ = ['PIPOGKDTrainer']
