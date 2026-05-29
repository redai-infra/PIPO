from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from swift.model import ModelMeta, ModelLoader, register_model
from swift.template import TemplateType


def _patch_cached_dataset_max_length() -> None:
    """Bypass the cached-dataset max_length filter when ``truncation_strategy``
    is ``right`` or ``left``.

    By default ms-swift's :func:`swift.pipelines.utils._select_dataset` filters
    out cached samples whose length exceeds ``max_length`` *before* training
    starts. This silently shrinks the dataset and changes the number of
    training steps when ``max_length`` is reduced. When the user explicitly
    asks for ``--truncation_strategy right`` (or ``left``) they want
    truncation to happen at training time inside ``LazyLLMDataset`` instead,
    keeping the dataset size — and hence the iteration count — independent of
    ``max_length``.

    This patch:
      1. Makes ``_select_dataset`` accept ``max_length=None`` (skip max-length filter).
      2. Wraps ``get_cached_dataset`` so it temporarily nullifies ``args.max_length``
         when ``truncation_strategy in {'right', 'left'}``.
      3. Re-binds the patched ``get_cached_dataset`` into modules that import the
         symbol directly (e.g. ``swift.pipelines.train.sft``).
    """
    from swift.pipelines import utils as _pipeline_utils
    from swift.utils import get_logger

    _logger = get_logger()
    _orig_select_dataset = _pipeline_utils._select_dataset
    _orig_get_cached_dataset = _pipeline_utils.get_cached_dataset

    def _patched_select_dataset(dataset, max_length, min_length=None):
        if 'length' in dataset.column_names and 'lengths' not in dataset.column_names:
            dataset = dataset.rename_column('length', 'lengths')
        if max_length is None and min_length is None:
            return dataset
        idxs = []
        for i, length in enumerate(dataset['lengths']):
            sample_length = max(length) if isinstance(length, list) else length
            if max_length is not None and sample_length > max_length:
                continue
            if min_length is not None and sample_length < min_length:
                continue
            idxs.append(i)
        new_dataset = dataset.select(idxs)
        filtered_count = len(dataset) - len(new_dataset)
        if filtered_count > 0:
            filter_parts = []
            if max_length is not None:
                filter_parts.append(f'max_length <= {max_length}')
            if min_length is not None:
                filter_parts.append(f'min_length >= {min_length}')
            _logger.info(
                f"Dataset filtered by ({', '.join(filter_parts)}), origin length: {len(dataset)}, "
                f'filtered dataset length: {len(new_dataset)}, filtered: {filtered_count}')
        return new_dataset

    def _patched_get_cached_dataset(args):
        truncation_strategy = getattr(args, 'truncation_strategy', None)
        if truncation_strategy not in ('right', 'left'):
            return _orig_get_cached_dataset(args)
        original_max_length = args.max_length
        args.max_length = None
        try:
            _logger.info(
                f'[PIPO patch] Bypassing cached-dataset max_length filter '
                f'(truncation_strategy={truncation_strategy}, max_length={original_max_length}). '
                f'Long samples will be truncated by LazyLLMDataset at training time.')
            result = _orig_get_cached_dataset(args)
        finally:
            args.max_length = original_max_length
        return result

    _pipeline_utils._select_dataset = _patched_select_dataset
    _pipeline_utils.get_cached_dataset = _patched_get_cached_dataset

    # Re-bind into modules that imported the symbol directly.
    for mod_path in ('swift.pipelines.train.sft', 'swift.pipelines.infer.infer'):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        if hasattr(mod, 'get_cached_dataset'):
            mod.get_cached_dataset = _patched_get_cached_dataset


_patch_cached_dataset_max_length()


class Qwen3_5MtpLoader(ModelLoader):
    """Load :class:`~pipo.qwen3_5.modeling_qwen3_5_mtp.Qwen3_5ForCausalPIPO` from a Qwen3.5 hub or local path."""

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        import json
        from pathlib import Path
        from pipo.qwen3_5.modeling_qwen3_5_mtp import Qwen3_5ForCausalPIPO

        text_config = getattr(config, "text_config", config)

        mtp_default = str(getattr(text_config, "mtp_loss_weight", 1.0))
        text_config.mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", mtp_default))

        compressor_default = getattr(text_config, "compressor_type", "mlp")
        model_path = Path(model_dir)
        
        # Try to read from LoraConfig (adapter_config.json) if it exists
        lora_compressor_type = None
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                from swift.tuners import LoraConfig
                lora_config = LoraConfig.from_pretrained(model_dir)
                lora_compressor_type = getattr(lora_config, "compressor_type", None)
            except Exception:
                pass
        
        # Try to read from additional_config.json (fallback)
        additional_config_path = model_path / "additional_config.json"
        saved_compressor_type = None
        if additional_config_path.exists():
            try:
                with open(additional_config_path) as f:
                    additional_config = json.load(f)
                saved_compressor_type = additional_config.get("compressor_type")
            except Exception:
                pass
        
        # Auto-detect from path if not already determined
        path_compressor_type = None
        if "mlp" in model_path.name.lower():
            path_compressor_type = "mlp"
        elif "linear" in model_path.name.lower():
            path_compressor_type = "linear"
        
        # Priority: env > LoraConfig > saved config > path detection > config default
        compressor_type = (
            os.environ.get("COMPRESSOR_TYPE")
            or lora_compressor_type
            or saved_compressor_type
            or path_compressor_type
            or compressor_default
        )
        text_config.compressor_type = compressor_type

        if getattr(text_config, "mtp_num_hidden_layers", 0) == 0:
            text_config.mtp_num_hidden_layers = 1

        attn_impl = self.attn_impl
        if attn_impl in (None, "sdpa"):
            attn_kw = "sdpa"
        elif attn_impl in ("flash_attn", "flash_attention_2"):
            attn_kw = "flash_attention_2"
        else:
            attn_kw = attn_impl

        load_kw = dict(model_kwargs)
        load_kw.setdefault("trust_remote_code", True)

        def _load(attn: str):
            return Qwen3_5ForCausalPIPO.from_pretrained(
                model_dir,
                config=text_config,
                attn_implementation=attn,
                **load_kw,
            )

        try:
            model = _load(attn_kw)
        except Exception as e:  # noqa: BLE001 — flash / backend optional deps
            if attn_kw == "flash_attention_2":
                from swift.utils import get_logger

                get_logger().warning("flash_attention_2 load failed (%s); retrying with sdpa.", e)
                model = _load("sdpa")
            else:
                raise
        model.config.use_cache = False

        if not _checkpoint_has_compressor(model_dir):
            model.compressor.init_weights()

        return model


def _checkpoint_has_keys_prefixed(model_dir: str, prefix: str) -> bool:
    """Return True iff *model_dir* contains any weight key starting with *prefix*.

    Reuses the same shard inspection strategy as the compressor / confidence head
    detectors below (safetensors index → single safetensors → pytorch_model.bin).
    Returns False when the directory cannot be inspected, treating "unknown" as
    "absent" to default to the safer init-from-scratch path.
    """
    import json
    from pathlib import Path

    model_dir = Path(model_dir)
    for index_file in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / index_file
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                keys = data.get("weight_map", {}).keys()
                return any(k.startswith(prefix) for k in keys)
            except Exception:
                pass
    for shard in ("model.safetensors", "pytorch_model.bin"):
        shard_path = model_dir / shard
        if shard_path.exists():
            try:
                if shard.endswith(".safetensors"):
                    from safetensors import safe_open
                    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                        return any(k.startswith(prefix) for k in f.keys())
                else:
                    import torch as _torch
                    sd = _torch.load(str(shard_path), map_location="cpu")
                    return any(k.startswith(prefix) for k in sd.keys())
            except Exception:
                pass
    return False


def _checkpoint_has_compressor(model_dir: str) -> bool:
    """Return True if *model_dir* is a fine-tuned checkpoint that already contains
    trained compressor weights (so we must NOT reinitialize them)."""
    return _checkpoint_has_keys_prefixed(model_dir, "compressor.")


register_model(
    ModelMeta(
        model_type="qwen3_5_mtp",
        model_groups=[],
        template=TemplateType.qwen3_5,
        loader=Qwen3_5MtpLoader,
        architectures=["Qwen3_5ForCausalPIPO"],
        task_type="pipo",
        is_multimodal=False,
    ),
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# Re-route ``--rlhf_type gkd`` to the PIPO-aware GKD trainer subclass.
#
# Loading this plugin (via ``--external_plugins pipo/trainer/swift_plugin.py``)
# hot-patches ``swift.trainers.trainer_factory.TrainerFactory.TRAINER_MAPPING['gkd']``
# so that ``swift rlhf --rlhf_type gkd`` invokes
# :class:`pipo.trainer.swift_gkd_trainer.PIPOGKDTrainer` instead of the stock
# ``swift.rlhf_trainers.GKDTrainer``. The subclass adds even-length padding for
# PIPO, forces ``use_cache=False`` during HF generate, and force-enables PIPO
# flags on the SGLang rollout engine.
#
# Implementation chosen over a brand-new mapping key (e.g. ``'pipo_gkd'``) so that
# users do not need a custom CLI flag — loading this plugin is the single switch.
# ---------------------------------------------------------------------------
def _patch_gkd_trainer_for_pipo() -> None:
    try:
        from swift.trainers.trainer_factory import TrainerFactory  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        from swift.utils import get_logger
        get_logger().warning(
            f'[PIPO plugin] Could not patch TrainerFactory for GKD: {exc}')
        return

    target = 'pipo.trainer.swift_gkd_trainer.PIPOGKDTrainer'
    current = TrainerFactory.TRAINER_MAPPING.get('gkd')
    if current == target:
        return
    TrainerFactory.TRAINER_MAPPING['gkd'] = target

    from swift.utils import get_logger
    get_logger().info(
        f"[PIPO plugin] Re-routed TRAINER_MAPPING['gkd']: {current} -> {target}")


_patch_gkd_trainer_for_pipo()
