import argparse
import json
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional

import torch
from compressed_tensors.quantization.quant_scheme import is_preset_scheme
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_dtype(name: str):
    name = name.lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_scheme_name(name: str) -> str:
    raw = str(name).strip()
    upper = raw.upper()
    aliases = {
        "FP8_STATIC": "FP8",
        "FP8-STATIC": "FP8",
        "STATIC_FP8": "FP8",
    }
    return aliases.get(upper, upper)


def _extract_text_from_record(record: Dict) -> Optional[str]:
    for key in ("text", "prompt", "instruction", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content.strip()}" if role else content.strip())
        if parts:
            return "\n".join(parts)

    user = record.get("user")
    assistant = record.get("assistant")
    parts: List[str] = []
    if isinstance(user, str) and user.strip():
        parts.append(f"user: {user.strip()}")
    if isinstance(assistant, str) and assistant.strip():
        parts.append(f"assistant: {assistant.strip()}")
    if parts:
        return "\n".join(parts)
    return None


def load_calibration_dataset(path: str, max_samples: int) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    max_samples = max(1, int(max_samples))
    texts: List[str] = []

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    text = _extract_text_from_record(payload)
                elif isinstance(payload, str):
                    text = payload.strip()
                else:
                    text = None
                if text:
                    texts.append(text)
                if len(texts) >= max_samples:
                    break
    else:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    texts.append(text)
                if len(texts) >= max_samples:
                    break

    if not texts:
        raise ValueError(
            "No usable calibration text found. Use text or jsonl with text/prompt/instruction/input/messages."
        )
    return Dataset.from_dict({"text": texts})


def finalize_output_dir(staging_dir: str, output_dir: str) -> None:
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        backup_dir = (
            output_dir.rstrip(os.sep)
            + f".bak-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        )
        shutil.move(output_dir, backup_dir)
    shutil.move(staging_dir, output_dir)


def copy_missing_assets(base_model_dir: str, dst_dir: str) -> None:
    for fname in (
        ".gitattributes",
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src = os.path.join(base_model_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    src_assets = os.path.join(base_model_dir, "assets")
    dst_assets = os.path.join(dst_dir, "assets")
    if os.path.isdir(src_assets) and not os.path.exists(dst_assets):
        shutil.copytree(src_assets, dst_assets)


def dedupe_tied_lm_head(model_dir: str) -> None:
    model_file = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(model_file):
        return

    try:
        with safe_open(model_file, framework="pt") as handle:
            keys = set(handle.keys())
            if "lm_head.weight" not in keys or "model.embed_tokens.weight" not in keys:
                return

            lm_head = handle.get_tensor("lm_head.weight")
            embed = handle.get_tensor("model.embed_tokens.weight")
            metadata = handle.metadata()

        if lm_head.shape != embed.shape or lm_head.dtype != embed.dtype:
            print("skip lm_head dedupe: shape or dtype mismatch with embedding")
            return
        if not torch.equal(lm_head, embed):
            print("skip lm_head dedupe: tensors are not identical")
            return

        tensors = load_file(model_file)
        tensors.pop("lm_head.weight", None)
        tmp_file = f"{model_file}.tmp"
        save_file(tensors, tmp_file, metadata=metadata)
        os.replace(tmp_file, model_file)
        print("removed duplicated lm_head.weight (tied with embed_tokens)")
    except Exception as exc:
        print(f"warning: failed to dedupe lm_head.weight: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply FP8 quantization compatible with vLLM compressed-tensors flow."
    )
    parser.add_argument("--base_model", required=True, help="Local HF model folder.")
    parser.add_argument("--output_dir", required=True, help="Output HF model folder.")
    parser.add_argument(
        "--scheme",
        default="FP8_DYNAMIC",
        help=(
            "Quantization preset scheme name. "
            "Examples: FP8_DYNAMIC, FP8 (static). "
            "FP8_STATIC is accepted as an alias of FP8 for convenience."
        ),
    )
    parser.add_argument(
        "--targets",
        default="Linear",
        help="Quantization targets (comma-separated, e.g. Linear or q_proj,k_proj,v_proj).",
    )
    parser.add_argument(
        "--ignore",
        default="lm_head",
        help="Module names to ignore (comma-separated).",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", default="auto", help="Transformers device_map.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    parser.add_argument("--fix_mistral_regex", action="store_true", default=False)

    # Optional FP8 KV cache quantization.
    parser.add_argument("--kv_cache_fp8", action="store_true", default=False)

    # Calibration options (used for FP8_STATIC or when explicitly provided).
    parser.add_argument(
        "--calib_file",
        default=None,
        help="Optional local text/jsonl file for calibration.",
    )
    parser.add_argument("--num_calibration_samples", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--text_column", default="text")
    parser.add_argument(
        "--dedupe_tied_lm_head",
        action="store_true",
        default=False,
        help="Enable post-save dedupe: remove lm_head.weight if identical to embeddings.",
    )
    parser.add_argument(
        "--no-dedupe_tied_lm_head",
        action="store_false",
        dest="dedupe_tied_lm_head",
        help="Disable tied lm_head deduplication (default).",
    )
    args = parser.parse_args()

    scheme = normalize_scheme_name(args.scheme)
    if not is_preset_scheme(scheme):
        raise ValueError(
            f"Unknown scheme '{args.scheme}'. "
            "Use a valid preset such as FP8_DYNAMIC or FP8."
        )

    # Static FP8 should always be calibrated.
    if scheme == "FP8" and not args.calib_file:
        raise ValueError("--scheme FP8 (or alias FP8_STATIC) requires --calib_file.")

    dtype = parse_dtype(args.dtype)
    targets = parse_csv(args.targets)
    ignore = parse_csv(args.ignore)

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        fix_mistral_regex=args.fix_mistral_regex,
    )

    print(
        f"loading model (dtype={args.dtype}, device_map={args.device_map}, "
        f"local_files_only={args.local_files_only})..."
    )
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if dtype != "auto":
        model_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.eval()

    kv_cache_scheme = None
    if args.kv_cache_fp8:
        kv_cache_scheme = {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        }

    recipe = QuantizationModifier(
        targets=targets if len(targets) > 1 else targets[0],
        scheme=scheme,
        ignore=ignore,
        kv_cache_scheme=kv_cache_scheme,
    )

    calib_dataset = None
    if args.calib_file:
        print(
            f"loading calibration dataset from {args.calib_file} "
            f"(max_samples={args.num_calibration_samples})..."
        )
        calib_dataset = load_calibration_dataset(
            args.calib_file,
            max_samples=args.num_calibration_samples,
        )
        print(f"calibration rows prepared: {len(calib_dataset)}")

    output_dir = os.path.abspath(args.output_dir)
    parent_dir = os.path.dirname(output_dir) or "."
    base_name = os.path.basename(output_dir.rstrip(os.sep)) or "output_model_fp8"
    staging_dir = tempfile.mkdtemp(prefix=f".{base_name}.tmp-", dir=parent_dir)

    print(
        "running oneshot quantization... "
        f"(scheme={scheme}, targets={targets}, ignore={ignore})"
    )
    try:
        oneshot(
            model=model,
            tokenizer=tokenizer,
            recipe=recipe,
            dataset=calib_dataset,
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            text_column=args.text_column,
            output_dir=staging_dir,
            save_compressed=True,
        )
        copy_missing_assets(args.base_model, staging_dir)
        if args.dedupe_tied_lm_head:
            dedupe_tied_lm_head(staging_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    finalize_output_dir(staging_dir, output_dir)
    print(f"saved fp8 model -> {output_dir}")


if __name__ == "__main__":
    main()
