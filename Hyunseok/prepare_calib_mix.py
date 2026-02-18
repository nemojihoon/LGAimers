import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset


def extract_text_from_record(record: Dict) -> Optional[str]:
    for key in ("text", "prompt", "instruction", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip()
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content.strip()}" if role else content.strip())
        if parts:
            return "\n".join(parts)

    user = record.get("user")
    assistant = record.get("assistant")
    parts = []
    if isinstance(user, str) and user.strip():
        parts.append(f"user: {user.strip()}")
    if isinstance(assistant, str) and assistant.strip():
        parts.append(f"assistant: {assistant.strip()}")
    if parts:
        return "\n".join(parts)
    return None


def reservoir_sample_texts_from_jsonl(path: str, k: int, seed: int) -> List[str]:
    if k <= 0:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"MANTA file not found: {path}")

    rng = random.Random(seed)
    sample: List[str] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = None
            if isinstance(obj, dict):
                text = extract_text_from_record(obj)
            elif isinstance(obj, str):
                text = obj.strip()
            if not text:
                continue

            seen += 1
            if len(sample) < k:
                sample.append(text)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    sample[j - 1] = text
    return sample


def sample_koalpaca_texts(
    dataset_name: str,
    split: str,
    instruction_key: str,
    output_key: str,
    k: int,
    seed: int,
) -> List[str]:
    if k <= 0:
        return []
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed)
    k = min(k, len(ds))
    ds = ds.select(range(k))

    texts: List[str] = []
    for row in ds:
        inst = row.get(instruction_key)
        out = row.get(output_key)
        if not isinstance(inst, str) or not inst.strip():
            continue
        if not isinstance(out, str) or not out.strip():
            continue
        text = f"user: {inst.strip()}\nassistant: {out.strip()}"
        texts.append(text)
    return texts


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create mixed calibration jsonl from MANTA and KoAlpaca."
    )
    parser.add_argument(
        "--manta_file",
        default="/home/dhgustjr8871/vllm/data/manta_train.jsonl",
        help="Path to local MANTA jsonl file.",
    )
    parser.add_argument(
        "--koalpaca_dataset",
        default="beomi/KoAlpaca-v1.1a",
        help="HF dataset id for KoAlpaca.",
    )
    parser.add_argument("--koalpaca_split", default="train")
    parser.add_argument("--instruction_key", default="instruction")
    parser.add_argument("--output_key", default="output")
    parser.add_argument(
        "--output",
        default="/home/dhgustjr8871/vllm/data/calib_manta_koalpaca_mix.jsonl",
        help="Output jsonl path.",
    )
    parser.add_argument("--max_samples", type=int, default=2048)
    parser.add_argument("--manta_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")
    if not (0.0 <= args.manta_ratio <= 1.0):
        raise ValueError("--manta_ratio must be in [0, 1]")

    manta_target = int(round(args.max_samples * args.manta_ratio))
    koalpaca_target = args.max_samples - manta_target

    manta_texts = reservoir_sample_texts_from_jsonl(
        args.manta_file,
        k=manta_target,
        seed=args.seed,
    )
    ko_texts = sample_koalpaca_texts(
        dataset_name=args.koalpaca_dataset,
        split=args.koalpaca_split,
        instruction_key=args.instruction_key,
        output_key=args.output_key,
        k=koalpaca_target,
        seed=args.seed + 1,
    )

    rows: List[Dict] = []
    for t in manta_texts:
        rows.append({"text": t, "source": "manta"})
    for t in ko_texts:
        rows.append({"text": t, "source": "koalpaca"})

    random.Random(args.seed + 2).shuffle(rows)
    write_jsonl(args.output, rows)

    print(
        "done: "
        f"total={len(rows)} "
        f"manta={len(manta_texts)} "
        f"koalpaca={len(ko_texts)} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
