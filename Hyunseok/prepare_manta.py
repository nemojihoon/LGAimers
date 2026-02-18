import argparse
import json
from typing import Iterable, List, Optional

from datasets import load_dataset


def format_conversation(
    conversations: List[dict],
    include_roles: bool,
    separator: str,
) -> str:
    lines = []
    for message in conversations:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if include_roles and role:
            lines.append(f"{role}: {content}")
        else:
            lines.append(content)
    return separator.join(lines).strip()


def iter_examples(
    split: str,
    streaming: bool,
    min_complexity: Optional[int],
    max_complexity: Optional[int],
) -> Iterable[dict]:
    dataset = load_dataset("LGAI-EXAONE/MANTA-1M", split=split, streaming=streaming)
    for ex in dataset:
        label = ex.get("complexity_label")
        if label is not None:
            if min_complexity is not None and label < min_complexity:
                continue
            if max_complexity is not None and label > max_complexity:
                continue
        yield ex


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MANTA-1M for pruning/KD.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--min_complexity", type=int, default=None)
    parser.add_argument("--max_complexity", type=int, default=None)
    parser.add_argument("--include_roles", action="store_true", default=True)
    parser.add_argument("--no-roles", action="store_false", dest="include_roles")
    parser.add_argument("--separator", default="\n")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()

    total = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for ex in iter_examples(
            args.split,
            args.streaming,
            args.min_complexity,
            args.max_complexity,
        ):
            conversations = ex.get("conversations") or []
            text = format_conversation(conversations, args.include_roles, args.separator)
            if not text:
                continue
            if args.format == "jsonl":
                payload = {
                    "text": text,
                    "id": ex.get("id"),
                    "complexity_label": ex.get("complexity_label"),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                handle.write(text + "\n")
            total += 1
            if args.log_every and total % args.log_every == 0:
                print(f"written={total}")
            if args.max_samples and total >= args.max_samples:
                break
    print(f"done: {total} samples -> {args.output}")


if __name__ == "__main__":
    main()
