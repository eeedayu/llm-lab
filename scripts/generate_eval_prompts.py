import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sft/eval.jsonl"
OUTPUT = ROOT / "data/eval/prompts.jsonl"

prompts = []
for line_number, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), 1):
    row = json.loads(line)
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{SOURCE}:{line_number}: expected a non-empty messages list")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{SOURCE}:{line_number}: expected the reference answer last")

    prompt_messages = messages[:-1]
    if not prompt_messages or not any(message.get("role") == "user" for message in prompt_messages):
        raise ValueError(f"{SOURCE}:{line_number}: no user message found before the reference")
    prompts.append({"messages": prompt_messages})

if len(prompts) != 50:
    raise ValueError(f"Expected 50 evaluation prompts, found {len(prompts)}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    for prompt in prompts:
        handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")

print(f"Wrote {len(prompts)} prompts: {OUTPUT}")
