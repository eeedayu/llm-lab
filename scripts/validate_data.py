import json, sys
from pathlib import Path



mode, path = sys.argv[1], Path(sys.argv[2])
rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "empty dataset"


def text_of(messages):
    assert isinstance(messages, list) and messages
    for m in messages:
        assert m.get("role") in {"system", "user", "assistant"}
        assert isinstance(m.get("content"), str) and m["content"].strip()
    return "".join(m["content"].strip() for m in messages)


for i, row in enumerate(rows, 1):
    if mode == "cpt":
        assert isinstance(row.get("text"), str) and len(row["text"].strip()) >= 40
    elif mode == "sft":
        text_of(row["messages"])
        assert row["messages"][-1]["role"] == "assistant"
    elif mode == "dpo":
        text_of(row["prompt"]); c = text_of(row["chosen"]); r = text_of(row["rejected"])
        assert c != r, f"row {i}: identical pair"
    else:
        raise ValueError("mode must be cpt/sft/dpo")
print(f"OK: {len(rows)} rows in {path}")