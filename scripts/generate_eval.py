import argparse
import json
from pathlib import Path
from typing import Any
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--prompts", default="data/eval/prompts.jsonl")
p.add_argument("--out", required=True)
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--load-in-4bit", action="store_true",
    help="Load the model in 4-bit to reduce GPU memory use")
a = p.parse_args()

if a.max_new_tokens < 1:
    p.error("--max-new-tokens must be at least 1")

prompt_path = Path(a.prompts)
prompts = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not prompts:
    p.error(f"No prompts found in {prompt_path}")

model_path = Path(a.model)
if not model_path.is_dir():
    p.error(
        f"Model path does not exist: {model_path}. "
        "For a DPO LoRA adapter, use --model runs/dpo/adapter_final."
    )

adapter_config_path = model_path / "adapter_config.json"
adapter_path = None
if adapter_config_path.is_file():
    adapter_path = model_path
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_path = Path(adapter_config["base_model_name_or_path"])
    if not base_model_path.is_absolute() and not base_model_path.is_dir():
        p.error(
            f"Adapter base model does not exist: {base_model_path}. "
            "Run this command from the project root or set the base model path in adapter_config.json."
        )
    model_path = base_model_path

tok = AutoTokenizer.from_pretrained(model_path)
load_options: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
if a.load_in_4bit:
    if not torch.cuda.is_available():
        p.error("--load-in-4bit requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_options["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

model = AutoModelForCausalLM.from_pretrained(model_path, **load_options)
if adapter_path is not None:
    model = PeftModel.from_pretrained(model, adapter_path)
model = model.eval()
device_map = getattr(model, "hf_device_map", {})
offloaded_devices = sorted({
    str(device)
    for device in device_map.values()
    if str(device) != "0" and not str(device).startswith("cuda")
})
if offloaded_devices:
    print(f"Model placement includes {', '.join(offloaded_devices)}; generation may be slow.", flush=True)

out_path = Path(a.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for row in tqdm(prompts, desc="Generating", unit="prompt"):
        messages = row["messages"]
        text = tok.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True)
        
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                do_sample=False, repetition_penalty=1.05)
        new = out[0, inputs.input_ids.shape[1]:]
        row.update({"model":a.model, "answer":tok.decode(new,
            skip_special_tokens=True)})
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()