import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()


tok = AutoTokenizer.from_pretrained(a.adapter, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
a.base, torch_dtype=torch.float16, device_map="cpu",
low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(base, a.adapter)
model = model.merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True, max_shard_size="4GB")
tok.save_pretrained(a.out)
print("saved:", a.out)