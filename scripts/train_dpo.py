import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
from training_utils import freeze_base_model_for_lora


MODEL = os.getenv("MODEL", "merged/sft")
OUT = os.getenv("OUT", "runs/dpo")
MAX_LEN = int(os.getenv("MAX_LEN", "512"))
MAX_PROMPT = int(os.getenv("MAX_PROMPT", "256"))
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL,
    quantization_config=bnb, torch_dtype=dtype)

model.config.use_cache = False
model = freeze_base_model_for_lora(model)
model = get_peft_model(model, LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", target_modules=["q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj"]
    ))

ds = load_dataset("json", data_files={
    "train":"data/dpo/train.jsonl", "eval":"data/dpo/eval.jsonl"})

args = DPOConfig(output_dir=OUT, beta=0.1,
    max_length=MAX_LEN,
    num_train_epochs=1, per_device_train_batch_size=1,
    per_device_eval_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=5e-6, warmup_steps=0.05,
    logging_steps=10, eval_strategy="steps", eval_steps=50,
    save_steps=50, save_total_limit=2, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=(dtype==torch.bfloat16), fp16=(dtype==torch.float16),
    report_to="none", seed=42, data_seed=42)

trainer = DPOTrainer(model=model, ref_model=None, args=args,
    train_dataset=ds["train"], eval_dataset=ds["eval"],
    processing_class=tok)

trainer.train()
trainer.save_model(f"{OUT}/adapter_final")
tok.save_pretrained(f"{OUT}/adapter_final")