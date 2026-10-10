import os, torch
from datasets import load_dataset
from transformers import (
AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training



MODEL = os.getenv("MODEL", "Qwen/Qwen3-0.6B-Base")
OUT = os.getenv("OUT", "runs/cpt")
MAX_LEN = int(os.getenv("MAX_LEN", "1024"))
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=dtype,
)
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
MODEL, quantization_config=bnb, torch_dtype=dtype
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32,
    lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    )

model = get_peft_model(model, lora)
model.print_trainable_parameters()


raw = load_dataset("json", data_files={
"train":"data/cpt/train.jsonl", "eval":"data/cpt/eval.jsonl"
})

def tokenize(batch):
    return tok(batch["text"], add_special_tokens=False)
encoded = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)


def group(batch):
    merged = {k: sum(batch[k], []) for k in batch}
    total = (len(merged["input_ids"]) // MAX_LEN) * MAX_LEN
    return {k: [v[i:i+MAX_LEN] for i in range(0, total, MAX_LEN)]
        for k, v in merged.items()}

lm = encoded.map(group, batched=True)


args = TrainingArguments(
        output_dir=OUT, num_train_epochs=1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=8e-5, warmup_steps=0.03,
        lr_scheduler_type="cosine", weight_decay=0.01,
        logging_steps=10, eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42, data_seed=42,
        )

trainer = Trainer(
    model=model, args=args,
    train_dataset=lm["train"], eval_dataset=lm["eval"],
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )


trainer.train(resume_from_checkpoint=os.getenv("RESUME") or None)
trainer.save_model(f"{OUT}/adapter_final")
tok.save_pretrained(f"{OUT}/adapter_final")