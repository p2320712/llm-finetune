"""QLoRA 微调 Qwen2.5-7B-Instruct"""
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = r"C:\Users\20939\models\Qwen\Qwen2___5-7B-Instruct"
DATA_DIR = r"C:\Users\20939\llm-finetune\data"
OUTPUT_DIR = r"C:\Users\20939\llm-finetune\saves\qwen-qlora"
MAX_LEN, LORA_R, LORA_ALPHA, LR, EPOCHS, BATCH_SIZE, GRAD_ACCUM = 512, 8, 16, 1e-4, 3, 1, 8

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
)
model.enable_input_require_grads()
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
    lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading data...")
with open(f"{DATA_DIR}/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(f"{DATA_DIR}/val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)
train_ds = Dataset.from_list(train_data)
val_ds = Dataset.from_list(val_data)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

def tokenize(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    full_text = prompt + output + "<|im_end|>"
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LEN, padding=False)
    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
val_ds = val_ds.map(tokenize, remove_columns=val_ds.column_names)

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, lr_scheduler_type="cosine",
    bf16=True, logging_steps=5, save_steps=100, eval_strategy="steps", eval_steps=50,
    per_device_eval_batch_size=1, max_grad_norm=1.0, warmup_ratio=0.05,
    report_to="none", save_total_limit=2,
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
print("Starting training...")
trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")
print(f"Done! Saved to {OUTPUT_DIR}/adapter")