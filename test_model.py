"""测试微调后的模型"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = r"C:\Users\20939\models\Qwen\Qwen2___5-7B-Instruct"
ADAPTER_PATH = r"C:\Users\20939\llm-finetune\saves\qwen-qlora\adapter"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

questions = [
    "请解释什么是LoRA",
    "用Python写一个函数，实现二分查找",
    "用类比的方式解释什么是API",
    "RAG有什么特点",
]

for q in questions:
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*50}")
    print(f"Q: {q}")
    print(f"A: {response}")