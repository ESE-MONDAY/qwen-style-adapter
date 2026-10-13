import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_HUB_ID = "EseMonday/qwen2.5-0.5b-style-adapter"

# Select device and precision
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on device: {device} ({dtype})...")

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 2. Load base model WITHOUT device_map="auto" to prevent meta-device issues on CPU
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    dtype=dtype
).to(device)

# 3. Attach fine-tuned adapter from Hugging Face Hub
model = PeftModel.from_pretrained(base_model, ADAPTER_HUB_ID)
model.eval()

# 4. Generate test post
messages = [
    {"role": "system", "content": "You are a writing assistant. Write concise social posts in the author's style."},
    {"role": "user", "content": "Write a short post about deploying lightweight AI models on edge servers."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

print("\n--- GENERATED OUTPUT ---")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))