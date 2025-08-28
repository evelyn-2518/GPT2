from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# 換成 LLaMA 3 模型（建議用 8B-instruct 版本，效果比較好）
model_name = "unsloth/llama-3-8b-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# 設定 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 清理異常符號
def clean_text(text):
    text = re.sub(r"[\u2000-\u206F\uFFF0-\uFFFF\uFE00-\uFEFF\uD800-\uDFFF]+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\u3000-\u303f\uFF00-\uFFEF\u0020-\u007E]+", "", text)
    return text.strip()

# 驗證是否為通順中文句
def is_valid_chinese(text):
    return all('\u4e00' <= char <= '\u9fff' or char in "，。！？、；：「」『』" for char in text)

# 初始輸入
text = input("請輸入文字:").strip()

# 參數
desired_options = 3
max_length = 40
generate_pool_size = 10

while True:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_length,
        num_return_sequences=generate_pool_size,
        do_sample=True,
        top_k=20,
        top_p=0.92,
        temperature=0.8,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    last_char = text[-1] if text else ""
    seen = set()
    filtered_options = []

    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        continuation = decoded[len(text):].strip()
        continuation = clean_text(continuation)

        if continuation.startswith(last_char):
            continuation = continuation[1:].lstrip()

        if continuation and is_valid_chinese(continuation) and continuation not in seen:
            seen.add(continuation)
            filtered_options.append(continuation)

        if len(filtered_options) >= desired_options:
            break

    while len(filtered_options) < desired_options:
        filtered_options.append("（無法產生有效句）")

    print("\n請選擇以下接續句：")
    for i, option in enumerate(filtered_options):
        print(f"{i+1}: {option}")

    choice = input("\n請輸入選項編號（或輸入 exit 離開）：").strip()
    if choice.lower() == "exit":
        print("感謝使用，再見！")
        break

    if not choice.isdigit() or not (1 <= int(choice) <= desired_options):
        print("無效選項，請重新選擇。")
        continue

    selected = filtered_options[int(choice) - 1]
    if "無法產生" in selected:
        print("跳過空白選項，請重新選擇。")
        continue

    text = text.strip() + selected
    print(f"\n當前句子：{text}")
