from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re

model_name = "Qwen/Qwen-1_8B-Chat"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 設定 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def clean_text(text):
    text = re.sub(r"[\u2000-\u206F\uFFF0-\uFFFF\uFE00-\uFEFF\uD800-\uDFFF]+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\u3000-\u303f\uFF00-\uFFEF\u0020-\u007E]+", "", text)
    return text.strip()

def is_valid_chinese(text):
    return all('\u4e00' <= char <= '\u9fff' or char in "，。！？、；：「」『』" for char in text)


text = input("請輸入文字:").strip()
if not text:
    text = " "


desired_options = 3
max_length = 40


while True:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if inputs["input_ids"].size(1) == 0:
        inputs["input_ids"] = torch.tensor([[tokenizer.pad_token_id]]).to(model.device)

    filtered_options = []

    # 用 for-loop 單序列生成多候選
    while len(filtered_options) < desired_options:
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_k=20,
                top_p=0.92,
                temperature=0.8,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        continuation = decoded[inputs["input_ids"].size(1):].strip()
        continuation = clean_text(continuation)

        # 🔹 避免候選與原句重複開頭
        if continuation.startswith(text):
            continuation = continuation[len(text):].lstrip()

        # 驗證中文並去重
        if continuation and is_valid_chinese(continuation) and continuation not in filtered_options:
            filtered_options.append(continuation)

    # 補空白候選
    while len(filtered_options) < desired_options:
        filtered_options.append("（無法產生有效句）")

    # 顯示候選句
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

    # 🔹 更新原句
    text = text.strip() + selected
    print(f"\n當前句子：{text}")
