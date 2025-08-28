import google.generativeai as genai
import re
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# 建立 Gemini 2.5 模型
model = genai.GenerativeModel("gemini-2.5-flash")

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
    prompt = f"請幫我續寫以下中文句子，給我 {generate_pool_size} 個不同的短句續寫，每個不超過 {max_length} 字：\n{text}"
    response = model.generate_content(prompt)

    raw_outputs = response.text.split("\n")  # Gemini 產生的內容通常以換行分隔
    last_char = text[-1] if text else ""
    seen = set()
    filtered_options = []

    for output in raw_outputs:
        continuation = clean_text(output.strip())
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
