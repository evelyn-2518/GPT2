# 專題自習 GPT-2中文補全
## Environment
- Python 3.10
- CUDA 12.4
- Ubuntu 22.04 
## Requirements
-  [`pyproject`](https://github.com/evelyn-2518/GPT2/blob/main/pyproject.toml) 
## 開始
```bash
# 從 GitHub 複製專案
git clone https://github.com/evelyn-2518/GPT2.git
cd GPT2

# 安裝所需套件
pip install -r pyproject.toml
```
## GPT2 程式碼
-  [`中研院`](https://github.com/evelyn-2518/GPT2/blob/main/中研院.py) 
-  [`社群風格`](https://github.com/evelyn-2518/GPT2/blob/main/社群.py)
## GPT-2操作心得
-  [`心得+更動`](https://github.com/evelyn-2518/GPT2/blob/main/心得.pdf) 
## Llama3 程式碼
-  [`Llama3中文`](https://github.com/evelyn-2518/GPT2/blob/main/llama3.py)
## Gemini 2.5程式碼
   先[`申請API`](https://aistudio.google.com/apikey)並複製
   避免將API放在公開網頁，將其存在系統環境變數
-  ```bash
   export GEMINI_API_KEY="你的API密鑰"
   ```
   要使用時讀取:
   在程式前端加上
   ```python
   import os
   import google.generativeai as genai
   
   # 從環境變數讀取 API key
   api_key = os.environ.get("GEMINI_API_KEY")
   if not api_key:
       raise ValueError("請先在系統設定 GEMINI_API_KEY 環境變數")
   # 設定 API key
   genai.configure(api_key=api_key)
   ```
   以取用API
-  [`Gemini 2.5`](https://github.com/evelyn-2518/GPT2/blob/main/genimi2-5.py)程式碼
## Qwen程式碼
   執行前先升級套件
   ```bash
   pip install --upgrade transformers bitsandbytes accelerate
   ```
確保兩者版本匹配，否則 BitsAndBytesConfig 會缺少 get_loading_attributes 方法。
<br> transformers >= 4.38.x
<br> bitsandbytes >= 0.42.x </br>
-  [`Qwen`](https://github.com/evelyn-2518/GPT2/blob/main/Qwen.py)
## 除GPT2外預期實作結果

| 模型 | 模型版本/大小 | 中文表現 | 回答能力 | 生成多樣性 | 使用難易度 |
|------|---------------|----------|----------------|------------|------------|
| LLaMA3 | 7B | 中文理解中等偏上，對長文有時斷句不佳 | 適合邏輯問答 | 生成句子自然，但創意有限 | 中等，需要 Transformers / HF hub |
| Gemini 2.5 | 約 10B | 中文能力強，對日常對話、短文生成流暢 | 常識與生活問題回答較佳 | 高創意，適合聊天、詩詞、故事 | 容易使用，官方 API 支援 Python |
| Qwen 1.8B Chat | 1.8B | 中文理解不錯，對專業問題表現穩定 | 推理與邏輯能力優秀，14B 版本更強 | 生成較多樣，適合技術或知識問答 | 中等，需要 HF Transformers，量化可降低 GPU 使用 |
