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
-  先[`申請API`](https://aistudio.google.com/apikey)並複製
-  避免將API放在公開網頁，將其存在系統環境變數
-  ```bash
   export GEMINI_API_KEY="你的API密鑰"
   ```
-  要使用時讀取:
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
-  [`Qwen`](https://github.com/evelyn-2518/GPT2/blob/main/Qwen.py)
