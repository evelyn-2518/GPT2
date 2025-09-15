import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import torch

# 資料前處理

data_path = "NOTEEVENTS.csv"
df = pd.read_csv(data_path)

# 篩選 Nursing 類別
nursing_df = df[df["CATEGORY"] == "Nursing"].dropna(subset=["TEXT"])

def remove_punct(text):
    return re.sub(r"[^\w\s]", "", text)

nursing_df["TEXT"] = nursing_df["TEXT"].apply(remove_punct)
nursing_df = nursing_df.sample(frac=0.05, random_state=42)

# 轉成 HF Dataset
dataset = Dataset.from_pandas(
    nursing_df[["TEXT"]].rename(columns={"TEXT": "text"})
)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Tokenizer

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA 2 需要 pad_token = eos_token

def tokenize(batch):
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128  
    )
    # labels 與 input_ids 一樣
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_train = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
tokenized_eval  = dataset["test"].map(tokenize, batched=True, remove_columns=["text"])

# 設定 torch tensor 格式
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    device_map={"": device},
    trust_remote_code=True
)

# 訓練設定

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=50,
    save_total_limit=1,

    eval_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval
)


# EarlyStopping + 保存最佳模型

best_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(int(training_args.num_train_epochs)):
    print(f"\n===== Epoch {epoch+1} =====")
    trainer.train()
    
    metrics = trainer.evaluate()
    val_loss = metrics["eval_loss"]
    print(f"Validation loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        print("New best model! Saving...")
        trainer.save_model("./best_model")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print("Early stopping triggered.")
        break
