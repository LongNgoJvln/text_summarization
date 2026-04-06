import json
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

# Thư mục lưu kết quả trên AWS
OUTPUT_DIR = "/mnt/ebs/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_NAME = "google/mt5-small"

print(">>> [STEP 1] Tải dữ liệu và Model...", flush=True)
dataset = load_dataset("csebuetnlp/xlsum", "vietnamese", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# --- TỐI ƯU TIỀN XỬ LÝ (Để Collator tự xử lý Padding) ---
def preprocess_function(examples):
    # Loại bỏ prefix "summarize: " để mT5 tập trung vào tiếng Việt
    model_inputs = tokenizer(examples["text"], max_length=512, truncation=True)
    
    # Không dùng padding="max_length" ở đây, để DataCollator lo
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print(">>> [STEP 2] Tokenizing dữ liệu...", flush=True)
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Lọc bỏ các chuỗi rỗng để tránh lỗi chia cho 0
    decoded_preds = [p if p.strip() != "" else "empty" for p in decoded_preds]
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v, 4) for k, v in result.items()}

# --- CẤU HÌNH GENERATION MẠNH MẼ HƠN ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-4, # Tăng nhẹ LR cho mT5-small
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    predict_with_generate=True,
    # Cấu hình ép model phải "viết"
    generation_max_length=128,
    generation_num_beams=4, # Tăng chùm tìm kiếm để câu văn hay hơn
    fp16=True,
    logging_steps=50,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    # Tăng lên 2000 mẫu để mô hình kịp học cấu trúc câu
    train_dataset=tokenized_dataset["train"].select(range(2000)),
    eval_dataset=tokenized_dataset["validation"].select(range(100)),
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    compute_metrics=compute_metrics
)

print(">>> [STEP 3] Bắt đầu huấn luyện...", flush=True)
trainer.train()

print(">>> [STEP 4] Đang đánh giá và tạo JSON chi tiết...", flush=True)
test_subset = tokenized_dataset["test"].select(range(50))
predictions = trainer.predict(test_subset)

# Giải mã kết quả
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
labels = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Tính điểm ROUGE chi tiết từng dòng
raw_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=False)

results = []
original_test = dataset["test"].select(range(50))
for i in range(len(decoded_preds)):
    # Nếu kết quả vẫn rỗng, ghi chú lại để debug
    pred_text = decoded_preds[i] if decoded_preds[i].strip() != "" else "[Model không tạo ra nội dung]"
    
    results.append({
        "id": original_test[i]["id"],
        "url": original_test[i]["url"],
        "summary_output": pred_text,
        "ground_truth": decoded_labels[i],
        "scores": {
            "rouge1": round(raw_rouge["rouge1"][i], 4),
            "rougeL": round(raw_rouge["rougeL"][i], 4)
        }
    })

output_file = os.path.join(OUTPUT_DIR, "detailed_eval_results.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Thông báo file tải về
print(json.dumps([output_file]))
print("true")