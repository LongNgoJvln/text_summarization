import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm

MODEL_NAME = "google/mt5-small"

SAVE_DIR = "/mnt/ebs/results"
OUTPUT_FILE = os.path.join(SAVE_DIR, "mt5_xlsum_vi_predictions.json")

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

print("Loading dataset...")
dataset = load_dataset("csebuetnlp/xlsum", "vietnamese")

test_data = dataset["test"].select(range(100))  # Lấy 100 mẫu đầu tiên để test nhanh

rouge = evaluate.load("rouge")

def generate_summary(text):

    input_text = "summarize this Vietnamese text: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


results = []

print("Running inference...")

for sample in tqdm(test_data):

    pred = generate_summary(sample["text"])

    score = rouge.compute(
        predictions=[pred],
        references=[sample["summary"]]
    )

    results.append({
        "id": sample["id"],
        "url": sample["url"],
        "summary_output": pred,
        "ground_truth": sample["summary"],
        "scores": {
            "rouge1": score["rouge1"],
            "rouge2": score["rouge2"],
            "rougeL": score["rougeL"]
        }
    })


print("Saving json...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Saved to {OUTPUT_FILE}")

# QUAN TRỌNG:
# script.py sẽ đọc 2 dòng cuối để copy file về local

print(json.dumps([OUTPUT_FILE]))
print("true")