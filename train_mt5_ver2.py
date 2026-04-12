import os
import json
import shutil
import numpy as np
import torch 

from datasets import load_dataset, disable_progress_bar
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline
)

import evaluate
import warnings
warnings.filterwarnings("ignore")
disable_progress_bar()

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
EBS_MODEL_DIR = "/mnt/ebs/mt5_xlsum_vi"
EBS_JSON_PATH = "/mnt/ebs/results/"
# LOCAL_JSON_PATH = "./evaluation_results.json"
# LOCAL_MODEL_DIR = "./mt5_xlsum_vi"

if os.path.exists(EBS_MODEL_DIR):
    shutil.rmtree(EBS_MODEL_DIR)
if os.path.exists(EBS_JSON_PATH):
    shutil.rmtree(EBS_JSON_PATH)

os.makedirs(EBS_MODEL_DIR, exist_ok=True)
os.makedirs("/mnt/ebs/results", exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("csebuetnlp/xlsum", "vietnamese")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

max_input_length = 512
max_target_length = 128

def preprocess(example):
    model_input = example["text"]
    target = example["summary"]

    inputs = tokenizer(
        model_input,
        max_length=max_input_length,
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target,
            max_length=max_target_length,
            truncation=True,
        )

    inputs["labels"] = labels["input_ids"]

    return inputs

print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=MODEL_NAME
)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(torch.tensor(labels) != -100, torch.tensor(labels), torch.tensor(tokenizer.pad_token_id))
    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )
    scores = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return scores

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

training_args = Seq2SeqTrainingArguments(
    output_dir=EBS_MODEL_DIR,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    fp16=True,
    load_best_model_at_end=True,
    report_to="none",
    predict_with_generate=True,
    max_grad_norm=1.0,
    save_total_limit=1 
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(1000)), 
    eval_dataset=tokenized_dataset["validation"].shuffle(seed=42).select(range(200)), 
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Start training...")
trainer.train()

print("Saving model to EBS...")

trainer.save_model(EBS_MODEL_DIR)
tokenizer.save_pretrained(EBS_MODEL_DIR)

print("Running inference on test set...")
test_data = dataset["test"].select(range(50))

results = []
all_preds = []
all_labels = []

for sample in test_data:
    input_text = sample["text"]
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )
    inputs = {k:v.to(model.device) for k,v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_length = max_target_length,
        num_beams=4,
        early_stopping=True
    )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    indiv_rouge = rouge.compute(predictions=[prediction], references=[sample["summary"]])

    results.append({
        "id": sample["id"],
        "url": sample["url"],
        "text": sample["text"],
        "reference_summary": sample["summary"],
        "generated_summary": prediction,
        "rouge_scores": {k: round(v, 4) for k, v in indiv_rouge.items()}
    })

    all_preds.append(prediction)
    all_labels.append(sample["summary"])

aggregate_rouge = rouge.compute(predictions=all_preds, references=all_labels)
final_output = {
    "individual_results": results,
    "aggregate_scores": {k: round(v, 4) for k, v in aggregate_rouge.items()}
}

print("Saving JSON...")
output_file = os.path.join(EBS_JSON_PATH, "evaluation_fulltext_input.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)

print("Compressing model for local download...")
zip_base_name = os.path.join(EBS_JSON_PATH, "mt5_finetuned_model")
shutil.make_archive(zip_base_name, 'zip', EBS_MODEL_DIR)
model_zip_full_path = zip_base_name + ".zip"
print("Copy model to local...")
print(json.dumps([output_file, model_zip_full_path]))
print(json.dumps([output_file]))
print("true")
