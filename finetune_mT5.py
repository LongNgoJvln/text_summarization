import os
import json
import shutil
import numpy as np
import torch 
import re 
import nltk

from datasets import load_dataset, disable_progress_bar, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline, 
    set_seed
)

import evaluate
import warnings
warnings.filterwarnings("ignore")
disable_progress_bar()

set_seed(42)

nltk.download('punkt')
nltk.download('punkt_tab')

MODEL_NAME = "google/mt5-small"
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
# dataset=load_from_disk("/mnt/ebs/data/xlsum_vi_rouge1_filtered")
# dataset=load_from_disk("/mnt/ebs/data/xlsum_vi_textrank_filtered")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

max_input_length = 512
max_target_length = 128

def positional_filter(text):
    if not text or len(text.split()) < 50:
        return text
    
    # Tách câu
    sentences = nltk.sent_tokenize(text)
    n = len(sentences)
    
    if n <= 3:
        return text
    
    # Chọn câu: Đầu, Giữa, Cuối
    first_sent = sentences[0]
    middle_sent = sentences[n // 2]
    last_sent = sentences[-1]
    
    # Kết hợp lại thành input mới
    return f"{first_sent} {middle_sent} {last_sent}"

def clean_newlines(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(example):
    # cleaned_texts = [clean_newlines(text) for text in example["text"]] 
    # model_input = ["summarize: " + text for text in cleaned_texts]
    # filtered_texts = [positional_filter(t) for t in example["text"]] 
    # model_input = ["summarize: " + text for text in filtered_texts]
    model_input = ["summarize: " + text for text in example["text"]]
    target = example["summary"]

    # model_input = [" ".join(text.split()) for text in example["text"]]    
    # target = [" ".join(summary.split()) for summary in example["summary"]]

    inputs = tokenizer(
        model_input,
        max_length=max_input_length,
        truncation=True,
        padding = "max_length"
    )

    labels = tokenizer(
        target,
        max_length=max_target_length,
        truncation=True,
        padding = "max_length"
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
    model=MODEL_NAME,
    padding = True,
)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
bert_score = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    # preds = np.argmax(preds, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )
    rouge_scores = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    bleu_score = bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        smooth=True
    )
    meteor_score = meteor.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    bert_results = bert_score.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        lang="vi", 
        model_type="bert-base-multilingual-cased" # Hoặc "vinai/phobert-base"
    )

    return {
        **rouge_scores,
        "bleu": bleu_score["bleu"],
        "meteor": meteor_score["meteor"],
        "bertscore_f1": np.mean(bert_results["f1"])
    }

print("Loading model...")
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = Seq2SeqTrainingArguments(
    seed=42,
    data_seed=42,
    output_dir=EBS_MODEL_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=50, 
    eval_steps=2000, 
    predict_with_generate=True,
    max_grad_norm=1.0,
    save_steps=2000, 
    save_total_limit=1,
    warmup_steps=100,
    weight_decay=0.03, 
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    fp16=False,
    # optim = "adafactor",
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(10000)), 
    eval_dataset=tokenized_dataset["validation"].shuffle(seed=42).select(range(1000)), 
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
test_data = dataset["test"].shuffle(seed=42).select(range(100)) 

results = []
all_preds = []
all_labels = []

for sample in test_data:
    input_text = " ".join(sample["text"].split()) 
    # filtered_text = positional_filter(sample["text"])
    # input_text = "summarize: " + filtered_text
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
        num_beams=5,
        no_repeat_ngram_size=2, 
        length_penalty=1.0, 
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
        "rouge_scores": {k: round(v, 4) for k, v in indiv_rouge.items()},
        "bleu_score": round(indiv_bleu["bleu"], 4),
        "meteor_score": round(indiv_meteor["meteor"], 4),
        # "bertscore_f1": round(indiv_bert["f1"][0], 4)
    })

    all_preds.append(prediction)
    all_labels.append(sample["summary"])

aggregate_rouge = rouge.compute(predictions=all_preds, references=all_labels)
# aggregate_bert = bert_score.compute(predictions=all_preds, 
#                                     references=all_labels, 
#                                     lang="vi", 
#                                     model_type="bert-base-multilingual-cased")
# avg_bert_f1 = float(np.mean(aggregate_bert["f1"]))
# avg_bert_p = float(np.mean(aggregate_bert["precision"]))
# avg_bert_r = float(np.mean(aggregate_bert["recall"]))
final_output = {
    "individual_results": results,
    "aggregate_rouge": {k: round(v, 4) for k, v in aggregate_rouge.items()},
    # "aggregate_bertscore_f1": round(avg_bert_f1, 4),
    # "aggregate_bertscore_precision": round(avg_bert_p, 4),
    # "aggregate_bertscore_recall": round(avg_bert_r, 4)
}

print("Saving JSON...")
output_file = os.path.join(EBS_JSON_PATH, "020526_rouge1_fulltext_10k_3epoch.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)
# Download finetuned model if needed
# print("Compressing model for local download...")
# zip_base_name = os.path.join(EBS_JSON_PATH, "mt5_finetuned_model")
# shutil.make_archive(zip_base_name, 'zip', EBS_MODEL_DIR)
# model_zip_full_path = zip_base_name + ".zip"
# print("Copy model to local...")
# print(json.dumps([output_file, model_zip_full_path]))
print(json.dumps([output_file]))
print("true")
