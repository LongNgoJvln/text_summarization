# import os

# import json
# import shutil
# import nltk
# import numpy as np
# from datasets import load_dataset
# import evaluate

# rouge_scorer = evaluate.load("rouge")

# # Tải tài nguyên tách câu
# nltk.download('punkt')
# nltk.download('punkt_tab')

# # Cấu hình đường dẫn lưu trên EBS
# LOCAL_OUTPUT_PATH = "/mnt/ebs/data/xlsum_vi_rouge_filtered"

# def rouge_based_filter(example, num_sentences=4):
#     """
#     Chọn ra n câu có điểm ROUGE-L cao nhất so với bản tóm tắt mẫu.
#     """
#     text = example["text"]
#     summary = example["summary"]
    
#     if not text or not summary or len(text.split()) < 50:
#         return {"text": text}
        
#     sentences = nltk.sent_tokenize(text)
#     if len(sentences) <= num_sentences:
#         return {"text": text}

#     scores = []
#     for sent in sentences:
#         res = rouge_scorer.compute(
#             predictions=[sent], 
#             references=[summary], 
#             use_stemmer=True
#         )
#         scores.append(res["rougeL"])

#     top_indices = np.argsort(scores)[-num_sentences:]
#     top_indices.sort()

#     selected_sentences = [sentences[i] for i in top_indices]
#     filtered_text = " ".join(selected_sentences)
    
#     return {"text": filtered_text}

# def main():
#     os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)
#     dataset = load_dataset("csebuetnlp/xlsum", "vietnamese", trust_remote_code=True)
#     filtered_dataset = dataset.map(
#         rouge_based_filter,
#         batched=False,
#         num_proc=16,
#         desc="Filtering by ROUGE-L score"
#     )
    
#     print(f"Đang lưu dataset đã lọc lên S3: {LOCAL_OUTPUT_PATH}")
#     filtered_dataset.save_to_disk(LOCAL_OUTPUT_PATH)
#     zip_path = LOCAL_OUTPUT_PATH + ".zip"
#     shutil.make_archive(LOCAL_OUTPUT_PATH, 'zip', LOCAL_OUTPUT_PATH)
#     print("Hoàn tất! Dữ liệu đã sẵn sàng trên S3.")
#     print(json.dumps([zip_path]))
#     print("true")

# if __name__ == "__main__":
#     main()
import os
import json
import shutil
import nltk
import numpy as np
from datasets import load_dataset

from rouge_score import rouge_scorer

# ⚠️ Quan trọng: load 1 lần
nltk.download('punkt')
nltk.download('punkt_tab')

LOCAL_OUTPUT_PATH = "/mnt/ebs/data/xlsum_vi_rouge2_filtered"

# ⚠️ GLOBAL SCORER (mỗi process sẽ copy riêng)
scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

def rouge_based_filter(example):
    text = example["text"]
    summary = example["summary"]

    if not text or not summary or len(text.split()) < 50:
        return {"text": text}

    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 4:
        return {"text": text}

    # ⚡ nhanh hơn evaluate rất nhiều
    scores = [
        scorer.score(summary, sent)['rouge2'].fmeasure
        for sent in sentences
    ]

    top_indices = np.argsort(scores)[-4:]
    top_indices.sort()

    filtered_text = " ".join([sentences[i] for i in top_indices])
    return {"text": filtered_text}


def main():
    os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)

    dataset = load_dataset(
        "csebuetnlp/xlsum",
        "vietnamese",
        trust_remote_code=True
    )

    filtered_dataset = dataset.map(
        rouge_based_filter,
        num_proc=16,
        desc="Filtering by ROUGE-2",
        load_from_cache_file=False
    )

    filtered_dataset.save_to_disk(LOCAL_OUTPUT_PATH)
    zip_path = LOCAL_OUTPUT_PATH + ".zip"
    shutil.make_archive(LOCAL_OUTPUT_PATH, 'zip', LOCAL_OUTPUT_PATH)

    print(json.dumps([zip_path]))
    print("true")


if __name__ == "__main__":
    main()