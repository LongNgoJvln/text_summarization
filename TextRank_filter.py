import os
import json
import shutil
from pyvi import ViTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Cấu hình đường dẫn S3 của bạn
# S3_OUTPUT_PATH = "s3://long-sample-bucket/xlsum_vi_filtered"
LOCAL_OUTPUT_PATH = "/mnt/ebs/data/xlsum_vi_filtered_test"

def extractive_filter(text, num_sentences=4):
    """
    Sử dụng TextRank để trích xuất các câu trụ cột, giúp mT5-small 
    tập trung vào ngữ nghĩa chính thay vì bị ngợp bởi fulltext.
    """
    if not text or len(text.split()) < 50:
        return text

    segmented_text = ViTokenizer.tokenize(text)
    parser = PlaintextParser.from_string(segmented_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    
    summary = summarizer(parser.document, num_sentences)
    filtered_text = " ".join([str(sentence) for sentence in summary])
    
    return filtered_text.replace("_", " ")

def main():
    print("Đang tải dataset XLSum Vietnamese...")
    dataset = load_dataset("csebuetnlp/xlsum", "vietnamese", trust_remote_code=True)
    
    print("Bắt đầu quá trình lọc văn bản (TextRank Filtering)...")
    # Apply filter cho toàn bộ các tập train, validation, test
    filtered_dataset = dataset.map(
        lambda x: {"text": extractive_filter(x["text"], num_sentences=4)},
        batched=False,     # batched=False để đảm bảo xử lý từng văn bản
        desc="Filtering sentences",
        num_proc=8,
    )
    
    print(f"Đang lưu dataset đã lọc lên S3: {LOCAL_OUTPUT_PATH}")
    filtered_dataset.save_to_disk(LOCAL_OUTPUT_PATH)
    zip_path = LOCAL_OUTPUT_PATH + ".zip"
    shutil.make_archive(LOCAL_OUTPUT_PATH, 'zip', LOCAL_OUTPUT_PATH)
    print("Hoàn tất! Dữ liệu đã sẵn sàng trên S3.")
    print(json.dumps([zip_path]))
    print("true")

if __name__ == "__main__":
    main()