# text_summarization
Run: python script.py --config_file=config.json 
Inference with pretrained mT5: config.json: "python_script": "inference_mt5.py"
Fine-tune with mT5, dataset xlsum-vi:   config.json: "python_script": "finetune_mt5.py"
Preprocess data for remove newlines case and select top, mid, tail sentences: uncomment codes in "finetune_mt5.py"
Preprocess data for TextRank and ROUGE-based sentence selection, run "TextRank_filtered.py" and "rougebased_filter.py" 
