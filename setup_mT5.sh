sudo apt update

pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.38.2 
pip install datasets==2.18.0
pip install sentencepiece
pip install evaluate
pip install rouge_score
pip install bert_score
pip install accelerate==0.27.2
pip install nltk 

mkdir -p /mnt/ebs/results