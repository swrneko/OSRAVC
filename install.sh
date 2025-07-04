#! /bin/bash

eval "$(conda shell.bash hook)"

conda create -n OSRAVC python=3.10 

export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
conda activate OSRAVC

echo "Installing python dependencies..."
pip install -r requirements.txt
pip install --no-cache-dir --compile --force-reinstall numpy==1.22.0 pandas==2.0.3 ctranslate2==4.6.0

BLIS_ARCH=generic pip install TTS==0.22.0 transformers==4.36.2

mkdir xttsv2_banana && cd xttsv2_banana
wget 'https://huggingface.co/Ftfyhh/xttsv2_banana/resolve/main/model_banana/v2.0.2/config.json'
wget 'https://huggingface.co/Ftfyhh/xttsv2_banana/resolve/main/model_banana/v2.0.2/speakers_xtts.pth'
wget 'https://huggingface.co/Ftfyhh/xttsv2_banana/resolve/main/model_banana/v2.0.2/vocab.json'
wget 'https://huggingface.co/Ftfyhh/xttsv2_banana/resolve/main/model_banana/v2.0.2/model.pth'
wget 'https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth'
wget 'https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth'
