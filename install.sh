#! /bin/bash

eval "$(conda shell.bash hook)"

conda create -n OSRAVC python=3.10 

export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
conda activate OSRAVC

echo "Installing python dependencies..."
pip install -r requirements.txt
pip install --no-cache-dir --compile --force-reinstall numpy==1.22.0 pandas==2.0.3 ctranslate2==4.6.0

BLIS_ARCH=generic pip install TTS==0.22.0 transformers==4.36.2
