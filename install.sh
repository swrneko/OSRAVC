conda create -n OSRAVC python=3.9 
conda activate OSRAVC
echo "Installing python dependencies..."
pip install --no-cache-dir --compile --force-reinstall -r requirements.txt
