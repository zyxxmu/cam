# Installation  
Step 1: Create a new conda environment:
```
conda create -n cam python=3.8
conda activate cam
```
Step 2: Install relevant packages
```
pip install crfm_helm==0.2.3
pip install lm_eval==0.3.0
pip install requests
pip install accelerate
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers==4.28.1
```
（Note: the Tranformers package for wiki2/pg19 tasks is the version of 4.33.0.
So you can install it by "pip install transformers==4.33.0".  ）
