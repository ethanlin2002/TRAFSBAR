# D2ST

[📖 Paper](https://arxiv.org/abs/2312.01431)

<img width="3652" height="2109" alt="Image" src="https://github.com/user-attachments/assets/e329c442-40fb-4c98-9957-bcf91e610d88" />

## Installation

```
# create virtual environment
conda create -n D2ST python=3.7
conda activate D2ST

# install pytorch
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# install CLIP
pip install git+https://github.com/openai/CLIP.git

# install other requirements
pip install -r requirements.txt
```

## Training and Testing

The commands for running experiments on different datasets with various backbones are as follows:

```
bash scripts/run_fine.sh

bash scripts/run_multi.sh

```
