# TEAM [CVPR 2025]

<img width="3652" height="2109" alt="Image" src="https://github.com/user-attachments/assets/97018441-53c1-4ad3-9967-f60ba39908cf" />

### Installation

```
bash tool/preprocess.sh
```

### 2. Install dependencies or Use docker
```
conda create -n TEAM python=3.10

conda activate TEAM

pip install -r requirements.txt
```

### 3. Training & Testing

```
bash scripts/run_fine.sh
bash scripts/test_fine.sh

bash scripts/run_multi.sh
bash scripts/test_multi.sh
```
