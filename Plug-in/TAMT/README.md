# TAMT [CVPR 2025]
 
<img width="3652" height="2109" alt="Image" src="https://github.com/user-attachments/assets/a87cba19-fbe2-47a5-b792-70ad39c38442" />
  
### Installation

```
conda create -n TAMT python=3.6.10

conda activate TAMT

pip install -r requirements.txt

```

### Dataset
  
  Put the data the same with your filelist:  
```
finesports
├── FreeThrow
└── Basket
```
### Train and Test
  Run following commands to start training or testing:

```
bash scripts/run_fine.sh
bash scripts/test_fine.sh

bash scripts/run_multi.sh
bash scripts/test_multi.sh

```

## Pre-trained Model
The following table shows the Pre-trained Model on K-400(364 classes) with 112 × 112 resolution.
|Pre-trained Model| Checkpoint|
| ------- | -------------------------- |
| vit_s | [Download](https://drive.google.com/file/d/1VZnFspeWyQqA1stHi68aBQWsJN4vzyJv/view?usp=sharing) |

## Finetuned Model
 The following table shows the results of TAMT on CDFSAR setting in terms of 5-way 5-shot accuracy.
|Dataset           | 5-way 5-shot Acc(%) | Checkpoint|
| --------- | ------- | -------------------------- |
| HMDB  | 74.14 |[Download](https://drive.google.com/drive/folders/1YbUrlzR94d7f4qd7FLYxNw1uO6Uer7cO?usp=sharing)|

