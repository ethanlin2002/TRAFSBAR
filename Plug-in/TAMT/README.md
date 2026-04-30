# TAMT [CVPR 2025]
 
<img width="3652" height="2109" alt="Image" src="https://github.com/user-attachments/assets/a87cba19-fbe2-47a5-b792-70ad39c38442" />
  
## Installation

```
conda create -n TAMT python=3.6.10

conda activate TAMT

pip install -r requirements.txt

```

### 2.Dataset
  
  Put the data the same with your filelist:  
```
hmdb51_org
├── brush_hair
└── cartwheel
```
### 3.Train and Test
  Run following commands to start training or testing:

```
cd scripts/hmdb51/run_meta_deepbdc
sh run_test.sh    # For test only.

sh run_metatrain.sh    # For train and test, for individual training or testing, please comment out parts of the code yourself.
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
| SSV2  | 59.18 |[Download](https://drive.google.com/drive/folders/1hvgnnAozAkYWinwOp39KKbtz1dT-lyYX?usp=sharing)|
| Diving | 45.18 |[Download](https://drive.google.com/drive/folders/18A7Rd9kmBArkxC3h_TLQEmgmlempGPx_?usp=sharing)|
| UCF  | 95.92 |[Download](https://drive.google.com/drive/folders/1mFnz41V0cljrrgWvQCiagX-VJovIpveB?usp=sharing)|
| RareAct   | 67.44 |[Download](https://drive.google.com/drive/folders/1iaklb-tr4-UqGUOEnW_CDizDW0CA5S-s?usp=sharing)|
