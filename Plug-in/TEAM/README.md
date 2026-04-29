# TEAM : Temporal Alignment-Free Video Matching for Few-shot Action Recognition (CVPR 2025 Paper) [Oral Presentation]
by 
SuBeen Lee, WonJun Moon, Hyun Seok Seong, Jae-Pil Heo

Sungkyunkwan University

[[Arxiv](https://arxiv.org/abs/2504.05956)]

## Prerequisites
### 0. Clone this repo

### 1. Prepare datasets
<b>1-1.</b> Download official dataset using link. [[HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)]
[[Kinetics](https://github.com/Showmax/kinetics-downloader)]
[[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)]
[[SSv2-Small](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)]  
<b>1-2.</b> Preprocess each dataset with following code.
```
python3 preprocess.py --dataset HMDB51 --source_dir 'video dir'
python3 preprocess.py --dataset Kinetics --source_dir 'video dir'
python3 preprocess.py --dataset UCF101 --source_dir 'video dir'
python3 preprocess.py --dataset SSv2-Small --source_dir 'video dir'
```

### 2. Install dependencies or Use docker
```
pip install -r requirements.txt
```
or   
```
docker pull leesb7426/subeen:TEAM
```

### 3. Training & Evaluating
<b>3-1.</b> Train each model follwing `scripts` or download pre-trained weights from [[Drive](https://drive.google.com/drive/folders/1-2q_MHetCYx2hGfJoH2DwrPLkWWBLeLl?usp=sharing)]  

Example code for training model without evaluating (<b>Recommended</b>)
```
python3 run_train.py --method TEAM --backbone ResNet --test_later --learning_rate 0.001 --shot 1 --agg_num 60 --num_workers 4 --tasks_per_batch 16 --dataset dataset_path/hmdb51_FSAR
```
Example code for training model with evaluating (<b>Not Recommended</b>)
```
python3 run_train.py --method TEAM --backbone ResNet --learning_rate 0.001 --shot 1 --agg_num 60 --num_workers 4 --tasks_per_batch 16 --dataset dataset_path/hmdb51_FSAR
```

<b>3-2.</b> Evaluate each model following `scripts` (if you trained the model without evaluating)
```
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ResNet --shot 1 --agg_num 60 --num_workers 4 --dataset dataset_path/hmdb51_FSAR -pc work/hmdb/TEAM/ResNet/1-shot/an60/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/hmdb/TEAM/ResNet/1-shot/an60
```



##  Cite TEAM (Temporal Alignment-Free Video Matching for Few-shot Action Recognition)

If you find this repository useful, please use the following entry for citation.
<!-- ```
@inproceedings{moon2023query,
  title={Query-dependent video representation for moment retrieval and highlight detection},
  author={Moon, WonJun and Hyun, Sangeek and Park, SangUk and Park, Dongchan and Heo, Jae-Pil},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23023--23033},
  year={2023}
}
``` -->

## Contributors and Contact

If there are any questions, feel free to contact with the author: SuBeen Lee (leesb7426@gmail.com).