gpuid=0
N_SHOT=5
DATA_ROOT=/Datasets/FineSports_video
MODEL_PATH=/TAMT/112112vit-s-140epoch.pt     # PATH of your Pretrained MODEL
CKPATH=/TAMT/output/finesports/5-shot
TRAJ_ROOT=/Datasets/Traj/traj_npz_yolo/finesports

echo "============= meta-train 5-shot ============="

find /tmp -maxdepth 1 -type d -name "pymp-*" -user "$USER" -exec rm -rf {} +

python meta_train.py \
 --dataset finesports \
 --data_path $DATA_ROOT  \
 --model VideoMAES \
 --method meta_deepbdc \
 --image_size 112 \
 --gpu ${gpuid} \
 --seed 2 \
 --lr 1e-3  \
 --epoch 30 \
 --train_n_episode 600 \
 --val_n_episode 300 \
 --milestones 10 20 \
 --n_shot $N_SHOT \
 --num_classes 24 \
 --reduce_dim 256 \
 --n_query 4 \
 --traj_lam 0.3 \
 --traj_lbda 0.3 \
 --traj_dim 6 \
 --traj_root $TRAJ_ROOT \
 --seq_len 8 \
 --save_freq 5 \
 --pretrain_path $MODEL_PATH \
 --checkpoint_dir $CKPATH \
 --use_traj \
