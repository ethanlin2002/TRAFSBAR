gpuid=0
N_SHOT=5
DATA_ROOT=/Datasets/FineSports_video
YOURPATH=/TAMT/output/finesports/5-shot
MODEL_PATH=$YOURPATH/best_model.tar
TRAJ_ROOT=/Datasets/Traj/traj_npz_yolo/finesports

echo "============= meta-test best_model ============="
python test.py \
 --dataset finesports \
 --data_path $DATA_ROOT \
 --model VideoMAES \
 --method meta_deepbdc \
 --image_size 112 \
 --gpu ${gpuid} \
 --n_shot $N_SHOT  \
 --reduce_dim 256 \
 --test_n_episode 2000 \
 --model_path $MODEL_PATH \
 --checkpoint_dir $YOURPATH \
 --test_task_nums 1 \
 --n_query 4 \
 --seq_len 8 \
 --traj_root $TRAJ_ROOT \
 --traj_dim 6 \
 --traj_lam 1 \
 --traj_lbda 0.3 \
 --use_traj