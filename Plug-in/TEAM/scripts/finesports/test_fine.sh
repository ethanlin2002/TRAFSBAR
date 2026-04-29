shot=5
K=70
lam=0.5
eval_lam=1
yolo=_yolo
#noyolo=n

python3 run_eval.py \
 --method TEAM \
 --backbone ViT \
 --shot $shot \
 --agg_num $K \
 --num_workers 4 \
 --dataset Finesports_FSAR \
 --query_per_class 4 \
 --seq_len 5 \
 --num_test_tasks 2000 \
 -pc /home/mmlab206/61347023S/TEAM/work/Finesports-T5/TEAM/ViT/$shot-shot/an$K-$noyolo$lam/checkpoint_best_val.pt \
 --traj_root /home/mmlab206/61347023S/Datasets/Traj/traj_npz$yolo/finesports \
 --traj_lam $eval_lam \
 --traj_lbda 0.3 \
 --traj_dim 6 \
 --use_traj