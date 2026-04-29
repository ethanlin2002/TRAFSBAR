python point_tracking/new_point_tracking.py \
  --csv_path TRAFSBAR/Traj-sample/point_tracking/sample.csv \
  --base_feat_path ./ \
  --num_frames_clustering 32 \
  --use_yolo_points \
  --yolo_weights Checkpoint/YOLO_v8_basket.pt \
  --yolo_class_whitelist 0,1  \
  --yolo_points_per_box 8 \
  --query_batch_size 512
