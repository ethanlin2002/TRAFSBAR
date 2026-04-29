"""
Point tracking module for extracting and tracking semantic points in videos.

This module provides functionality to:
- Extract semantic points from videos using clustering methods OR YOLO bounding boxes
- Track points across video frames using CoTracker
- Save tracking results and generate visualizations
"""
import sys
import gc
import os
import time
import random
import argparse
import pickle
import torch
import numpy as np
from einops import rearrange
import pandas as pd
from utils import convert_points_for_tracking, save_video
from feat_extractor import feature_extract
from get_semantic_points import get_points_from_clustering
from new_video_loader import load_video_pyvideo_reader
from omni_vis import vis_trail

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# set seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = '/fs/cfar-projects/actionloc/camera_ready/tats_v2/dumps'

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
# pylint: disable=redefined-outer-name


def check_columns_in_df(df):
    """Check if the dataframe has the required columns.

    Args:
        df (pd.DataFrame): Dataframe to check

    Raises:
        ValueError: If the dataframe does not have the required columns
    """
    required_columns = ['video_path', 'dataset']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the dataframe")


# ========= YOLO helper =========

def sample_yolo_queries_from_video(
    video_np,
    num_frames_clustering,
    yolo_model,
    num_points_per_box,
    conf_thres,
    yolo_max_frames=None,
    class_whitelist=None,
):
    """
    使用 YOLO 在影片中偵測物件，並在 bbox 內撒點，產生 CoTracker queries。

    Args:
        video_np: numpy array, shape (T, H, W, C), uint8 or float in [0, 255]
        num_frames_clustering: 取多少個 frame 來撒點（類似原本 clustering 的 frame 數）
        yolo_model: 已載好的 YOLO model
        num_points_per_box: 每個 bbox 撒多少點（可以後續自行改為依 class 調整）
        conf_thres: YOLO 信心門檻
        yolo_max_frames: 最多跑多少 frame 的 YOLO（防止影片太長）
        class_whitelist: 若不為 None，則只保留這些 class id

    Returns:
        queries_list: list of [frame_idx, x, y]
        obj_ids: list of int, 代表每個 query point 的 obj / class id
    """
    T, H, W, C = video_np.shape

    # 轉成 uint8 給 YOLO 比較保險
    if video_np.dtype != np.uint8:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)

    # 選擇要跑 YOLO 的 frame index
    if num_frames_clustering is None or num_frames_clustering <= 0:
        frame_indices = list(range(T))
    else:
        num = min(num_frames_clustering, T)
        if yolo_max_frames is not None:
            num = min(num, yolo_max_frames)
        # 均勻抽樣 num 個 index
        if num == T:
            frame_indices = list(range(T))
        else:
            frame_indices = np.linspace(0, T - 1, num=num, dtype=int).tolist()

    queries_list = []
    obj_ids = []

    # Ultralytics YOLO: model(frame) 回傳 list，取 [0]
    for t in frame_indices:
        frame = video_np[t]  # H, W, C (RGB)
        results = yolo_model(frame, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes
        xyxy = boxes.xyxy.cpu().numpy()  # (N_det, 4)
        cls = boxes.cls.cpu().numpy().astype(int)  # (N_det,)
        conf = boxes.conf.cpu().numpy()  # (N_det,)

        for det_idx in range(xyxy.shape[0]):
            if conf[det_idx] < conf_thres:
                continue
            cls_id = int(cls[det_idx])
            if class_whitelist is not None and cls_id not in class_whitelist:
                continue

            x1, y1, x2, y2 = xyxy[det_idx]
            # Clip to image bounds
            x1 = float(np.clip(x1, 0, W - 1))
            x2 = float(np.clip(x2, 0, W - 1))
            y1 = float(np.clip(y1, 0, H - 1))
            y2 = float(np.clip(y2, 0, H - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # 目前簡單做法：每個 bbox 撒固定數量的點
            for _ in range(num_points_per_box):
                x = float(np.random.uniform(x1, x2))
                y = float(np.random.uniform(y1, y2))
                queries_list.append([t, x, y])
                # obj_id 先直接用 class id，你之後想 encode 成 (player/ball/others) 可自行擴充
                obj_ids.append(cls_id)

    return queries_list, obj_ids


def extract_points(
    args,
    cotracker,
    feat_extractor,
    video_path,
    ds_dump_path,
    custom_fps=None,
    yolo_model=None,
):
    """Extract points from a video and save them to a pickle file.

    Args:
        args (argparse.Namespace): Arguments
        cotracker (torch.nn.Module): Cotracker model
        feat_extractor (torch.nn.Module): Feature extractor model
        video_path (str): Path to the video
        ds_dump_path (str): Path to the directory where the pickle file will be saved
        custom_fps (int): Custom fps to use for the video if video duration > 90s
        yolo_model: YOLO model if using YOLO points

    Returns:
        bool: True if the points were extracted, False otherwise
    """
    vid_name = video_path.split('/')[-1].split('.')[0]
    debug_vis_dump_root = os.path.join(ds_dump_path, 'debug_vis', vid_name)
    feat_dump_path = os.path.join(ds_dump_path, 'feat_dump', f'{vid_name}.pkl')
    gif_dump_path = os.path.join(ds_dump_path, 'gif_dump', f'{vid_name}.gif')

    if os.path.exists(feat_dump_path) and not args.rerun:
        return True

    # ============================================================
    # ============== YOLO-based semantic point branch ============
    # ============================================================
    if getattr(args, "use_yolo_points", False):
        if yolo_model is None:
            raise ValueError("use_yolo_points=True，但沒有提供 yolo_model。")

        # 直接載全影片，給 YOLO + CoTracker 共用
        video_loaded, video, _ = load_video_pyvideo_reader(
            video_path,
            return_tensor=True,
            use_float=True,
            device=args.device,
            sample_all_frames=True,
            fps=custom_fps if custom_fps is not None else args.fps,
        )  # B T C H W

        if not video_loaded:
            print(f"[YOLO] Video {vid_name} not loaded")
            return None

        # 轉給 YOLO 用的 numpy (T, H, W, C)
        video_np = video.cpu().squeeze(0).numpy()  # T C H W
        video_np = rearrange(video_np, "t c h w -> t h w c")

        if args.debug_mode:
            time_start = time.time()

        # 在 YOLO bbox 內撒點，產生 queries
        queries_list, obj_ids = sample_yolo_queries_from_video(
            video_np=video_np,
            num_frames_clustering=args.num_frames_clustering,
            yolo_model=yolo_model,
            num_points_per_box=args.yolo_points_per_box,
            conf_thres=args.yolo_conf_thres,
            yolo_max_frames=args.yolo_max_frames,
            class_whitelist=args.yolo_class_whitelist,
        )

        if len(queries_list) == 0:
            print(f"[YOLO] No queries generated for video {vid_name}, skip.")
            return None

        # 轉成 CoTracker 需要的 queries tensor: (B, N, 3) -> [frame_idx, x, y]
        queries_points = torch.tensor(queries_list, device=args.device, dtype=torch.float32)[None, ...]  # 1 x N x 3
        cluster_ids_all_frames = np.array(obj_ids, dtype=np.int64)

        if args.debug_mode:
            time_end = time.time()
            print(f"[YOLO] Time taken to get YOLO-based points: {time_end - time_start:.2f} seconds")

        torch.cuda.empty_cache()

        if args.debug_mode:
            time_start = time.time()

        if args.use_grid:
            pred_tracks, pred_visibility = cotracker(
                video,
                grid_size=args.cotracker_grid_size,
                queries=None,
                backward_tracking=False,
            )
        else:
            pred_tracks, pred_visibility = cotracker(
                video,
                queries=queries_points,
                backward_tracking=True,
            )

        if args.debug_mode:
            time_end = time.time()
            print(f"[YOLO] Time taken to run CoTracker: {time_end - time_start:.2f} seconds")

        point_queries = queries_points.cpu().squeeze(0).numpy()[:, 0]  # frame index for each query
        pred_tracks = pred_tracks.cpu().squeeze(0).numpy()    # T x N x 2
        pred_visibility = pred_visibility.cpu().squeeze(0).numpy()  # T x N x 1 or T x N
        video_for_vis = video.cpu().squeeze(0).numpy()        # T x C x H x W
        video_for_vis = rearrange(video_for_vis, "t c h w -> t h w c")

        pt_obj_cluster_dict = {}

        dump_dict = {
            "pred_tracks": torch.tensor(pred_tracks).half(),
            "pred_visibility": torch.tensor(pred_visibility).bool(),
            "obj_ids": torch.tensor(cluster_ids_all_frames).long(),
            "point_queries": torch.tensor(point_queries).long(),
            **pt_obj_cluster_dict,
        }

        os.makedirs(os.path.dirname(feat_dump_path), exist_ok=True)
        pickle.dump(dump_dict, open(feat_dump_path, "wb"))
        torch.cuda.empty_cache()

        if args.debug_mode or args.make_vis:
            frames = vis_trail(video_for_vis, pred_tracks, pred_visibility, cluster_ids=cluster_ids_all_frames)
            os.makedirs(os.path.dirname(gif_dump_path), exist_ok=True)
            save_video(frames, gif_dump_path)

        return True

    # ============================================================
    # ============== 原本 DINO clustering pipeline ===============
    # ============================================================

    # load video for DINO feat extractor (只取 num_frames_clustering 幀)
    video_loaded, video_frames, frames_id_dict = load_video_pyvideo_reader(
        video_path,
        return_tensor=True,
        use_float=False,
        num_frames=args.num_frames_clustering,
        sample_all_frames=False,
        fps=custom_fps if custom_fps is not None else args.fps,
    )  # (B, T, C, H, W)

    if not video_loaded:
        print(f"Video {vid_name} not loaded")
        return None

    video_frames = rearrange(video_frames, "b t c h w -> b t h w c")
    video_frames = video_frames.cpu().numpy()

    if args.debug_mode:
        time_start = time.time()

    base_point_info = get_points_from_clustering(
        args, video_frames, feat_extractor, debug_vis_dump_root
    )
    points_list, point_labels_list, component_labels_list = base_point_info

    queries_points, cluster_ids_all_frames = convert_points_for_tracking(
        points_list,
        point_labels_list,
        frames_id_dict=frames_id_dict,
        component_labels_list=component_labels_list,
        use_connected_components=args.use_connected_components,
        device=args.device,
    )

    if args.debug_mode:
        os.makedirs(debug_vis_dump_root, exist_ok=True)
        time_end = time.time()
        print(f"Time taken to get points and labels: {time_end - time_start} seconds")

    torch.cuda.empty_cache()

    # 再載一次 full video，給 CoTracker 用
    _, video, _ = load_video_pyvideo_reader(
        video_path,
        return_tensor=True,
        use_float=True,
        device=args.device,
        sample_all_frames=True,
        fps=custom_fps if custom_fps is not None else args.fps,
    )  # B T C H W

    if args.debug_mode:
        time_start = time.time()

    if args.use_grid:
        pred_tracks, pred_visibility = cotracker(
            video,
            grid_size=args.cotracker_grid_size,
            queries=None,
            backward_tracking=False,
        )
    else:
        pred_tracks, pred_visibility = cotracker(
            video,
            queries=queries_points,
            backward_tracking=True,
        )

    if args.debug_mode:
        time_end = time.time()
        print(f"Time taken to run cotracker: {time_end - time_start} seconds")

    point_queries = queries_points.cpu().squeeze(0).numpy()[:, 0]
    pred_tracks = pred_tracks.cpu().squeeze(0).numpy()
    pred_visibility = pred_visibility.cpu().squeeze(0).numpy()
    video = video.cpu().squeeze(0).numpy()
    video = rearrange(video, "t c h w -> t h w c")
    pt_obj_cluster_dict = {}

    dump_dict = {
        "pred_tracks": torch.tensor(pred_tracks).half(),
        "pred_visibility": torch.tensor(pred_visibility).bool(),
        "obj_ids": torch.tensor(cluster_ids_all_frames).long(),
        "point_queries": torch.tensor(point_queries).long(),
        **pt_obj_cluster_dict,
    }

    os.makedirs(os.path.dirname(feat_dump_path), exist_ok=True)
    pickle.dump(dump_dict, open(feat_dump_path, "wb"))
    torch.cuda.empty_cache()

    if args.debug_mode or args.make_vis:
        frames = vis_trail(video, pred_tracks, pred_visibility, cluster_ids=cluster_ids_all_frames)
        os.makedirs(os.path.dirname(gif_dump_path), exist_ok=True)
        save_video(frames, gif_dump_path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_mode", action="store_true",
                        help="Enable debug mode")

    parser.add_argument("--use_connected_components", action="store_true",
                        help="Use connected components")

    parser.add_argument("--num_frames_clustering", type=int, default=32,
                        help="Number of frames to cluster / use for YOLO sampling")

    parser.add_argument("--merge_ratio", type=int, default=25,
                        help="Merge ratio")

    parser.add_argument("--num_iters", type=int, default=11,
                        help="Number of iterations")

    parser.add_argument("--clustering_method", type=str, default='bipartite',
                        help="Clustering method to use")

    parser.add_argument("--n_clusters", type=int, default=32,
                        help="Number of clusters")

    parser.add_argument("--num_points_per_entity", type=int, default=16,
                        help="Number of samples per mask (DINO branch)")

    parser.add_argument("--use_grid", action="store_true",
                        help="Use grid")

    parser.add_argument("--cotracker_grid_size", type=int, default=16,
                        help="Cotracker grid size")

    parser.add_argument("--csv_path", type=str, default='sample.csv',
                        help='Path to csv file')

    parser.add_argument("--fps", type=int, default=None,
                        help="FPS for point tracking")

    parser.add_argument("--base_feat_path", type=str, default=BASE_PATH,
                        help="Base path for feature dumps")

    parser.add_argument("--make_vis", action="store_true",
                        help="Make gifs")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun the point tracking")

    # ===== YOLO-related args =====
    parser.add_argument("--use_yolo_points", action="store_true",
                        help="Use YOLO detections instead of DINO clustering to sample query points")
    parser.add_argument("--yolo_weights", type=str, default=None,
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--yolo_conf_thres", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--yolo_points_per_box", type=int, default=8,
                        help="Number of points to sample per YOLO bbox")
    parser.add_argument("--yolo_max_frames", type=int, default=None,
                        help="Max number of frames to run YOLO on (None = no limit)")
    # class_whitelist 先用簡單的逗號字串表示，例如 \"0,1\"，你可自行在程式裡轉成 list[int]
    parser.add_argument("--yolo_class_whitelist", type=str, default=None,
                        help="Comma-separated class ids to keep, e.g. '0,1'. None = keep all classes.")

    args = parser.parse_args()

    # 處理 class_whitelist 成 list[int] 或 None
    if args.yolo_class_whitelist is None:
        args.yolo_class_whitelist = None
    else:
        args.yolo_class_whitelist = [int(x) for x in args.yolo_class_whitelist.split(",") if x != ""]

    use_connected_components = args.use_connected_components

    if args.clustering_method == 'kmeans':
        CLUSTER_STR = f'kmeans_n{args.n_clusters}'
    elif args.clustering_method == 'bipartite':
        CLUSTER_STR = 'bip'
    else:
        raise ValueError(f"Invalid clustering method: {args.clustering_method}")

    df = pd.read_csv(args.csv_path)
    check_columns_in_df(df)
    if args.debug_mode:
        # 這邊你可以換成你資料集裡的一個 video_name 來 debug
        # df = df[df['video_name'] == '27k-12-1-2|P1|6116|6514.mp4']
        # 或直接只取一筆
        df = df.iloc[:1]
        args.rerun = True
        args.make_vis = True

    dump_name = f'cotracker3_{CLUSTER_STR}_fr_{args.num_frames_clustering}'
    if args.merge_ratio != 25 or args.num_iters != 11:  # if not default then add to dump name
        dump_name += f'_m{args.merge_ratio}_i{args.num_iters}'
    if use_connected_components:
        dump_name += '_concomp'
    if args.fps is not None:
        dump_name += f'_fps_{args.fps}'
    if args.use_yolo_points:
        dump_name += '_yolo'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(args, 'device', device)

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    feat_extractor = feature_extract()

    # 載 YOLO 模型
    yolo_model = None
    if args.use_yolo_points:
        if args.yolo_weights is None:
            raise ValueError("請提供 --yolo_weights 路徑（你的微調 YOLO 模型）")
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_weights)
        # ultralytics 會自動選裝置，你要強制用 GPU 可以這樣：
        # yolo_model.to(str(device))

    # 先把 df 轉成可操作的 list
    pending_rows = df.to_dict('records')

    while len(pending_rows) > 0:
        vid_info_row = pending_rows[0]   # 目前要處理的就是第 0 個
        dataset = vid_info_row['dataset']
        video_path = vid_info_row['video_path']

        if 'duration' in vid_info_row:
            duration = vid_info_row['duration']
            if duration > 90:
                custom_fps = 1
            else:
                custom_fps = None
        else:
            custom_fps = None

        video_uniq_id = video_path.split('/')[-1].split('.')[0]
        feat_dump_name = f'{video_uniq_id}'
        ds_dump_path = os.path.join(args.base_feat_path, dump_name, dataset)

        try:
            extract_points(
                args,
                cotracker,
                feat_extractor,
                video_path,
                ds_dump_path,
                custom_fps=custom_fps,
                yolo_model=yolo_model,
            )

            print(f"[OK] 成功處理: {video_path}")

            # 成功後，把目前這筆從待處理清單移除
            pending_rows.pop(0)

        except torch.OutOfMemoryError:
            print(f"[OOM] torch.OutOfMemoryError: {video_path}")
            torch.cuda.empty_cache()
            gc.collect()

            # 把目前這筆移到最後
            failed_row = pending_rows.pop(0)
            pending_rows.append(failed_row)

            print(f"[REQUEUE] 已將失敗項目移到最後，剩餘待處理數: {len(pending_rows)}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] RuntimeError OOM: {video_path}")
                torch.cuda.empty_cache()
                gc.collect()

                # 把目前這筆移到最後
                failed_row = pending_rows.pop(0)
                pending_rows.append(failed_row)
                print(f"[REQUEUE] 已將失敗項目移到最後，剩餘待處理數: {len(pending_rows)}")
            else:
                raise
        except ValueError:
            # 把目前這筆移到最後
            failed_row = pending_rows.pop(0)
            pending_rows.append(failed_row)
        
