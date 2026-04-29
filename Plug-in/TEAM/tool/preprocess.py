import os
import sys
import shutil
import threading
import argparse
import json


def extract(source_path, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    cmd = (
        'ffmpeg -i "{}" -threads 1 '
        '-vf "scale=\'if(gt(iw,ih),-1,256)\':\'if(gt(iw,ih),256,-1)\'" '
        '-q:v 0 "{}/%06d.jpg"'
    ).format(source_path, target_dir)
    os.system(cmd)


def process_video_list(video_list):
    for source_path, target_dir in video_list:
        extract(source_path, target_dir)


def organize_dir(args, split):
    if args.dataset == 'HMDB51':
        txt_dir = os.path.join('splits', 'hmdb')
        target = 'hmdb51_FSAR'
    elif args.dataset == 'Kinetics':
        txt_dir = os.path.join('splits', 'kinetics')
        target = 'kinetics_FSAR'
    elif args.dataset == 'UCF101':
        txt_dir = os.path.join('splits', 'ucf')
        target = 'ucf101_FSAR'
    elif args.dataset == 'SSv2-Small':
        txt_dir = os.path.join('splits', 'ssv2_small')
        target = 'ssv2_small_FSAR'
    elif args.dataset == 'Finesports':
        txt_dir = os.path.join('splits', 'Finesports')
        target = 'Finesports_FSAR'
    elif args.dataset == 'Multisports':
        txt_dir = os.path.join('splits', 'Multisports')
        target = 'Multisports_FSAR'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    txt_file = os.path.join(txt_dir, '{}list.txt'.format(split))
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    video_pairs = []
    for line in lines:
        class_name, dir_name = line.strip().rsplit('/', 1)

        if args.dataset == 'SSv2-Small':
            candidates = [os.path.join(args.source_dir, dir_name + '.webm')]
        else:
            # 先試「原本的資料夾名」（含空格），再試「空格轉底線」
            candidates = [
                os.path.join(args.source_dir, class_name, dir_name + '.avi'),
                os.path.join(args.source_dir, class_name.replace(' ', '_'), dir_name + '.avi'),
            ]

        source_path = next((p for p in candidates if os.path.exists(p)), None)

        target_dir = os.path.join(target, split, class_name, dir_name)

        if source_path is not None:
            video_pairs.append((source_path, target_dir))
        else:
            print(f"[!] Missing: {candidates[0]}  (also tried underscore variant)")
    
    # Split for multithreading
    chunk_size = (len(video_pairs) + args.num_threads - 1) // args.num_threads
    chunks = [video_pairs[i:i+chunk_size] for i in range(0, len(video_pairs), chunk_size)]

    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=process_video_list, args=(chunk,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, 
                        help='Path to the source directory containing videos')
    parser.add_argument('--dataset', type=str, default='HMDB51', 
                        choices=['HMDB51', 'UCF101', 'Kinetics', 'SSv2-Small','Finesports','Multisports'])
    parser.add_argument('--num_threads', type=int, default=100)
    args = parser.parse_args()
    
    for split in ['train', 'val', 'test']:
        organize_dir(args, split)


# python3 preprocess.py --dataset HMDB51 --source_dir /home/mmlab206/61347023S/Datasets/HMDB51
# python3 preprocess.py --dataset Kinetics
# python3 preprocess.py --dataset UCF101 --source_dir /home/subeenlee/code/Research/ucf101/videos
# python3 preprocess.py --dataset SSv2-Small --source_dir /home/subeenlee/code/Research/ssv2