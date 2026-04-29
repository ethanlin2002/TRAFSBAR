import torch
import numpy as np
import argparse
import os
from utils import print_and_log, get_log_files, TestAccuracies, aggregate_accuracy
from model import *

import torchvision
import video_reader
import random
import torch.nn.functional as F
import time
import lr_policy
import math
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.args.num_gpus > 0) else 'cpu')
        self.model = self.init_model()

        self.video_dataset = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.video_dataset, batch_size=1,
                                                        num_workers=self.args.num_workers)
        self.val_accuracies = TestAccuracies([self.args.dataset])

        self.accuracy_fn = aggregate_accuracy

    def init_model(self):        
        model = eval(self.args.method)(self.args)
        model = model.to(self.device)

        if torch.cuda.is_available() and self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--method", default="TEM", help="few-shot method to use")
        parser.add_argument("--dataset", type=str, default="data/ssv2small", help="Path to dataset")
        parser.add_argument("--tasks_per_batch", type=int, default=1,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--test_model_name", "-m", default="checkpoint_best_val.pt",
                            help="Path to model to load and test.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--eval_way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", "-qpc", type=int, default=6,
                            help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", "-qpct", type=int, default=1,
                            help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=16, help="Num dataloader workers.")
        parser.add_argument("--backbone", choices=["ResNet", "ViT"], default="ResNet")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--pretrained_backbone", "-pt", type=str, default=None,
                            help="pretrained backbone path, used by PAL")
        parser.add_argument("--val_on_test", default=False, action="store_true",
                            help="Danger: Validate on the test set, not the validation set. "
                                 "Use for debugging or checking overfitting on test set. "
                                 "Not good practice to use when developing, hyperparameter tuning or training models.")
        parser.add_argument("--pretrained_checkpoint", "-pc", type=str, default=None)
        parser.add_argument("--agg_num", type=int, default=3)

        parser.add_argument("--use_traj", action="store_true")
        parser.add_argument("--traj_root")
        parser.add_argument("--traj_lam", type=float, default=0.0)
        parser.add_argument("--traj_lbda", type=float, default=0.3)
        parser.add_argument("--traj_dim", type=int, default=64)
        parser.add_argument("--traj_dropout", type=float, default=0.1)

        
        args = parser.parse_args()
        print(args)

        if 'hmdb' in args.dataset:
            args.traintestlist = "splits/hmdb"
        if 'kinetics' in args.dataset:
            args.traintestlist = "splits/kinetics"
        if 'ucf' in args.dataset:
            args.traintestlist = "splits/ucf"
        if 'ssv2_small' in args.dataset:
            args.traintestlist = "splits/ssv2_small"
        if 'Finesports' in args.dataset:
            args.traintestlist = "splits/Finesports"
        if 'Multisports' in args.dataset:
            args.traintestlist = "splits/Multisports"

        
        args.checkpoint_dir = '/'.join(args.pretrained_checkpoint.split('/')[:-1])

        return args

    def run(self):
        if self.args.pretrained_checkpoint is not None:
            self.load_checkpoint_from_pc(self.args.pretrained_checkpoint)

        accuracy_dict = self.evaluate("test")
        for dataset in [self.args.dataset]:
            total_query_num = self.args.eval_way * self.args.query_per_class_test * self.args.num_test_tasks
            yolo= 'yolo' in self.args.traj_root
            result = "{0:}: {1:.1f}+/-{2:.1f}\t{3}/{4}\nshot={5}, K={6}, lam={7}, yolo={8}".\
                format(dataset,
                       accuracy_dict[dataset]["acc"],
                       accuracy_dict[dataset]["con"],
                       int(accuracy_dict[dataset]["count"]),
                       total_query_num,
                       self.args.shot,
                       self.args.agg_num,
                       self.args.traj_lam,
                       yolo
                       )
            result ="\n"+ self.args.pretrained_checkpoint.split('/')[-1] + "\t" + result
            print(result)

            text_file = os.path.join(self.args.checkpoint_dir, 'eval.txt')
            if os.path.isfile(text_file):
                with open(text_file, 'a') as file:
                    file.write(result)
            else:
                with open(text_file, 'w') as file:
                    file.write(result)

    def evaluate(self, mode="test"):
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.split = mode
            n_tasks = self.args.num_test_tasks

            accuracy_dict = {}
            accuracies = []
            count = 0
            iteration = 0
            item = self.args.dataset

            # 如果 dataloader 有 __len__，就用更準確的 total；沒有就用 n_tasks
            try:
                total = min(n_tasks, len(self.video_loader))
            except TypeError:
                total = n_tasks

            pbar = tqdm(self.video_loader, total=total, desc=f"Eval ({mode})", leave=False)

            for task_dict in pbar:
                if iteration >= n_tasks:
                    break
                iteration += 1

                task_dict = self.prepare_task(task_dict)

                with torch.cuda.amp.autocast():
                    model_dict = self.model(
                        task_dict['support_set'],
                        task_dict['support_labels'],
                        task_dict['target_set'],
                        None,
                        spt_traj=task_dict.get('support_traj', None),
                        tar_traj=task_dict.get('target_traj', None),
                    )

                    #target_logits = F.softmax(model_dict['logits'], dim=-1).to(task_dict["target_labels"].get_device())
                    #acc = self.accuracy_fn(target_logits, task_dict["target_labels"])
                    target_logits = model_dict['logits']
                    acc = self.accuracy_fn(target_logits, task_dict["target_labels"])
                    
                    count += (acc * target_logits.size(0)).item()
                    accuracies.append(acc.item())
                    del target_logits

                # 進度條顯示一些即時資訊（可自行刪減）
                pbar.set_postfix({
                    "acc": f"{acc.item()*100:.2f}%",
                    "mean": f"{(np.mean(accuracies)*100 if accuracies else 0):.2f}%",
                })

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            accuracy_dict[item] = {"acc": accuracy,
                                "con": confidence,
                                "count": count}
            self.video_loader.dataset.split = "train"

        return accuracy_dict

    def prepare_task(self, task_dict):
        """
        Remove first batch dimension (as we only ever use a batch size of 1) and move data to device.
        """
        for k in task_dict.keys():
            task_dict[k] = task_dict[k][0].to(self.device)
        return task_dict

    def load_checkpoint_from_pc(self, name="checkpoint.pt"):
        checkpoint = torch.load(name)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except:
            state_dict = checkpoint['model_state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=True)


def main():
    learner = Learner()
    learner.run()


if __name__ == "__main__":
    main()