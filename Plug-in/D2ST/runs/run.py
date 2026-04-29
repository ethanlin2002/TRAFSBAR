#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.abspath(os.curdir))

from utils.config import Config

def main():
    cfg = Config(load=True)

    # 必須盡量早設定，最好在 import torch / model / train_net 之前
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)

    from utils.launcher import launch_task
    from train_net_few_shot import train_few_shot
    from test_net_few_shot import test_few_shot

    def _prepare_data(cfg):
        train_func = train_few_shot
        test_func = test_few_shot
        run_list = []

        if cfg.TRAIN.ENABLE:
            cfg.SUPPORT_SHOT = cfg.TRAIN.SHOT
            run_list.append([cfg.deep_copy(), train_func])

        if cfg.TEST.ENABLE:
            for i in cfg.TEST.SHOT:
                cfg.SUPPORT_SHOT = i
                run_list.append([cfg.deep_copy(), test_func])

        return run_list

    run_list = _prepare_data(cfg)

    for run in run_list:
        launch_task(cfg=run[0], init_method=run[0].INIT_METHOD, func=run[1])

    print("Finish running with config: {}".format(cfg.args.cfg_file))


if __name__ == "__main__":
    main()