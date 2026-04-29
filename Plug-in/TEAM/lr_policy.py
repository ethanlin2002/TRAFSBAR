import math


def get_epoch_lr(args, cur_epoch):
    lr = lr_func_steps_with_relative_lrs(args, cur_epoch)
    return lr


def lr_func_steps_with_relative_lrs(args, cur_epoch):
    ind = get_step_index(args, cur_epoch)
    return args.lrs[ind] * args.learning_rate


def get_step_index(args, cur_epoch):
    steps = args.steps + [args.max_epoch]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1
