import os
import argparse


def remain_only_best(dir):
    eval_txt = os.path.join(dir, 'eval.txt')
    f = open(eval_txt, 'r')
    lines = f.readlines()

    file_name_list, acc_mean_list = [], []
    for line in lines:
        line = line.split('\t')
        try:
            file_name_list.append(line[0])
            acc_mean_list.append(float(line[1].split(': ')[-1].split('+/-')[0]))
        except:
            continue

    max_acc_mean = max(acc_mean_list)
    index = acc_mean_list.index(max_acc_mean)
    max_file_name = file_name_list[index]
    del file_name_list[index]

    eval_txt = os.path.join(dir, 'eval.txt')
    with open(eval_txt, 'a') as file:
        file.write('\n\n**Best** {} {}'.format(max_file_name, max_acc_mean))
    for file_name in file_name_list:
        os.remove(os.path.join(dir, file_name))
    os.rename(os.path.join(dir, max_file_name), os.path.join(dir, 'checkpoint_best_val.pt'))
    
    print('**Best** {} {}'.format(max_file_name, max_acc_mean))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='work-16/ucf/TEAM_pos_neg/1-shot/an30')
    args = parser.parse_args()

    remain_only_best(args.dir)