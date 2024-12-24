import os
import sys
import cv2 as cv
import pdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import shutil
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def run(
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        result_project=ROOT / 'runs/detect',  # save results to project/name
):
    save_path = str(ROOT / project / name)
    result_path = str(ROOT / result_project / name / 'labels/')

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    Path(result_path).mkdir(parents=True, exist_ok=True)

    small_txt = open(save_path + '/small_thresh.txt', 'r')
    simple_txt = open(save_path + '/simple_thresh.txt', 'r')
    contrast_txt = open(save_path + '/contrast_thresh.txt', 'r')
    complexity_txt = open(save_path + '/complex_thresh.txt', 'r')

    small_list = []
    simple_list = []
    contrast_list = []
    complex_list = []

    small_content = small_txt.readlines()
    simple_content = simple_txt.readlines()
    contrast_content = contrast_txt.readlines()
    complex_content = complexity_txt.readlines()

    small_txt.close()
    simple_txt.close()
    contrast_txt.close()
    complexity_txt.close()

    for line in small_content:
        line = line.strip('\n').split(' ')
        small_list.append(float(line[2]))
        #pdb.set_trace()

    for line in simple_content:
        line = line.strip('\n').split(' ')
        simple_list.append(float(line[2]))
        #pdb.set_trace()

    for line in contrast_content:
        line = line.strip('\n').split(' ')
        contrast_list.append(float(line[2]))
        #pdb.set_trace()

    for line in complex_content:
        line = line.strip('\n').split(' ')
        complex_list.append(float(line[2]))
        #pdb.set_trace()

    bash_thresh = 0.75  #0.75 for cross_dataset cross_real 0.85 for cross_sim

    num_small = 0
    num_contrast = 0
    num_complex = 0
    num_simple = 0
    for x in range(len(small_list)):
        if small_list[x] > bash_thresh:
            num_small = num_small + 1
    small_percent = num_small / (len(small_list) + 0.1)
    print(small_percent)

    for x in range(len(complex_list)):
        if complex_list[x] > bash_thresh:
            num_complex = num_complex + 1
    complex_percent = num_complex / (len(complex_list) + 0.1)
    print(complex_percent)

    for x in range(len(contrast_list)):
        if contrast_list[x] > bash_thresh:
            num_contrast = num_contrast + 1
    contrast_percent = num_contrast / (len(contrast_list)+ 0.1)
    print(contrast_percent)

    for x in range(len(simple_list)):
        if simple_list[x] > bash_thresh:
            num_simple = num_simple + 1
    simple_percent = num_simple / (len(simple_list)+ 0.1)
    print(simple_percent)

    plt.hist([small_list, simple_list, contrast_list, complex_list], bins=30)
    plt.title("data analyze")
    plt.xlabel("height")
    plt.ylabel("rate")
    plt.savefig('small.jpg')

    max_percent = max(simple_percent, contrast_percent, complex_percent, contrast_percent)
    small_thresh = small_percent * bash_thresh / max_percent
    contrast_thresh = contrast_percent * bash_thresh / max_percent
    simple_thresh = simple_percent * bash_thresh / max_percent
    complex_thresh = complex_percent * bash_thresh / max_percent
    print(simple_thresh, contrast_thresh, small_thresh, complex_thresh)
    for x in range(len(small_content)):
        line = small_content[x].strip('\n').split(' ')
        thresh = float(line[2])

        if thresh >= small_thresh:
            txt_path = result_path + '/' + line[0] + '.txt'
            ori_txt_path = save_path + '/labels/' + line[0] + '.txt'
            ori_line_num = int(line[1])
            ori_txt = open(ori_txt_path, 'r')
            ori_txt_content = ori_txt.readlines()
            same_content = ori_txt_content[ori_line_num]
            ori_thresh = float(same_content.strip('\n').split(' ')[-1])
            res = same_content.strip('\n').split(' ')[0:5]
            if thresh != ori_thresh:
                print(line)
                pdb.set_trace()
            if os.path.isfile(txt_path):
                txt = open(txt_path, 'a')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()
            else:
                txt = open(txt_path, 'w')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()

    for x in range(len(simple_content)):
        line = simple_content[x].strip('\n').split(' ')
        thresh = float(line[2])
        if thresh >= simple_thresh:
            txt_path = result_path + '/' + line[0] + '.txt'
            ori_txt_path = save_path + '/labels/' + line[0] + '.txt'
            ori_line_num = int(line[1])
            ori_txt = open(ori_txt_path, 'r')
            ori_txt_content = ori_txt.readlines()
            same_content = ori_txt_content[ori_line_num]
            ori_thresh = float(same_content.strip('\n').split(' ')[-1])
            res = same_content.strip('\n').split(' ')[0:5]
            if thresh != ori_thresh:
                print(line)
                pdb.set_trace()
            if os.path.isfile(txt_path):
                txt = open(txt_path, 'a')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()
            else:
                txt = open(txt_path, 'w')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()

    for x in range(len(complex_content)):
        line = complex_content[x].strip('\n').split(' ')
        thresh = float(line[2])
        if thresh >= complex_thresh:
            txt_path = result_path + '/' + line[0] + '.txt'
            ori_txt_path = save_path + '/labels/' + line[0] + '.txt'
            ori_line_num = int(line[1])
            ori_txt = open(ori_txt_path, 'r')
            ori_txt_content = ori_txt.readlines()
            same_content = ori_txt_content[ori_line_num]
            ori_thresh = float(same_content.strip('\n').split(' ')[-1])
            res = same_content.strip('\n').split(' ')[0:5]
            if thresh != ori_thresh:
                print(line)
                pdb.set_trace()
            if os.path.isfile(txt_path):
                txt = open(txt_path, 'a')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()
            else:
                txt = open(txt_path, 'w')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()

    for x in range(len(contrast_content)):
        line = contrast_content[x].strip('\n').split(' ')
        thresh = float(line[2])
        if thresh >= contrast_thresh:
            txt_path = result_path + '/' + line[0] + '.txt'
            ori_txt_path = save_path + '/labels/' + line[0] + '.txt'
            ori_line_num = int(line[1])
            ori_txt = open(ori_txt_path, 'r')
            ori_txt_content = ori_txt.readlines()
            same_content = ori_txt_content[ori_line_num]
            ori_thresh = float(same_content.strip('\n').split(' ')[-1])
            res = same_content.strip('\n').split(' ')[0:5]
            if thresh != ori_thresh:
                print(line)
                pdb.set_trace()
            if os.path.isfile(txt_path):
                txt = open(txt_path, 'a')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()
            else:
                txt = open(txt_path, 'w')
                txt.write(res[0] + ' ' + res[1] + ' ' + res[2] + ' ' + res[3] + ' ' + res[4] + '\n')
                txt.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--result-project', default=ROOT / 'runs/detect', help='save results to project/name')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
