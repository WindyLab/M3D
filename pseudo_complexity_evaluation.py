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
        adapt_dataset='RealDataset4',  # save results to project/name
):
    trainval_Path = '/home/zy/data/zhangyin/datasets/' + adapt_dataset + '/trainval'
    AnnoPath = str(ROOT / project / name / 'labels/') + '/'
    save_path = str(ROOT / project / name) + '/'
    classes = ['Drone', 'UAV']
    def convert(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    files = os.listdir(AnnoPath)
    files.sort()
    small_thresh_txt = open(save_path + 'small_thresh.txt', 'w')
    complex_thresh_txt = open(save_path + 'complex_thresh.txt', 'w')
    contrast_thresh_txt = open(save_path + 'contrast_thresh.txt', 'w')
    simple_thresh_txt = open(save_path + 'simple_thresh.txt', 'w')
    for im in range(0, len(files)):
        image_name, ext = os.path.splitext(files[im])
        if ext == '.txt':
            if adapt_dataset == 'MDM-Sim-All-Target':
                imgfile = trainval_Path + '/' + image_name + '.png'
            else:
                imgfile = trainval_Path + '/' + image_name + '.jpg'
            txtfile = AnnoPath + image_name + '.txt'
            if os.path.exists(txtfile):

                # print(image)
                # 打开txt文档
                data = open(txtfile, 'r')
                # 得到文档元素对象
                img = cv.imread(imgfile)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                print(im, imgfile)
                [h, w] = img.shape
                data_content = data.readlines()
                for num in range(len(data_content)):
                    line = data_content[num]
                    line = line.strip('\n')
                    line_content = line.split()
                    object_name = classes[int(line_content[0])]

                    x_center = float(line_content[1]) * w
                    y_center = float(line_content[2]) * h
                    width = float(line_content[3]) * w
                    height = float(line_content[4]) * h
                    thresh = line_content[5]
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    fore_img = img[y1:y2, x1:x2]
                    h_fore, w_fore = fore_img.shape
                    area = h_fore * w_fore
                    addwidth = width * 0.25
                    addheight = height * 0.25

                    x_min = int(max(x1 - addwidth, 0))
                    y_min = int(max(y1 - addheight, 0))
                    x_max = int(min(x2 + addwidth, w))
                    y_max = int(min(y2 + addheight, h))
                    #print(x_min, x_max, y_min, y_max)
                    if x_max == x_min:
                        continue
                    if x_max < 0:
                        continue
                    if y_max < y2:
                        y_max = y2
                    new_image = img[y_min:y_max, x_min:x_max]
                    h_whole, w_whole = new_image.shape
                    #cv.imwrite(save_path + '/' + label_pre + num_box_name + '.jpg', new_image)  # save picture

                    # Calculate the average intensity of the foreground image
                    avg_I = np.sum(fore_img) / (h_fore * w_fore)
                    dif_fore = 0
                    # Calculate the difference of the foreground image
                    for i in range(h_fore):
                        for j in range(w_fore):
                            dif_fore = dif_fore + (fore_img[i,j] - avg_I) ** 2
                    #Calculate the difference of the whole image
                    dif_whole = 0
                    for i in range(h_whole):
                        for j in range(w_whole):
                            dif_whole = dif_whole + (new_image[i, j] - avg_I) ** 2
                    #Calculate the contrast value
                    contrast = ((dif_whole - dif_fore) / (h_whole * w_whole - h_fore * w_fore)) ** 0.5
                    print(contrast)
                    stacked_img = np.stack((new_image,) * 3, axis=-1)
                    #Histgram
                    fore_flatten = fore_img.flatten()
                    whole_flatten = new_image.flatten()
                    #plt.figure(save_path + '/' + label_pre + '.jpg')

                    #n1, bins1, patches1 = plt.hist(whole_flatten, bins=256, facecolor='red', alpha=0.5)
                    #n2, bins2, patches2 = plt.hist(fore_flatten, bins=256, facecolor='blue', alpha=0.5)
                    #back_hist = n1 - n2
                    array_back = []
                    for i in range(y1 - y_min):
                        for j in range(w_whole):
                            array_back.append(new_image[i][j])
                    for i in range(h_whole-(y_max - y2), h_whole):
                        for j in range(w_whole):
                            array_back.append(new_image[i][j])
                    #pdb.set_trace()
                    for i in range(y1 - y_min + 1, h_whole-(y_max - y2)):
                        for j in range(x1 - x_min):
                            array_back.append(new_image[i][j])
                    for i in range(y1 - y_min + 1, h_whole - (y_max - y2)):
                        for j in range(w_whole-(x_max - x2), w_whole):
                            array_back.append(new_image[i][j])
                    #n3, bins3, patches3 = plt.hist(array_back, bins=256, facecolor='green', alpha=0.5)
                    #plt.savefig(save_path + '/' + label_pre + '_hist.jpg')
                    #pdb.set_trace()
                    arr_std = np.std(array_back, ddof=1)
                    content = str(round(arr_std, 2))
                    if area <= 16*16:
                        small_thresh_txt.write(image_name + ' ' + str(num) + ' ' + thresh + '\n')
                    if area > 16*16 and contrast <= 10:
                        contrast_thresh_txt.write(image_name + ' ' + str(num) + ' ' + thresh + '\n')
                    if area > 16*16 and contrast > 10 and arr_std <= 10:
                        simple_thresh_txt.write(image_name + ' ' + str(num) + ' ' + thresh + '\n')
                    if area > 16*16 and contrast > 10 and arr_std > 10:
                        #complexity_num += 1
                        #complexity_txt.write(image_line)
                        complex_thresh_txt.write(image_name + ' ' + str(num) + ' ' + thresh + '\n')
                        #cv.putText(stacked_img, content, (10, 10), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=1)
                        #cv.imwrite(save_path + '/' + label_pre + '.jpg', stacked_img)  # save picture
                #pdb.set_trace()
                data.close()
    small_thresh_txt.close()
    complex_thresh_txt.close()
    contrast_thresh_txt.close()
    simple_thresh_txt.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--adapt-dataset', default='RealDataset4', help='save results to project/name')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
