import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET

cross_scene_dataset_root = './cross_dataset/' # Change to your final result path
scene_path = './Drone-vs-bird/images/' # Change to your target image path
scene_label_path = './Drone-vs-bird/labels/' # Change to your target image path
source_seg = './cropped_images/masks/' # Change to your saliency mask path
source_path = './cropped_images/images/' # Change to your saliency image path
box_path = './cropped_images/labels_xml/' # Change to your saliency label path
result_path = cross_scene_dataset_root + '/images/'
result_label_path = cross_scene_dataset_root + '/labels/'
adapt_dataset_name = 'Drone-vs-Bird'

files = os.listdir(scene_path)
model_files = os.listdir(source_seg)

classes = ['UAV', 'Drone']

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

files.sort()

for i in range(0, len(files)):
    file = files[i]
#pdb.set_trace()
if file.split('.')[1] == 'jpg' or file.split('.')[1] == 'png':
    filename = file.split('.')[0]
    print(scene_path + '/' + file)
    image = cv2.imread(scene_path + '/' + file)
    h, w, c = image.shape
    txt_name = scene_label_path + '/' + filename + '.txt'
    txt = open(txt_name)
    content = txt.readlines()
    txt.close()
    new_image = image

    txt_file_name = result_label_path + '/' + filename + '.txt'
    txt_file = open(txt_file_name, 'w')
    for line in content:
        oneline = line.strip().split(" ")
        x_center, y_center, width, height = float(oneline[1]) * w, float(oneline[2]) * h, float(
            oneline[3]) * w, float(oneline[4]) * h
        x_min = int(x_center - width / 2)
        x_max = int(x_center + width / 2)
        y_min = int(y_center - height / 2)
        y_max = int(y_center + height / 2)
        txt_file.write(line)
    for n in range(3):
        rm = random.randint(0, len(model_files) - 1)
        model = model_files[rm]
        model_name = model.split('.')[0]
        src_ori = cv2.imread(source_path + '/' + model_name + '.jpg')

        src_mask_ori = cv2.imread(source_seg + '/' + model_name + '.png')
        src_box = open(box_path + '/' + model_name + '.xml')
        h_source_ori, w_source_ori, c_source_ori = src_ori.shape
        tree = ET.parse(src_box)
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w_source_ori, h_source_ori), b)

        addwidth = 10

        box_xmin = int(max(b[0] - addwidth, 0))
        box_xmax = int(min(b[1] + addwidth, w_source_ori))
        box_ymin = int(max(b[2] - addwidth, 0))
        box_ymax = int(min(b[3] + addwidth, h_source_ori))

        src_ori = src_ori[box_ymin:box_ymax, box_xmin:box_xmax, :]
        src_mask_ori = src_mask_ori[box_ymin:box_ymax, box_xmin:box_xmax, :]
        h_source_ori, w_source_ori, c_source_ori = src_ori.shape
        b = ((b[0] - box_xmin), (b[1] - box_xmin), (b[2] - box_ymin), (b[3] - box_ymin))
        bb = convert((w_source_ori, h_source_ori), b)

        # Resize source images

        ratio = min(h_source_ori / height, w_source_ori / width)

        src = cv2.resize(src_ori, None, fx=(1 / ratio), fy=(1 / ratio))
        src_mask = cv2.resize(src_mask_ori, None, fx=(1 / ratio), fy=(1 / ratio))

        h_source, w_source, c_source = src.shape

        # This is where the CENTER of the airplane will be placed
        center_xmin = int(w_source / 2)
        center_xmax = int(w - w_source / 2)
        center_ymin = int(h_source / 2)
        center_ymax = int(h - h_source / 2)

        if center_ymax < center_ymin or center_xmax < center_xmin:
            continue

        center_x = random.randint(center_xmin, center_xmax - 1)
        center_y = random.randint(center_ymin, center_ymax - 1)

        if center_x < center_xmin:
            center_x = center_xmin
        if center_x > center_xmax:
            center_x = center_xmax
        if center_y < center_ymin:
            center_y = center_ymin
        if center_y > center_ymax:
            center_y = center_ymax

        center = (center_x, center_y)

        ## b.设置卷积核5*5
        kernel = np.ones((10, 10), np.uint8)

        ## c.图像的腐蚀，默认迭代次数
        erosion = cv2.dilate(src_mask, kernel, 10)

        src_mask = erosion
        print(i, src.shape, new_image.shape, center)
        if src.shape[0] <= 4:
            continue
        # Normal Cloning
        output1 = cv2.seamlessClone(src, new_image, src_mask, center, cv2.NORMAL_CLONE)


        # Generate txt

        x0 = center_x - w_source / 2
        y0 = center_y - h_source / 2

        x_min = int((bb[0] - bb[2] / 2) * w_source + x0)
        x_max = int((bb[0] + bb[2] / 2) * w_source + x0)

        y_min = int((bb[1] - bb[3] / 2) * h_source + y0)
        y_max = int((bb[1] + bb[3] / 2) * h_source + y0)

        back_xmin = int(center_x - w_source / 2)
        back_ymin = int(center_y - h_source / 2)
        back_xmax = int(center_x + w_source / 2)
        back_ymax = int(center_y + h_source / 2)

        new_box = convert((w, h), (x_min, x_max, y_min, y_max))
        txt_file.write('0' + " " + " ".join([str(a) for a in new_box]) + '\n')
        new_image = output1
    txt_file.close()
    cv2.imwrite(result_path + '/' + file, output1)
