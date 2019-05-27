##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Weinong Wang
## Tencent, Youtu
## Email: weinong.wang@hotmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import glob
import numpy as np

split_ratio = 0.1

images_path = '/media/youtu/本地磁盘1/data2/way/images'

out_base_path = '/media/youtu/本地磁盘1/data2/way/labels'
train_file = os.path.join(out_base_path, 'train1.txt')
val_file = os.path.join(out_base_path,'val1.txt')

f_train = open(train_file, 'w')
f_val = open(val_file, 'w')

class_dir = os.listdir(images_path)

for class_i in class_dir:
    class_i_path = os.path.join(images_path, class_i + '/*jpg')
    img_list = glob.glob(class_i_path)

    num_val = int(np.ceil(len(img_list) * split_ratio))

    # train set
    for img_i in img_list[0:-num_val]:
        target_path = os.path.join('images', class_i , img_i.split('/')[-1])
        f_train.write(target_path+"\n")

    # val set
    for img_i in img_list[-num_val:]:
        target_path = os.path.join('images', class_i , img_i.split('/')[-1])
        f_val.write(target_path+"\n")

f_train.close()
f_val.close()