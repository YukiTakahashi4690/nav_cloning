#!/usr/bin/env python3
from nav_cloning_pytorch import *
import cv2
import csv
from skimage.transform import resize
import time
import os
import joblib

class cource_following_learning_node:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/" + self.start_time)
        self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/")
        self.ang_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/00_01/")
        self.img_right_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/00_01/right")
        self.img_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/00_01/center")
        self.img_left_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/00_01/left")
        self.learn_no = 4000
        self.pos_no = 0
        self.data_num = 1677
        
        # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")

    def learn(self):
        ang_list = []
        img_right_list = []
        img_list = []
        img_left_list = []
        for i in range(self.data_num):
              for j in ["-5", "0", "+5"]:
            # for j in ["-10", "-5", "0", "+5", "+10"]:
            # for j in ["center", "right", "left"]:
                img_right = cv2.imread(self.img_right_path + str(i) + "_" + j + ".jpg")
                img = cv2.imread(self.img_path + str(i) + "_" + j + ".jpg")
                img_left = cv2.imread(self.img_left_path + str(i) + "_" + j + ".jpg")
                img_right_list.append(img_right)
                img_list.append(img)
                img_left_list.append(img_left)

        with open(self.ang_path + 'ang.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_ang = row
                ang_list.append(float(tar_ang))
    
        for k in range(self.data_num * 3):
       
            img_right = img_right_list[k]
            img = img_list[k]
            img_left = img_left_list[k]
            target_ang = ang_list[k]

            img_right = resize(img_right, (48, 64), mode='constant')
            r, g, b = cv2.split(img_right)
            imgobj_right = np.asanyarray([r, g, b])

            img = resize(img, (48, 64), mode='constant')
            r, g, b = cv2.split(img)
            imgobj = np.asanyarray([r, g, b])

            img_left = resize(img_left, (48, 64), mode='constant')
            r, g, b = cv2.split(img_left)
            imgobj_left = np.asanyarray([r, g, b])
            
            """
            self.dl.make_dataset(imgobj_right, target_ang + 0.2)
            self.dl.make_dataset(imgobj, target_ang)
            self.dl.make_dataset(imgobj_left, target_ang - 0.2)
            """

            # if 884 <= k <= 1092:
            #     for n in range(3):
            #             self.dl.make_dataset(img_right, target_ang + 0.2)
            #             self.dl.make_dataset(img, target_ang)
            #             self.dl.make_dataset(img_left, target_ang - 0.2)
            #             print("dataset:" + str(k))

            # if k == 1093:
            #     print("--------------------")
            #     print("coner learning end!!")
            #     print("--------------------")
            #     pass
            # else:            
            self.dl.make_dataset(img_right, target_ang + 0.2)
            self.dl.make_dataset(img, target_ang)
            self.dl.make_dataset(img_left, target_ang - 0.2)
            print("dataset:" + str(k))
        # joblib.dump((self.dataset_right, self.dataset_center, self.dataset_left), open('/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/dataset/dataset.pkl', 'wb'), compress=6)

        # self.dataset_right, self.dataset_center, self.dataset_left =joblib.load(open('/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/dataset/dataset.pkl', 'rb'))
        for l in range(self.learn_no):
            loss = self.dl.trains()
            print("train" + str(l))
            with open("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/" + self.start_time + "/loss.csv", 'a') as fw:
                writer = csv.writer(fw, lineterminator='\n')
                line = [str(loss)]
                writer.writerow(line)
        
        self.dl.save(self.save_path)

if __name__ == '__main__':
    rg = cource_following_learning_node()
    rg.learn()
         


