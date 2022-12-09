#!/usr/bin/env python3
from nav_cloning_pytorch import *
import cv2
import csv
from skimage.transform import resize
import time
import os

class cource_following_learning_node:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/" + self.start_time)
        self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")
        self.ang_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/ang/dist025/")
        self.img_right_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/img/dist025/right")
        self.img_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/img/dist025/center")
        self.img_left_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/img/dist025/left")
        self.learn_no = 10000
        
        # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")

    def learn(self):
        ang_list = []
        img_right_list = []
        img_list = []
        img_left_list = []
        for i in range(886):
        # for i in range(1687):
            for j in ["-7", "-5", "-3", "0", "+3", "+5", "+7"]:
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

        # for k in range(1687 * 7):
        for k in range(886 * 3):
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
            
            # self.dl.make_dataset(imgobj_right, target_ang + 0.2)
            # self.dl.make_dataset(imgobj, target_ang)
            # self.dl.make_dataset(imgobj_left, target_ang - 0.2)
            self.dl.make_dataset(img_right, target_ang + 0.2)
            self.dl.make_dataset(img, target_ang)
            self.dl.make_dataset(img_left, target_ang - 0.2)
            print("dataset:" + str(k))

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
         


