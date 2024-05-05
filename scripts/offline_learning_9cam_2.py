#!/usr/bin/env python3
from nav_cloning_pytorch import *
import cv2
import csv
from skimage.transform import resize
import time
import os
import sys
import random 

class cource_following_learning_node:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        # os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/" + self.start_time)
        self.model_num = str(sys.argv[1])
        self.pro = "9cam"
        self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model"+str(self.model_num)+".pt")
        # self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/01/model"+str(self.model_num)+".pt")
        self.ang_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/"+str(self.pro)+"/")
        # self.img_right_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        self.img_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        # self.img_left_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        self.learn_no = 4000
        self.count = 0
        self.data = 1689
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro), exist_ok=True)
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/", exist_ok=True)
        
        # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")

    def learn(self):
        ang_list = []

        img_list = []

        for i in range(self.data):
                for j in ["-5", "0", "+5"]:
                    img = cv2.imread(self.img_path + str(i) + "_" + j + ".jpg")
                    # img_left = cv2.imread(self.img_left_path + str(self.num) + "_" + str(j) + ".jpg")
                    # img_right = cv2.imread(self.img_right_path + str(self.num) + "_" + str(j) + ".jpg")

                    img_list.append(img)
                    self.count += 1

        with open(self.ang_path + 'ang.csv', 'r') as f:
        # with open(self.ang_path + 'ang_old.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_ang = row 
                ang_list.append(float(tar_ang))
        
        for k in range(self.count):
            img = img_list[k]
            target_ang = ang_list[k]
            self.dl.make_dataset(img, target_ang)
            # self.dl.make_dataset(img_left, target_ang)
            # self.dl.make_dataset(img_right, target_ang)
            print("dataset:" + str(k))

        for l in range(self.learn_no):
            loss = self.dl.trains(self.count)
            print("train" + str(l))
            with open("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/"+str(self.model_num)+".csv", 'a') as fw:
            # with open("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/01/"+str(self.model_num)+".csv", 'a') as fw:
                writer = csv.writer(fw, lineterminator='\n')
                line = [str(loss)]
                writer.writerow(line)
        self.dl.save(self.save_path)
        sys.exit()

if __name__ == '__main__':
    rg = cource_following_learning_node()
    rg.learn()
         

