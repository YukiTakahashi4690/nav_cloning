#!/usr/bin/env python3
from nav_cloning_pytorch import *
import cv2
import csv
from skimage.transform import resize
import time
import os
import sys
import random 
import numpy as np

class cource_following_learning_node:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        # os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/" + self.start_time)
        self.model_num = str(sys.argv[1])
        self.pro = "694_520_01hz"
        self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model"+str(self.model_num)+".pt")
        # self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/01/model"+str(self.model_num)+".pt")
        self.ang_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/"+str(self.pro)+"/")
        self.ang_vel_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/analysis/00_02/")

        self.img_right_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/right")
        self.img_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        self.img_left_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/left")
        self.learn_no = 4000
        self.count = 0
        self.data = 653
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro), exist_ok=True)
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/", exist_ok=True)

    def learn(self):
        ang_list = []
        ang_left_list = []
        ang_right_list = []

        ang_left_center_list = []
        ang_left_left_list = []
        ang_left_right_list = []

        ang_right_center_list = []
        ang_right_left_list = []
        ang_right_right_list = []

        img_left_left_list = []
        img_left_center_list = []
        img_left_right_list = []

        img_right_list = []
        img_list = []
        img_left_list = []

        img_right_left_list = []
        img_right_center_list = []
        img_right_right_list = []

        for i in range(self.data):
            # if i % 3 == 0:
            #     pass
            # else:
                # for j in ["0"]:
                # for j in ["-5", "0", "+5"]:
                    img_left_left = cv2.imread(self.img_left_path + str(i) + "_" + "+5" + ".jpg")
                    img_left_center = cv2.imread(self.img_left_path + str(i) + "_" + "0" + ".jpg")
                    img_left_right = cv2.imread(self.img_left_path + str(i) + "_" + "-5" + ".jpg")

                    img_left = cv2.imread(self.img_path + str(i) + "_" + "+5" + ".jpg")
                    img = cv2.imread(self.img_path + str(i) + "_" + "0" + ".jpg")
                    img_right = cv2.imread(self.img_path + str(i) + "_" + "-5" + ".jpg")

                    img_right_left = cv2.imread(self.img_right_path + str(i) + "_" + "+5" + ".jpg")
                    img_right_center = cv2.imread(self.img_right_path + str(i) + "_" + "0" + ".jpg")
                    img_right_right = cv2.imread(self.img_right_path + str(i) + "_" + "-5" + ".jpg")
                    
                    img_left_left_list.append(img_left_left)
                    img_left_center_list.append(img_left_center)
                    img_left_right_list.append(img_left_right)

                    img_right_list.append(img_right)
                    img_list.append(img)
                    img_left_list.append(img_left) 

                    img_right_left_list.append(img_right_left)
                    img_right_center_list.append(img_right_center)
                    img_right_right_list.append(img_right_right) 

                    # self.count += 1

        with open(self.ang_path + 'ang.csv', 'r') as f:
        # with open(self.ang_path + '0m_0deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_ang = row
                # if float(no) % 3 == 0:
                #     pass
                # else:    
                ang_list.append(float(tar_ang))

        with open(self.ang_vel_path + '0m_5deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_left_ang = row   
                ang_left_list.append(float(tar_left_ang))

        with open(self.ang_vel_path + '0m_-5deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_right_ang = row   
                ang_right_list.append(float(tar_right_ang))

        with open(self.ang_vel_path + '02m_0deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_left_center_ang = row   
                ang_left_center_list.append(float(tar_left_center_ang))

        with open(self.ang_vel_path + '02m_5deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_left_left_ang = row   
                ang_left_left_list.append(float(tar_left_left_ang))

        with open(self.ang_vel_path + '02m_-5deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_left_right_ang = row   
                ang_left_right_list.append(float(tar_left_right_ang))

        with open(self.ang_vel_path + '-02m_0deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_right_center_ang = row   
                ang_right_center_list.append(float(tar_right_center_ang))

        with open(self.ang_vel_path + '-02m_5deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_right_left_ang = row   
                ang_right_left_list.append(float(tar_right_left_ang))
                self.count += 1

        # with open(self.ang_vel_path + '-02m_-5deg.csv', 'r') as f:
        #     for row in csv.reader(f):
        #         no, tar_right_right_ang = row   
        #         ang_right_right_list.append(float(tar_right_right_ang))
        
        for k in range(self.count):
        # for k in range(self.data):
            img_left_left = img_left_left_list[k]
            img_left_center = img_left_center_list[k]
            img_left_right = img_left_right_list[k]

            img_right = img_right_list[k]
            img = img_list[k]
            img_left = img_left_list[k]

            img_right_left = img_right_left_list[k]
            img_right_center = img_right_center_list[k]
            img_right_right = img_right_right_list[k]

            target_ang_left = ang_left_list[k]
            target_ang = ang_list[k]
            target_ang_right = ang_right_list[k]

            target_ang_left_left = ang_left_left_list[k]
            target_ang_left_center = ang_left_center_list[k]
            target_ang_left_right = ang_left_right_list[k]

            target_ang_right_left = ang_right_left_list[k]
            target_ang_right_center = ang_right_center_list[k]
            target_ang_right_right = ang_right_right_list[k]

            self.dl.make_dataset(img_left_left, target_ang_left_left)
            self.dl.make_dataset(img_left_center, target_ang_left_center)
            self.dl.make_dataset(img_left_right, target_ang_left_right)

            self.dl.make_dataset(img_left, target_ang_left)
            self.dl.make_dataset(img, target_ang)
            self.dl.make_dataset(img_right, target_ang_right)

            self.dl.make_dataset(img_right_left, target_ang_right_left)
            self.dl.make_dataset(img_right_center, target_ang_right_center)
            self.dl.make_dataset(img_right_right,  target_ang_right_right)

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
         

