#!/usr/bin/env python3
from nav_cloning_pytorch import *
import cv2
import csv
from skimage.transform import resize
import time
import os
import sys

class cource_following_learning_node:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        # os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/" + self.start_time)
        self.model_num = str(sys.argv[1])
        self.pro = "00_01"
        self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model"+str(self.model_num)+".pt")
        # self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/00_4000/model"+str(self.model_num)+".pt")
        self.ang_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/"+str(self.pro)+"/")
        self.img_right_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/right")
        self.img_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        self.img_left_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/left")
        self.learn_no = 4000
        self.pos_no = 0
        self.data = 1677
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro), exist_ok=True)
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/", exist_ok=True)
        
        # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")

    def learn(self):
        ang_list = []
        ang_list2 = []
        img_right_list = []
        img_list = []
        img_left_list = []

        with open(self.ang_path + 'ang.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_ang = row
                ang_list.append(float(tar_ang))

        for i in range(self.data):
            for a in ["-5"]:
                img_right = cv2.imread(self.img_right_path + str(i) + "_" + a + ".jpg")
                img = cv2.imread(self.img_path + str(i) + "_" + a + ".jpg")
                img_left = cv2.imread(self.img_left_path + str(i) + "_" + a + ".jpg")
                img_right_list.append(img_right)
                img_list.append(img)
                img_left_list.append(img_left) 

                for b in img_right_list:
                    if b % 3 != 2:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for c in img_list:
                    if c % 3 != 2:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for d in img_left_list:
                    if d % 3 != 2:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for e in ang_list:
                    if e % 9 == 6:
                        ang_list2.append(ang_list)

            for a in ["0"]:
                img_right = cv2.imread(self.img_right_path + str(i) + "_" + a + ".jpg")
                img = cv2.imread(self.img_path + str(i) + "_" + a + ".jpg")
                img_left = cv2.imread(self.img_left_path + str(i) + "_" + a + ".jpg")
                img_right_list.append(img_right)
                img_list.append(img)
                img_left_list.append(img_left) 

                for b in img_right_list:
                    if b % 3 != 0:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for c in img_list:
                    if c % 3 != 0:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for d in img_left_list:
                    if d % 3 != 0:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for e in ang_list:
                    if e % 9 == 1:
                        ang_list2.append(ang_list)

            for a in ["+5"]:
                img_right = cv2.imread(self.img_right_path + str(i) + "_" + a + ".jpg")
                img = cv2.imread(self.img_path + str(i) + "_" + a + ".jpg")
                img_left = cv2.imread(self.img_left_path + str(i) + "_" + a + ".jpg")
                img_right_list.append(img_right)
                img_list.append(img)
                img_left_list.append(img_left) 

                for b in img_right_list:
                    if b % 3 != 1:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for c in img_list:
                    if c % 3 != 1:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for d in img_left_list:
                    if d % 3 != 1:
                        img_right_list.remove(b)
                        img_list.remove(b)
                        img_left_list.remove(b)

                for e in ang_list:
                    if e % 9 == 5:
                        ang_list2.append(ang_list)
        
        for k in range(self.data):
            img_right = img_right_list[k]
            img = img_list[k]
            img_left = img_left_list[k]
            target_ang = ang_list2[k]

            img_right = resize(img_right, (48, 64), mode='constant')
            r, g, b = cv2.split(img_right)

            img = resize(img, (48, 64), mode='constant')
            r, g, b = cv2.split(img)

            img_left = resize(img_left, (48, 64), mode='constant')
            r, g, b = cv2.split(img_left)

            self.dl.make_dataset(img_right, target_ang + 0.2)
            self.dl.make_dataset(img, target_ang)
            self.dl.make_dataset(img_left, target_ang - 0.2)
            print("dataset:" + str(k))
            
        for l in range(self.learn_no):
            loss = self.dl.trains()
            print("train" + str(l))
            with open("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/"+str(self.model_num)+".csv", 'a') as fw:
            # with open("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/00_4000/"+str(self.model_num)+".csv", 'a') as fw:
                writer = csv.writer(fw, lineterminator='\n')
                line = [str(loss)]
                writer.writerow(line)
        self.dl.save(self.save_path)
        sys.exit()

if __name__ == '__main__':
    rg = cource_following_learning_node()
    rg.learn()
         

