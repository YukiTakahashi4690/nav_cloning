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
        self.pro = "omni_0614"
        self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model"+str(self.model_num)+".pt")
        # self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/01/model"+str(self.model_num)+".pt")
        self.ang_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/"+str(self.pro)+"/")
        self.img_right_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/right")
        self.img_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        self.img_left_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/left")
        self.learn_no = 4000
        self.count = 0
        self.data = 432
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro), exist_ok=True)
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/", exist_ok=True)
        
        # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")

    def learn(self):
        ang_list = []

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

                    self.count += 1

        with open(self.ang_path + 'ang.csv', 'r') as f:
        # with open(self.ang_path + '0m_0deg.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_ang = row
                # if float(no) % 3 == 0:
                #     pass
                # else:    
                ang_list.append(float(tar_ang))
        
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

            target_ang = ang_list[k]

            # img_left_left = resize(img_left_left, (48, 64), mode='constant')
            # img_left_center = resize(img_left_center, (48, 64), mode='constant')
            # img_left_right = resize(img_left_right, (48, 64), mode='constant')
            # # r, g, b = cv2.split(img_left)
            # # imgobj_left = np.asanyarray([r, g, b])

            # img_left = resize(img_left, (48, 64), mode='constant')
            # img = resize(img, (48, 64), mode='constant')
            # img_right = resize(img_right, (48, 64), mode='constant')
            # # r, g, b = cv2.split(img)
            # # imgobj = np.asanyarray([r, g, b])

            # img_right_left = resize(img_right_left, (48, 64), mode='constant')
            # img_right_center = resize(img_right_center, (48, 64), mode='constant')
            # img_right_right = resize(img_right_right, (48, 64), mode='constant')
            # r, g, b = cv2.split(img_right)
            # imgobj_right = np.asanyarray([r, g, b])

            """
            self.dl.make_dataset(imgobj_right, target_ang + 0.25)
            self.dl.make_dataset(imgobj, target_ang)
            self.dl.make_dataset(imgobj_left, target_ang - 0.2)
            """

            self.dl.make_dataset(img_left_left, target_ang -0.244)
            self.dl.make_dataset(img_left_center, target_ang -0.182)
            self.dl.make_dataset(img_left_right, target_ang -0.057)

            self.dl.make_dataset(img_left, target_ang -0.0128)
            self.dl.make_dataset(img, target_ang)
            self.dl.make_dataset(img_right, target_ang + 0.134)

            self.dl.make_dataset(img_right_left, target_ang + 0.196)
            self.dl.make_dataset(img_right_center, target_ang + 0.245)
            self.dl.make_dataset(img_right_right, target_ang + 0.26)

            print("dataset:" + str(k))

            # self.dl.make_dataset(img_left_left, target_ang -0.25)
            # self.dl.make_dataset(img_left_center, target_ang -0.25)
            # self.dl.make_dataset(img_left_right, target_ang -0.25)

            # self.dl.make_dataset(img_left, target_ang -0.002 - random.uniform(-0.001, 0.001))
            # self.dl.make_dataset(img, target_ang)
            # self.dl.make_dataset(img_right, target_ang + 0.002 + random.uniform(-0.001, 0.001))

            # self.dl.make_dataset(img_right_left, target_ang + 0.2)
            # self.dl.make_dataset(img_right_center, target_ang + 0.2)
            # self.dl.make_dataset(img_right_right, target_ang + 0.2)
            # print("dataset:" + str(k))

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
         

