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
        self.pro = "test2"
        self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model"+str(self.model_num)+".pt")
        # self.save_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/01/model"+str(self.model_num)+".pt")
        self.ang_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/"+str(self.pro)+"/")
        # self.img_right_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/right")
        self.img_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        # self.img_left_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/img/"+str(self.pro)+"/left")
        self.learn_no = 4000
        self.pos_no = 0
        self.count = 0
        self.data = 1688
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro), exist_ok=True)
        os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/", exist_ok=True)
        
        # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")

    def learn(self):
        ang_list = []
        # img_right_list = []
        img_list = []
        # img_left_list = []

        for i in range(self.data):
            # if i % 3 == 0:
            #     pass
            # else:
                # for j in ["0"]:
                for j in ["-5", "0", "+5"]:
                    # img_right = cv2.imread(self.img_right_path + str(i) + "_" + j + ".jpg")
                    img = cv2.imread(self.img_path + str(i) + "_" + j + ".jpg")
                    # img_left = cv2.imread(self.img_left_path + str(i) + "_" + j + ".jpg")
                    # img_right_list.append(img_right)
                    img_list.append(img)
                    # img_left_list.append(img_left) 

                    self.count += 1

        with open(self.ang_path + 'ang.csv', 'r') as f:
        # with open(self.ang_path + 'mazemaze.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_ang = row
                # if float(no) % 3 == 0:
                #     pass
                # else:    
                ang_list.append(float(tar_ang))
        
        for k in range(self.count):
        # for k in range(self.data):
            # img_right = img_right_list[k]
            img = img_list[k]
            # img_left = img_left_list[k]
            target_ang = ang_list[k]

            # img_right = resize(img_right, (48, 64), mode='constant')
            # r, g, b = cv2.split(img_right)
            # imgobj_right = np.asanyarray([r, g, b])

            img = resize(img, (48, 64), mode='constant')
            r, g, b = cv2.split(img)
            imgobj = np.asanyarray([r, g, b])

            # img_left = resize(img_left, (48, 64), mode='constant')
            # r, g, b = cv2.split(img_left)
            # imgobj_left = np.asanyarray([r, g, b])
            
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

            # for n in range(3):
            self.dl.make_dataset(img, target_ang)
            print("dataset:" + str(k))
                
            # self.dl.make_dataset(img_right, target_ang + 0.2)
            
            # self.dl.make_dataset(img_left, target_ang - 0.2)
            
        # joblib.dump((self.dataset_right, self.dataset_center, self.dataset_left), open('/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/dataset/dataset.pkl', 'wb'), compress=6)

        # self.dataset_right, self.dataset_center, self.dataset_left =joblib.load(open('/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/dataset/dataset.pkl', 'rb'))
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
         
