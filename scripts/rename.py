#!/usr/bin/env python3
import os
import rospy
import glob

class rename_node:
    def __init__(self):
        rospy.init_node('rename_node', anonymous = True)
        self.img_path = glob.glob('/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/img/corner_data_cp/' + 'center' + self.i)
        # self.right_img_path = glob.glob('/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/img/corner_data_cp/' + self.i)
        # self.left_img_path = glob.glob('/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/img/corner_data_cp/' + self.i)
        self.nom = 926
        self.count = self.i + 1530
        
        for before_file_name in self.img_path:
            for j in ['-5', '0', '+5']:
                after_file_name = before_file_name.replace(self.nom, self.count)
                os.rename(self.img_path + 'center' + str(self.i) + '_' + j + '.jpg', self.img_path + str(self.i + self.count) + '_' + j + '.jpg')
                # os.rename(self.right_img_path + 'right' + str(self.i) + '_' + j + '.jpg', self.right_img_path + str(self.i + self.count) + '_' + j + '.jpg')
                # os.rename(self.left_img_path + 'left' + str(self.i) + '_' + j + '.jpg', self.left_img_path + str(self.i + self.count) + '_' + j + '.jpg')

        print('rename complete!!')