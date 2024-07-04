#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.num = rospy.get_param("/nav_cloning_node/num", "1")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image/mercator", Image, self.callback)
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        # self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        # self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.waypoint_num = rospy.Subscriber("/count_waypoint", Int8, self.callback_waypoint)
        self.gazebo_pos_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_gazebo_pos, queue_size = 2)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((520,694,3), np.uint8)
        # self.cv_left_image = np.zeros((520,694,3), np.uint8)
        # self.cv_right_image = np.zeros((520,694,3), np.uint8)
        # self.learning = True
        self.learning = False
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.pro = "omni_0614"
        # self.load_path = "/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model1.pt"
        self.load_path = "/home/y-takahashi/catkin_ws/src/nav_cloning/data/model/"+str(self.pro)+"/model"+str(self.num)+".pt"
        self.score = "/home/y-takahashi/catkin_ws/src/nav_cloning/data/score/"+str(self.pro)+".csv"
        if self.learning == False:
            print(self.load_path)
            self.dl.load(self.load_path)
        # self.model_num = 1
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.old_wp = 0
        self.lap = 0
        self.collision_list = [[],[]]
        self.is_started = False
        self.start_time_s = rospy.get_time()
        # os.makedirs(self.path, exist_ok=True)
        # os.makedirs(self.save_path, exist_ok=True)
        # os.makedirs(self.path + self.start_time)
        # os.makedirs(roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_'+str(self.mode)+'/'+str(self.start_time))

        # with open(self.path + 'training.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)', 'direction'])
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)


    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    # def callback_dl_training(self, data):
    #     resp = SetBoolResponse()
    #     self.learning = data.data
    #     resp.message = "Training: " + str(self.learning)
    #     resp.success = True
    #     return resp

    def callback_waypoint(self, data):
        self.current_wp = data.data
    
    def callback_gazebo_pos(self, data):
        self.gazebo_pos_x = data.pose[2].position.x
        self.gazebo_pos_y = data.pose[2].position.y
    
    def collision(self):
        collision_flag = False
        self.collision_list[0].append(self.gazebo_pos_x)
        self.collision_list[1].append(self.gazebo_pos_y)
        if len(self.collision_list[0]) == 10:
            average_x = sum(self.collision_list[0]) / len(self.collision_list[0])
            average_y = sum(self.collision_list[1]) / len(self.collision_list[1])
            distance = np.sqrt(abs((self.gazebo_pos_x - average_x)**2 + (self.gazebo_pos_y - average_y)**2))
            self.collision_list[0] = self.collision_list[0][1:]
            self.collision_list[1] = self.collision_list[1][1:]

            if distance < 0.1:
                collision_flag = True
                print("collision")

        return collision_flag

    def loop(self):
        if self.cv_image.size != 1280 * 640 * 3:
        # if self.cv_image.size != 694 * 520 * 3:
            return
        # if self.cv_left_image.size != 640 * 480 * 3:
        #     return
        # if self.cv_right_image.size != 640 * 480 * 3:
        #     return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return
        
        crop_img = self.cv_image[80:560, 320:960]

        img = resize(crop_img, (48, 64), mode='constant')
        # img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        # img_right = resize(self.cv_right_image, (48, 64), mode='constant')

        ros_time = str(rospy.Time.now())

        # if self.episode == 0:
        #     self.learning = False
        #     self.dl.save(self.save_path)
        #     self.dl.load(self.load_path)
        #     os.system('killall roslaunch')
        #     sys.exit()

        if self.old_wp == 0 and self.current_wp == 1:
            self.lap += 1
            # if self.lap > 1 and self.old_wp == 0 and self.current_wp == 1 and self.learning == True:
            #     os.system('rosnode kill /my_bag')
            #     self.dl.save(self.save_path + 'model' + str(self.model_num) + '.pt')
            #     self.model_num += 1
            #     if self.model_num > 4:
            #         os.system('killall roslaunch')
            #         sys.exit()
        self.old_wp = self.current_wp

        if self.lap == 2:
            line = ['model_'+str(self.num), 'True', str(self.lap)]
            with open(self.score, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            os.system('killall roslaunch')
            sys.exit()
        
        if self.episode > 5:
            collision_flag = self.collision()
            if collision_flag:
                line = ['model_'+str(self.num), 'False', str(self.lap), str(self.pos_x), str(self.pos_y)]
                with open(self.score, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(line)
                os.system('killall roslaunch')
                sys.exit()

        # if self.episode == 8000 and self.learning == True:
        #     self.dl.save(self.save_path + 'model' + str(self.episode) + '.pt')
        #     os.system('killall roslaunch')
        #     sys.exit()

        if self.learning == False:
            target_action = self.dl.act(img)
            distance = self.min_distance
            # print(str(self.episode) + ", test, angular:" + str(target_action) + ", distance: " + str(distance))
            print(str(self.episode) + ", test, angular:" + str(target_action))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            line = [str(self.episode), "test", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)]
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        # temp = copy.deepcopy(img_left)
        # cv2.imshow("Resized Left Image", temp)
        # temp = copy.deepcopy(img_right)
        # cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()