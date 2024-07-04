#!/usr/bin/env python3
from __future__ import print_function
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
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
#from gazebo_msgs.srv import SetModelState
#from gazebo_msgs.srv import GetModelState
#from gazebo_msgs.msg import ModelState
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
from nav_msgs.msg import Odometry
import math

class cource_following_learning_node:
    def __init__(self):
        rospy.init_node('cource_following_learning_node', anonymous=True)
        self.action_num = rospy.get_param("/LiDAR_based_learning_node/action_num", 1)
        print("action_num: " + str(self.action_num))
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_sub = rospy.Subscriber("/camera_center_left/rgb/image_raw", Image, self.callback_center_left_camera)
        self.image_sub = rospy.Subscriber("/camera_center_right/rgb/image_raw", Image, self.callback_center_right_camera)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_left_left_sub = rospy.Subscriber("/camera_left_left/rgb/image_raw", Image, self.callback_left_left_camera)
        self.image_left_right_sub = rospy.Subscriber("/camera_left_right/rgb/image_raw", Image, self.callback_left_right_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.image_right_left_sub = rospy.Subscriber("/camera_right_left/rgb/image_raw", Image, self.callback_right_left_camera)
        self.image_right_right_sub = rospy.Subscriber("/camera_right_right/rgb/image_raw", Image, self.callback_right_right_camera)

        self.vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.callback_vel, queue_size=10)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        # self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()

        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_center_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_center_right_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_right_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_right_image = np.zeros((480,640,3), np.uint8)

        # self.cv_image = np.zeros((480,689,3), np.uint8)
        # self.cv_left_image = np.zeros((480,689,3), np.uint8)
        # self.cv_right_image = np.zeros((480,689,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model/'
        self.previous_reset_time = 0
        self.start_time_s = rospy.get_time()
        self.save_img_no = 0
        self.save_img_left_no = 1
        self.save_img_center_no = 0
        self.save_img_right_no = -1
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/'
        self.pose_x = 0
        self.pose_y = 0
        self.old_pose_x = 0
        self.old_pose_y = 0
        self.current_pose_x = 0
        self.current_pose_y = 0
        self.flag = False
        os.makedirs(self.path + "img/" + self.start_time)
        os.makedirs(self.path + "ang/" + self.start_time)
        self.pose_sub = rospy.Subscriber("/tracker", Odometry, self.callback_odom_pose)

    def capture_img(self):
            Flag = True
            try:
                # cv2.imwrite(self.path + "img/" + self.start_time + "/left" + str(self.save_img_no) + "_" + "+5" + ".jpg", self.resize_left_left_img)
                # cv2.imwrite(self.path + "img/" + self.start_time + "/left" + str(self.save_img_no) + "_" + "0" + ".jpg", self.resize_left_center_img)
                # cv2.imwrite(self.path + "img/" + self.start_time + "/left" + str(self.save_img_no) + "_" + "-5" + ".jpg", self.resize_left_right_img)

                # cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_no) + "_" + "+5" + ".jpg", self.resize_left_img)
                # cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_no) + "_" + "0" + ".jpg", self.resize_img)
                # cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_no) + "_" + "-5" + ".jpg", self.resize_right_img)

                # cv2.imwrite(self.path + "img/" + self.start_time + "/right" + str(self.save_img_no) + "_" + "+5" + ".jpg", self.resize_right_left_img)
                # cv2.imwrite(self.path + "img/" + self.start_time + "/right" + str(self.save_img_no) + "_" + "0" + ".jpg", self.resize_right_center_img)
                # cv2.imwrite(self.path + "img/" + self.start_time + "/right" + str(self.save_img_no) + "_" + "-5" + ".jpg", self.resize_right_right_img)

                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_left_no) + "_" + "+5" + ".jpg", self.resize_left_left_img)
                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_left_no) + "_" + "0" + ".jpg", self.resize_left_center_img)
                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_left_no) + "_" + "-5" + ".jpg", self.resize_left_right_img)

                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_center_no) + "_" + "+5" + ".jpg", self.resize_left_img)
                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_center_no) + "_" + "0" + ".jpg", self.resize_img)
                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_center_no) + "_" + "-5" + ".jpg", self.resize_right_img)

                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_right_no) + "_" + "+5" + ".jpg", self.resize_right_left_img)
                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_right_no) + "_" + "0" + ".jpg", self.resize_right_center_img)
                cv2.imwrite(self.path + "img/" + self.start_time + "/center" + str(self.save_img_right_no) + "_" + "-5" + ".jpg", self.resize_right_right_img)
  
            except:
                print('Not save image')
                Flag = False
            finally:
                if Flag:
                    print('Save image Number:', self.save_img_no)

    def capture_ang(self):
            line = [str(self.save_img_no), str(self.action)]
            with open(self.path + "ang/" + self.start_time + '/ang.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_center_left_camera(self, data):
        try:
            self.cv_center_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_center_right_camera(self, data):
        try:
            self.cv_center_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_left_camera(self, data):
        try:
            self.cv_left_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_right_camera(self, data):
        try:
            self.cv_left_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_left_camera(self, data):
        try:
            self.cv_right_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_right_camera(self, data):
        try:
            self.cv_right_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_path(self, data):
        self.path_pose = data

    def callback_odom_pose(self, data):
        self.pose_x = data.pose.pose.position.x
        self.pose_y = data.pose.pose.position.y

    def check_distance(self):
        distance = math.sqrt((self.pose_x - self.old_pose_x)**2 + (self.pose_y - self.old_pose_y)**2)
        print("distance : ", distance)
        if distance >= 0.1:
            self.old_pose_x = self.pose_x
            self.old_pose_y = self.pose_y
            self.flag = True

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

    def loop(self):
        self.check_distance()
        if self.flag:

            self.resize_left_left_img = cv2.resize(self.cv_left_left_image, dsize=(64, 48))
            self.resize_left_center_img = cv2.resize(self.cv_left_image, dsize=(64, 48))
            self.resize_left_right_img = cv2.resize(self.cv_left_right_image, dsize=(64, 48))

            self.resize_left_img = cv2.resize(self.cv_center_left_image, dsize=(64, 48))
            self.resize_img = cv2.resize(self.cv_image, dsize=(64, 48))
            self.resize_right_img = cv2.resize(self.cv_center_right_image, dsize=(64, 48))

            self.resize_right_left_img = cv2.resize(self.cv_right_left_image, dsize=(64, 48))
            self.resize_right_center_img = cv2.resize(self.cv_right_image, dsize=(64, 48))
            self.resize_right_right_img = cv2.resize(self.cv_right_right_image, dsize=(64, 48))

            self.capture_img()
            self.capture_ang()
            self.save_img_no += 1
            self.save_img_left_no += 3
            self.save_img_center_no += 3
            self.save_img_right_no += 3
            self.flag = False
        
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        
        if self.vel.linear.x == 0:
            return

        if self.episode == 4000:
            self.learning = False

if __name__ == '__main__':
    rg = cource_following_learning_node()
    DURATION = 0.1
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()