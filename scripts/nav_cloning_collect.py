#!/usr/bin/env python3
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from geometry_msgs.msg import PoseWithCovarianceStamped,Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_net import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState

import math
import tf

from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys

class cource_following_learning_node:
    def __init__(self):
        rospy.init_node('cource_following_learning_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.min_distance = 0.0
        self.action = 0.0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.init = True
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result/'
        self.collect_data_srv = rospy.Service('/collect_data', Trigger, self.collect_data)
        self.goal_pub_srv = rospy.Service('/goal_pub', Trigger, self.goal_pub)
        self.save_img_no = 0
        self.save_img_no1= 0
        self.goal_img_no2 = 0
        self.csv_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
        self.pos_list = []
        self.goal_list1 = []
        self.goal_list2 = []
        self.pos = PoseWithCovarianceStamped()
        self.g_pos = PoseStamped()
        self.orientation = 0
        self.r = rospy.Rate(10)
        self.capture_rate = rospy.Rate(0.3)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.state = ModelState()
        self.state.model_name = 'mobile_base'
        self.amcl_pose_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=100)
        self.count = 0
        # self.simple_goal_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.callback_simple_goal)
        self.simple_goal_pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)
        os.makedirs(self.path + self.start_time)
        os.makedirs(self.path + "analysis/img/" + self.start_time)
        os.makedirs(self.path + "analysis/ang/" + self.start_time)

    def capture_img(self):
            Flag = True
            try:
                cv2.imwrite(self.path + "analysis/img/" + self.start_time + "/center" + str(self.save_img_no) + ".jpg", self.cv_image)
                cv2.imwrite(self.path + "analysis/img/" + self.start_time + "/right" + str(self.save_img_no) + ".jpg", self.cv_right_image)
                cv2.imwrite(self.path + "analysis/img/" + self.start_time + "/left" + str(self.save_img_no) + ".jpg", self.cv_left_image)
            except:
                print('Not save image')
                Flag = False
            finally:
                if Flag:
                    print('Save image Number:', self.save_img_no)

    def capture_ang(self):
            line = [str(self.save_img_no), str(self.action)]
            with open(self.path + "analysis/ang/" + self.start_time + '/ang.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)

    def read_csv(self):
            if self.init:
                f = open(self.csv_path + 'traceable_pos.csv', 'r')
                for row in f:
                    self.pos_list.append(row)
                self.init = False
            cur_pos = self.pos_list[self.save_img_no]
            pos = cur_pos.split(',')
            x = float(pos[1])
            y = float(pos[2])
            theta = float(pos[3])
            print('Moving_pose:', x, y, theta)
            return x, y, theta

    def simple_goal(self):
            # ft = open(self.csv_path + 'traceable_pos.csv', 'r')
            # for row in ft:
            #     self.goal_list1.append(row)
            # goal_pos1 = self.goal_list1[self.save_img_no]
            # self.goal_img_no1 += 1

            fs = open(self.csv_path + 'simple_goal_org.csv', 'r')
            for row in fs:
                self.goal_list2.append(row)
            goal_pos2 = self.goal_list2[self.save_img_no1]
            simple_pos = goal_pos2.split(',')
            x = float(simple_pos[1])
            y = float(simple_pos[2])

            self.g_pos.header.stamp = rospy.Time.now()

            self.g_pos.header.frame_id = 'map'
            self.g_pos.pose.position.x = x - 9.821
            self.g_pos.pose.position.y = y - 16.1
            self.g_pos.pose.position.z = 0

            self.g_pos.pose.orientation.x = 0 
            self.g_pos.pose.orientation.y = 0
            self.g_pos.pose.orientation.z = 0
            self.g_pos.pose.orientation.w = 1.0

            self.simple_goal_pub.publish(self.g_pos)

            # self.goal_img_no2 += 1
            
            # if self.goal_img_no2 == 1:
            #     # print('curent_position:', x, y)
            #     self.simple_goal_pub.publish(self.g_pos)
            #     self.goal_img_no2 = 0

    def robot_moving(self, x, y, angle):
            #amcl
            #replace_pose = PoseWithCovarianceStamped()

            self.pos.header.frame_id = 'odom'
            self.pos.pose.pose.position.x = x - 9.821
            self.pos.pose.pose.position.y = y - 16.1

            self.pos.pose.pose.orientation.x = 0 
            self.pos.pose.pose.orientation.y = 0
            self.pos.pose.pose.orientation.z = 0
            self.pos.pose.pose.orientation.w = 0 
            self.pos.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
            
            self.amcl_pose_pub.publish(self.pos)
            #gazebo
            for offset_ang in [-5, 0, 5]:
                the = angle + math.radians(offset_ang)
                the = the - 2.0 * math.pi if the >  math.pi else the
                the = the + 2.0 * math.pi if the < -math.pi else the
                self.state.pose.position.x = x
                self.state.pose.position.y = y
                quaternion = tf.transformations.quaternion_from_euler(0, 0, the)
                self.state.pose.orientation.x = quaternion[0]
                self.state.pose.orientation.y = quaternion[1]
                self.state.pose.orientation.z = quaternion[2]
                self.state.pose.orientation.w = quaternion[3]
                try:
                    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                    resp = set_state( self.state )
                    #test
                    self.capture_img()
                    self.capture_ang()
                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)
                self.r.sleep()
                self.r.sleep()
                self.r.sleep()

    def goal_pub(self, data):
        rospy.wait_for_service('/goal_pub')
        service = rospy.ServiceProxy('/goal_pub', Trigger)

        # for i in range(903):
        self.simple_goal()
        self.save_img_no1 += 1
    
    def collect_data(self, data):
        rospy.wait_for_service('/collect_data')
        service = rospy.ServiceProxy('/collect_data', Trigger)

        for i in range(903):
            self.capture_img()
            self.capture_ang()
            x, y, theta = self.read_csv()
            self.robot_moving(x, y, theta)
            print("current_position:", x, y, theta)

            self.save_img_no += 1
            self.capture_rate.sleep()

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

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_pose(self, data):
        distance_list = []
        pos1 = data.pose.pose.position

        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos1.x - path.x)**2 + (pos1.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)

if __name__ == '__main__':
    rg = cource_following_learning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        # rg.loop()
        r.sleep() 