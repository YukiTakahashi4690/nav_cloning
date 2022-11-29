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
        self.action_num = rospy.get_param("/LiDAR_based_learning_node/action_num", 1)
        print("action_num: " + str(self.action_num))
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        # self.path_pub = rospy.Publisher("/path", Path, queue_size=10)
        # self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        # self.waypoint_sub = rospy.Subscriber("/count_waypoint", Int8, self.callback_waypoint)
        # self.current_waypoint = rospy.Subscriber("/current_waypoints", Int8, self.callback_waypoint)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        # self.target_path = PoseStamped()
        self.target_path = Path()
        # target_pose = PoseStamped()
        self.path_pose = PoseArray()
        self.path_pose_x = 0
        self.path_pose_y = 0
        self.old_path_x = self.path_pose_x
        self.old_path_y = self.path_pose_y
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.init = True
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model/'
        self.previous_reset_time = 0
        self.start_time_s = rospy.get_time()
        self.collect_data_srv = rospy.Service('/collect_data', Trigger, self.collect_data)
        self.save_img_no = 0
        self.csv_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
        self.pos_list = []
        self.path_list = []
        self.pos = PoseWithCovarianceStamped()
        self.orientation = 0
        self.r = rospy.Rate(10)
        self.capture_rate = rospy.Rate(0.3)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.state = ModelState()
        self.state.model_name = 'mobile_base'
        self.amcl_pose_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=100)
        self.diff = 0
        self.write_flag = True
        self.old_waypoint = 0
        self.count = 0
        os.makedirs(self.path + self.start_time)
        os.makedirs(self.path + "analysis/img/" + self.start_time)
        os.makedirs(self.path + "analysis/ang/" + self.start_time)

        with open(self.path + self.start_time + '/' +  'reward.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)'])

        # with open(self.path + 'analysis/path/path_comp5.csv', 'r') as f:
        #     for row in f:
        #         target_pose = PoseStamped()
        #         self.path_list.append(row)
        #         x, y = (','.join(self.path_list[self.count].splitlines())).split(',')
        #         target_pose.pose.position.x = float(x)
        #         target_pose.pose.position.y = float(y)
        #         self.target_path.poses.append(target_pose)
        #         self.count += 1
        #     # print(self.target_path)
        # self.path_pub.publish(self.target_path)

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

    # def callback_path(self, data):
    #     if self.write_flag:
    #         a = len(pos.pose)
    #         with open(self.path + 'analysis/path/path_comp6.csv', 'a') as f:
    #             for i in range(a):
    #                 self.path_pose_x = pos.pose.pose.position.x
    #                 # self.path_pose_y = target_pose.position.y
    #                 # self.path_pose_orientation_x = target_pose.orientation.x
    #                 # self.path_pose_orientation_y = target_pose.orientation.y
    #                 # path_line = [str(self.path_pose_x), str(self.path_pose_y), str(self.path_pose_orientation_x), str(self.path_pose_orientation_y)]
    #                 path_line = [str(self.path_pose_x)]
    #                 writer = csv.writer(f, lineterminator='\n')
    #                 writer.writerow(path_line)
    #                 print("x: ", self.path_pose_x)
    #     self.write_flag = False

    # def callback_path(self, data):
    #     if self.write_flag:
    #         a = len(data.poses)
    #         with open(self.path + 'analysis/path/path_comp5.csv', 'a') as f:
    #             for i in range(a):
    #                 self.path_pose_x = data.poses[i].pose.position.x
    #                 self.path_pose_y = data.poses[i].pose.position.y
    #                 path_line = [str(self.path_pose_x), str(self.path_pose_y)]
    #                 writer = csv.writer(f, lineterminator='\n')
    #                 writer.writerow(path_line)
    #     self.write_flag = False

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos1 = data.pose.pose.position

        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos1.x - path.x)**2 + (pos1.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    # def callback_waypoint(self, data):
    #     self.waypoint = data.data
    #     if self.waypoint - self.old_waypoint == 1 or self.waypoint == 0:
    #         self.write_flag = True
    #     else:
    #         self.write_flag = False
    #     self.old_waypoint = self.waypoint
        
        # if self.waypoint == self.diff:
        #     pass
        # else:
        #     self.write_flag = True          
        # self.diff = self.waypoint   

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

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



    def collect_data(self, data):
        rospy.wait_for_service('/collect_data')
        service = rospy.ServiceProxy('/collect_data', Trigger)

        for i in range(903):
            self.capture_img()
            self.capture_ang()
            x, y, theta = self.read_csv()
            self.robot_moving(x, y, theta)

            self.save_img_no += 1
            self.capture_rate.sleep()

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        """
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        try:
            previous_model_state = get_model_state('mobile_base', 'world')
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        """
        img = resize(self.cv_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img)
        imgobj = np.asanyarray([r,g,b])

        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_left)
        imgobj_left = np.asanyarray([r,g,b])

        img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_right)
        imgobj_right = np.asanyarray([r,g,b])

        ros_time = str(rospy.Time.now())


        # if self.episode == 1000:
        #     self.learning = False
            #self.dl.save(self.save_path)
            #self.dl.load(self.load_path)
            # sys.exit()

        if self.learning:
            target_action = self.action
            distance = self.min_distance

            # manual
            # if distance > 0.1:
            #     self.select_dl = False
            # elif distance < 0.05:
            #     self.select_dl = True
            # if self.select_dl and self.episode >= 0:
            #     target_action = 0
            # action, loss = self.dl.act_and_trains(imgobj, target_action)
            # if abs(target_action) < 0.1:
            #     action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
            #     action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
            # angle_error = abs(action - target_action)

            # zigzag
            # action, loss = self.dl.act_and_trains(imgobj, target_action)
            # if abs(target_action) < 0.1:
            #     action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
            #     action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
            #     angle_error = abs(action - target_action)
            # if distance > 0.1:
            #     self.select_dl = False
            # elif distance < 0.05:
            #     self.select_dl = True
            # if self.select_dl and self.episode >= 0:
            #     target_action = 0

            # use_dl_output
            # action, loss = self.dl.act_and_trains(imgobj, target_action)
            # if abs(target_action) < 0.1:
            #     action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
            #     action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
            # angle_error = abs(action - target_action)
            # if distance > 0.1:
            #     self.select_dl = False
            # elif distance < 0.05:
            #     self.select_dl = True
            # if self.select_dl and self.episode >= 0:
            #     target_action = action
            
            # follow line method
            action, loss = self.dl.act_and_trains(imgobj, target_action)
            if abs(target_action) < 0.1:
                action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
            angle_error = abs(action - target_action)

            # selected_training
            # action = self.dl.act(imgobj)
            #     angle_error = abs(action - target_action)
            #     loss = 0
            #     if angle_error > 0.05:
            #         action, loss = self.dl.act_and_trains(imgobj, target_action)
            #         if abs(target_action) < 0.1:
            #             action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
            #             action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
            #     if distance > 0.1:
            #         self.select_dl = False
            #     elif distance < 0.05:
            #         self.select_dl = True
            #     if self.select_dl and self.episode >= 0:
            #         target_action = action

          
            # end method

            print(" episode: " + str(self.episode) + ", loss: " + str(loss) + ", angle_error: " + str(angle_error) + ", distance: " + str(distance))
            self.episode += 1
            line = [str(self.episode), "training", str(loss), str(angle_error), str(distance)]
            with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.0
            self.vel.angular.z = target_action
            self.vel.angular.z = 0
            self.nav_pub.publish(self.vel)

        else:
            target_action = self.dl.act(imgobj)
            distance = self.min_distance
            print("TEST MODE: " + " angular:" + str(target_action) + ", distance: " + str(distance))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            line = [str(self.episode), "test", "0", str(angle_error), str(distance)]
            with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = cource_following_learning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()