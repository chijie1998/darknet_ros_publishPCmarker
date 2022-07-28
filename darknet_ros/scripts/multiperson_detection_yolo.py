#!/usr/bin/env python3

from typing import Dict, List
import numpy as np
import os
import math
import time
import cv2
import struct
from numpy.core.numeric import NaN
#import open3d as o3d
import matplotlib.pyplot as plt
import glob
import ctypes
import warnings


#ROS Required
import rospy
import ros_numpy
from rospy.core import is_shutdown
from std_msgs.msg import Header,String
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, Image,CompressedImage
from std_msgs.msg import Bool,Float32MultiArray,Float32
from visualization_msgs.msg import Marker,MarkerArray
#from tomato_detection.srv import SelectTomato,SelectTomatoResponse
from geometry_msgs.msg import Point,Pose,PoseStamped,PoseArray,PointStamped
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from actionlib_msgs.msg import GoalID

#Yolor & Tensorrt
# import torch
import json


class PersonDetecor(object):
    def __init__(self):
        self.CAMINFO = {'topic': '/rgb/camera_info', 'msg': CameraInfo}
        self.COLOR = {'topic': '/rgb/image_raw', 'msg': Image}
        self.DEPTH = {'topic': '/depth_to_rgb/image_raw', 'msg': Image}
        self.isCamInfo = False
        self.PC = {'topic': '/hand_tomatoPC', 'msg': PointCloud2}
        
        self.H = 720
        self.W = 1280
        self.header = Header() #Use for point cloud publisher

        self.color_image = np.empty((self.H, self.W ,3), dtype=np.uint8)
        self.depth_image = np.empty((self.H, self.W), dtype=np.uint16)
        self.aligned_image  = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask           = np.empty((self.H, self.W), dtype=np.bool)
        self.mask_image     = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask_depth     = np.empty((self.H, self.W), dtype=np.uint8)
        self.depth_scale = 1000

        self.bboxes:List =[]
        self.markerColors = np.random.uniform(0,1,[80,3])

        self.camera_matrix = np.array([[0.0, 0, 0.0], [0, 0.0, 0.0], [0, 0, 1]], dtype=np.float32) 
        self.camera_distortion = np.array([0,0,0,0,0], dtype=np.float32)

        self.x_lim = 2.0
        self.z_lim = 3.0

        self.goal =[0,0,0]

    def camInfoCallback(self, msg):
        self.header = msg.header
        self.K = msg.K
        self.width = msg.width  
        self.height = msg.height
        self.ppx = msg.K[2]
        self.ppy = msg.K[5]
        self.fx = msg.K[0]
        self.fy = msg.K[4] 
        
        self.k1 = msg.D[0]
        self.k2 = msg.D[1]
        self.t1 = msg.D[2]
        self.t2 = msg.D[3]
        self.k3 = msg.D[4]
        self.isCamInfo = True
        self.camera_matrix = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float32) 
        self.camera_distortion = np.array([self.k1,self.k2,self.t1,self.t2,self.k3], dtype=np.float32)

    def colorCallback(self, msg):
        self.color_image = ros_numpy.numpify(msg)

    def depthCallback(self, msg):
        numpyImage = ros_numpy.numpify(msg)
        numpyImage = np.nan_to_num(numpyImage, copy=True, nan=0.0)
        self.depth_image = numpyImage

    def goalCallback(self, msg):
        self.goal = msg.pose

    def bboxCallback(self, msg):
        self.bboxes = []
        num_box = len(msg.bounding_boxes)
        for bbox in msg.bounding_boxes:
            if (num_box)>0:
                box=[0]*4
                box[0]=bbox.xmin
                box[1]=bbox.ymin
                box[2]=bbox.xmax
                box[3]=bbox.ymax
                self.bboxes.append(box)

    def publishObjectPos3MarkerArray(self,camPos)-> None:
        objectMarkerArray = MarkerArray()
        colors = self.markerColors
        id = 0
        numClass = 0 
        objectMarker = Marker()
        objectMarker.color.r = 255
        objectMarker.color.g = 0
        objectMarker.color.b = 0
        objectMarker.color.a = 1.0
        objectMarker.header.frame_id = self.header.frame_id # Camera Optical Frame
        objectMarker.header.stamp = rospy.Time.now()
        objectMarker.type = 2 # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        # Set the scale of the marker
        objectMarker.scale.x = 0.25
        objectMarker.scale.y = 0.25
        objectMarker.scale.z = 0.25
        # Set the color
        objectMarker.id = id
        objectMarker.pose.position.x = camPos[0]
        objectMarker.pose.position.y = camPos[1]
        objectMarker.pose.position.z = camPos[2]
        objectMarker.lifetime = rospy.Duration(0.1)
        objectMarkerArray.markers.append(objectMarker)
        id += 1
        # if objectMarker.pose.position.z>0.001:
        self.object_pub.publish(objectMarker)
        
    
    def publishImage(self,image):
        msg = ros_numpy.msgify(Image, image,encoding = "bgr8")
        # Publish new image
        self.image_pub.publish(msg)
    

    def pixel_crop(self,img, dim,pixel):
        width, height = img.shape[1], img.shape[0]
        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(pixel[0]), int(pixel[1])
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
        
    def depthPixelToPoint3(self, depth_image, U, V):
        x = (U - self.K[2])/self.K[0]
        y = (V - self.K[5])/self.K[4] 
        # print(x,y)     
        z = depth_image[V,U]*1/1000
        x *= z
        y *= z
        # print(x,y,z)
        point = [x,y,z]
        return point
    
    def pos3FromBboxes(self, depthImage, bboxes:list):
        pos3List:List =[]
        for bbox in bboxes:
            cX,cY = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
            pos3 = self.depthPixelToPoint3(depthImage, cX, cY)
            pos3List.append(pos3)
        # print(pos3List)
        cv2.circle(self.color_image, (cX, cY), 5, (0, 0, 255), -1)
        cv2.imshow("circle", self.color_image)
        cv2.waitKey(1)
        return pos3List

    def cancelGoal(self):
        # os.system("rostopic pub /move_base/cancel actionlib_msgs/GoalID -- {}")
        cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        cancel_msg = GoalID()
        cancel_pub.publish(cancel_msg)
        print("published cancel_goal signal")

    def process(self):
        color_image, depth_image    = self.color_image, self.depth_image
        if not len(self.bboxes) == 0:
            camPoses = self.pos3FromBboxes(depth_image,self.bboxes)
            minZ=5
            mincount = 0
            for count, camPos in enumerate(camPoses):
                if (camPos[0]==0 and camPos[1]==0 and camPos[2]==0):
                    return
                
                # print(camPos)
                if (camPoses[count][2]<minZ):
                    minZ=camPoses[count][2]
                    mincount=count
                # self.publishObjectPos3MarkerArray(camPos)
                # if (-1*self.x_lim<camPos[0]<self.x_lim and camPos[2]<self.z_lim):
                #     self.cancelGoal()
                # cv2.waitKey(1)
            if (camPoses[mincount][0]==0 and camPoses[mincount][1]==0 and camPoses[mincount][2]==0):
                return
            if (camPoses[mincount][2]<2 and -60<math.tan(camPoses[mincount][2]/camPoses[mincount][0])<60):
                self.publishObjectPos3MarkerArray(camPoses[mincount])
                print("Bounding box no. : " ,camPoses[mincount])
                print("")
                print("Distance : " ,camPoses[mincount][2])
                self.cancelGoal()
                cv2.waitKey(1)
        # testpos= [1,0,2]
        # self.publishObjectPos3MarkerArray(testpos)
        # testpos1= [1.5,0,1]
        # self.publishObjectPos3MarkerArray(testpos1)
        # cv2.waitKey(1)
        

    def rosinit(self):
        rospy.init_node('markerFinder', anonymous=True)
        rospy.Subscriber(self.CAMINFO['topic'], self.CAMINFO['msg'], self.camInfoCallback)
        rospy.Subscriber(self.COLOR['topic'], self.COLOR['msg'], self.colorCallback)
        rospy.Subscriber(self.DEPTH['topic'], self.DEPTH['msg'], self.depthCallback)
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bboxCallback)
        # rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goalCallback)
        self.object_pub     = rospy.Publisher( '/object/person',    Marker , queue_size=10)
                    
        while not rospy.is_shutdown():
                if self.isCamInfo: #Wait for camera ready
                    self.process()

if __name__ == '__main__':
    try:
        person_detector = PersonDetecor()
        person_detector.rosinit()

    except rospy.ROSInterruptException:
        pass