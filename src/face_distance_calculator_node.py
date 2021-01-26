#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('unibas_face_distance_calculator')
import sys
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class face_detector:

  def __init__(self):
    self.bridge = CvBridge()
    
    self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
    
    self.pub = rospy.Publisher('/unibas_face_detector/faces', Image, queue_size=1)	

  def callback(self, rgb_data):
    
    try:
      img = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
      face_cascade = cv2.CascadeClassifier('/home/bloisi/catkin_ws/src/unibas_face_detector/haarcascade/haarcascade_frontalface_default.xml')
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
            
    except CvBridgeError as e:
      print(e)

    #convert opencv format back to ros format and publish result
    try:
      faces_message = self.bridge.cv2_to_imgmsg(img, "bgr8")
      self.pub.publish(faces_message)
    except CvBridgeError as e:
      print(e)
    

def main(args):
  rospy.init_node('face_distance_calculator_node', anonymous=True)
  fd = face_detector()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)

