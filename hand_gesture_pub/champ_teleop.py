#!/usr/bin/env python
#credits to: https://github.com/ros-teleop/teleop_twist_keyboard/blob/master/teleop_twist_keyboard.py

from __future__ import print_function

import roslib; roslib.load_manifest('champ_teleop')
import rospy

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from champ_msgs.msg import Pose as PoseLite
from geometry_msgs.msg import Pose as Pose
import tf

import sys, select, termios, tty
import numpy as np

import cv2
import mediapipe as mp
import time
import numpy as np
import tensorflow as tf
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

#arduinoData = serial.Serial('com7', 9600)

loaded_model = tf.keras.models.load_model('../resource/CNN_LSTM_Latest.h5', compile=False)
detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)
global Data
Data = np.zeros((1, 63, 1))
pred = np.zeros((3, 1))

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


class Teleop:
    def test():
        print('Test function')


    def __init__(self):
        self.velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        self.pose_lite_publisher = rospy.Publisher('body_pose/raw', PoseLite, queue_size = 1)
        self.pose_publisher = rospy.Publisher('body_pose', Pose, queue_size = 1)
        self.joy_subscriber = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.swing_height = rospy.get_param("gait/swing_height", 0)
        self.nominal_height = rospy.get_param("gait/nominal_height", 0)

        self.speed = rospy.get_param("~speed", 0.5)
        self.turn = rospy.get_param("~turn", 1.0)

        self.msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
u    i    o
j    k    l
m    ,    .
For Holonomic mode (strafing), hold down the shift key:
---------------------------
U    I    O
J    K    L
M    <    >
t : up (+z)
b : down (-z)
anything else : stop
q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
CTRL-C to quit
        """

        self.velocityBindings = {
                'i':(1,0,0,0),
                'o':(1,0,0,-1),
                'j':(0,0,0,1),
                'l':(0,0,0,-1),
                'u':(1,0,0,1),
                ',':(-1,0,0,0),
                '.':(-1,0,0,1),
                'm':(-1,0,0,-1),
                'O':(1,-1,0,0),
                'I':(1,0,0,0),
                'J':(0,1,0,0),
                'L':(0,-1,0,0),
                'U':(1,1,0,0),
                '<':(-1,0,0,0),
                '>':(-1,-1,0,0),
                'M':(-1,1,0,0),
                'v':(0,0,1,0),
                'n':(0,0,-1,0),
            }

        self.poseBindings = {
                'f':(-1,0,0,0),
                'h':(1,0,0,0),
                't':(0,1,0,0),
                'b':(0,-1,0,0),
                'r':(0,0,1,0),
                'y':(0,0,-1,0),
            }

        self.speedBindings={
                'q':(1.1,1.1),
                'z':(.9,.9),
                'w':(1.1,1),
                'x':(.9,1),
                'e':(1,1.1),
                'c':(1,.9),
            }
        
        self.poll_keys()

    def joy_callback(self, data):
        twist = Twist()
        twist.linear.x = data.axes[1] * self.speed
        twist.linear.y = data.buttons[4] * data.axes[0] * self.speed
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = (not data.buttons[4]) * data.axes[0] * self.turn
        self.velocity_publisher.publish(twist)

        body_pose_lite = PoseLite()
        body_pose_lite.x = 0
        body_pose_lite.y = 0
        body_pose_lite.roll = (not data.buttons[5]) *-data.axes[3] * 0.349066
        body_pose_lite.pitch = data.axes[4] * 0.174533
        body_pose_lite.yaw = data.buttons[5] * data.axes[3] * 0.436332
        if data.axes[5] < 0:
            body_pose_lite.z = data.axes[5] * 0.5

        self.pose_lite_publisher.publish(body_pose_lite)

        body_pose = Pose()
        body_pose.position.z = body_pose_lite.z

        quaternion = tf.transformations.quaternion_from_euler(body_pose_lite.roll, body_pose_lite.pitch, body_pose_lite.yaw)
        body_pose.orientation.x = quaternion[0]
        body_pose.orientation.y = quaternion[1]
        body_pose.orientation.z = quaternion[2]
        body_pose.orientation.w = quaternion[3]

        self.pose_publisher.publish(body_pose)

    def poll_keys(self):
        self.settings = termios.tcgetattr(sys.stdin)

        x = 0
        y = 0
        z = 0
        th = 0
        roll = 0
        pitch = 0
        yaw = 0
        status = 0
        cmd_attempts = 0

        try:
            print(self.msg)
            print(self.vels( self.speed, self.turn))
            
            while not rospy.is_shutdown():
                
                success, img = cap.read()
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)

                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                            for id, lm in enumerate(handLms.landmark):
                                Data[:, id*3] = lm.x
                                Data[:, id*3+1] = lm.y
                                Data[:, id*3+2] = lm.z


                pred = loaded_model.predict(Data)

                if pred[:, 0] >=0.7:
                    x = 1
                    y = 0
                    z = 0
                    th = 0
                    #arduinoData.write(b'1')
                    #cv2.putText(img, str("Start"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 1] >=0.7:
                    x = 0
                    y = 0
                    z = 0
                    th = 0
                # arduinoData.write(b'0')
                    #cv2.putText(img, str("Stop"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 2] >=0.7:
                    x = -1
                    y = 0
                    z = 0
                    th = 0
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 3] >=0.7:
                    x = 0
                    y = 0
                    z = 0
                    th =-1
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 4] >= 0.7:
                    x = 0
                    y = 0
                    z = 0
                    th =1
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 5] >=0.7:
                    x = 1
                    y = 0
                    z = 0
                    th = 1
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 6] >=0.7:
                    x = 1
                    y = 0
                    z = 0
                    th = -1
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 7] >=0.7:
                    x = -1
                    y = 0
                    z = 0
                    th = -1
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 8] >=0.7:
                    x = -1
                    y = 0
                    z = 0
                    th = 1
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 9] >=0.7:
                    print("Increase Velocity")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 10] >=0.7:
                    print("Decrease Velocity")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 11] >=0.7:
                    print("Repeat Command")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 12] >=0.7:
                    print("Buckle")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 13] >=0.7:
                    print("Hold")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 14] >=0.7:
                    print("Palmer")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 15] >=0.7:
                    print("Spherical")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 16] >=0.7:
                    print("Tip")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 17] >=0.7:
                    print("Clockwise")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 18] >=0.7:
                    print("Anti-Clockwise")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if pred[:, 19] >=0.7:
                    print("Spare")
                    #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

                success, img1 = cap.read()
                hand, img1 = detector.findHands(img1)
                
                cv2.imshow("Image", img1)
                cv2.waitKey(1)
                
                

                    
                twist = Twist()
                twist.linear.x = x *self.speed
                twist.linear.y = y * self.speed
                twist.linear.z = z * self.speed
                twist.angular.x = 0
                twist.angular.y = 0
                twist.angular.z = th * self.turn
                self.velocity_publisher.publish(twist)


        except Exception as e:
            print(e)

        finally:
            twist = Twist()
            twist.linear.x = 0
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = 0
            self.velocity_publisher.publish(twist)

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        
    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def vels(self, speed, turn):
        return "currently:\tspeed %s\tturn %s " % (speed,turn)

    def map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

if __name__ == "__main__":
    rospy.init_node('champ_teleop')
    teleop = Teleop()