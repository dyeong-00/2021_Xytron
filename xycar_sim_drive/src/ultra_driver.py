#!/usr/bin/env python

import rospy, math
from std_msgs.msg import Int32MultiArray

fl = 0

fm = 0

fr = 0

r = 0

l = 0


def callback(msg):
    global fl

    global fm

    global fr

    global r

    global l
    
    # 초음파 센서 메세지
    fl = msg.data[0] # 정면좌측

    fm = msg.data[1] # 정면중앙

    fr = msg.data[2] # 정면우측

    r = msg.data[6] # 우측

    l = msg.data[7] #좌측
    print(msg.data)

rospy.init_node('guide')
motor_pub = rospy.Publisher('xycar_motor_msg', Int32MultiArray, queue_size=1)
ultra_sub = rospy.Subscriber('ultrasonic', Int32MultiArray, callback)

xycar_msg = Int32MultiArray()

while not rospy.is_shutdown():
    angle = 0
    if fm <= 300:
        # 정면에 벽이 가까워지고 있는 경우
        if (fr-fl)>100: # 차량 우측에 길이 나있는 경우
            angle=50 # 우회전
        elif (fr-fl)<-100: #차량 좌측에 길이 나있는 경우
            angle = -50 #좌회전
        else:
            angle = 0
    else:
        # 차선 중앙에서 주행하도록 steering angle 조정
        if fl <= 100:
            angle = 50
        if fr <= 100:
            angle = -50

    xycar_msg.data = [angle, 100]
    motor_pub.publish(xycar_msg)
