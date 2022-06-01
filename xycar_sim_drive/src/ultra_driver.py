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

    fl = msg.data[0]

    fm = msg.data[1]

    fr = msg.data[2]

    r = msg.data[6]

    l = msg.data[7]
    print(msg.data)

rospy.init_node('guide')
motor_pub = rospy.Publisher('xycar_motor_msg', Int32MultiArray, queue_size=1)
ultra_sub = rospy.Subscriber('ultrasonic', Int32MultiArray, callback)

xycar_msg = Int32MultiArray()

while not rospy.is_shutdown():
    angle = 0
    if fm <= 300:
        if (fr-fl)>100:
            angle=50
        elif (fr-fl)<-100:
            angle = -50
        else:
            angle = 0
    else:
        if fl <= 100:
            angle = 50
        if fr <= 100:
            angle = -50

    xycar_msg.data = [angle, 100]
    motor_pub.publish(xycar_msg)
