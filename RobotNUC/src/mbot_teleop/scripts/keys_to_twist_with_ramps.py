#!/usr/bin/env python

import rospy
import math
from std_msgs.msg import String
from geometry_msgs.msg import Twist

key_mapping = {'A': [0, 1], 'C': [0, -1],
               'D': [1, 0], 'F': [-1, 0],
               'L': [0, 0]}

g_twist_pub = None
g_vel_scales = [0.1, 0.1]

def keys_cb(msg):
    global g_twist_pub, g_vel_scales
    if len(msg.data) == 0 or not key_mapping.has_key(msg.data[0]):
        return

    vels = key_mapping[msg.data[0]]
    twist = Twist()
    twist.angular.z = vels[0] * g_vel_scales[0]
    twist.linear.x = vels[1] * g_vel_scales[1]

    g_twist_pub.publish(twist)

def fetch_param(name, default):
    if rospy.has_param(name):
        return rospy.get_param(name)
    else:
        rospy.logwarn("Parameter [%s] not defined. Defaulting to %.3f" % (name, default))
        return default

if __name__ == '__main__':
    rospy.init_node('keys_to_twist')

    g_twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('keys', String, keys_cb)

    g_vel_scales[0] = fetch_param('~angular_scale', 0.1)
    g_vel_scales[1] = fetch_param('~linear_scale', 0.1) 

    rospy.spin()

