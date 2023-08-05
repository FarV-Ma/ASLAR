#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def publish_message():
    rospy.init_node('message_publisher')
    message_pub = rospy.Publisher('keys', String, queue_size=1)

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        message = String()
        message.data = "null"

        message_pub.publish(message)
        rospy.loginfo('Published message: %s', message.data)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_message()
    except rospy.ROSInterruptException:
        pass

