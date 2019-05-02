#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import json
import time
import numpy as np

import constants

self_id = rospy.get_param('selfID',0)


#PID
dI = [0.0,0.0]  
prev_pos = None
dt = 0.1

Kp = constants.Kp
Ki = constants.Ki

pub = rospy.Publisher('/sim_input', String, queue_size=10)

dst = [-6,-6]


robots = []


def update_force(src):

    
    fx,fy = get_forces(src,dst)


    robot = {}
    robot["name"] = self_id
    robot["force"] = [fx,fy]

    print src

    json_string = json.dumps(robot)
    pub.publish(json_string)


def callback(data):
    robots =  json.loads(data.data)
    
    for k in robots.keys():
        if k == str(self_id):
            src = robots[k]

    update_force(src)


def get_forces(src,dst):
    global dI,prev_pos

    fx_i = 0.0
    fy_i = 0.0

    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    
    fx_p = Kp*dx
    fy_p = Kp*dy

    if prev_pos == None:
        prev_pos = src
    else:
        dI[0] = dI[0] + (src[0]-prev_pos[0])*dt
        dI[1] = dI[1] + (src[1]-prev_pos[1])*dt
        prev_pos = src

    fx_i = Ki*dI[0]
    fy_i = Ki*dI[1]

    fx = fx_p + fx_i
    fy = fy_p + fy_i

    print dI

    return fx,fy


def talker():
    
    rospy.init_node('talker', anonymous=True)
    rospy.Subscriber("/sim_output", String, callback)
    

    rospy.spin()

    # rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    #     update_force()
    #     rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass





# def wait(time_delay_in_secs):
#     t_end = time.time() + time_delay_in_secs
#     while time.time() < t_end: 
#         pass
