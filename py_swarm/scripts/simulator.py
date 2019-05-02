#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


pub = rospy.Publisher('sim_output', String, queue_size=10)

robots = []

#time step
dt = 0.1

robot_initial_positions = {1: [0,0], 2: [-1,1], 3: [1,-1], 4: [1,1], 5: [-1,-1], \
                           6: [0,1],7: [0,-1],8: [-1,0],9: [1,0] }
resource_positions = [[6,-6,-6,6],[6,6,-6,-6]]


# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1],autoscale_on=True, frameon=False)
ax.set_xlim(-11, 11), ax.set_xticks([])
ax.set_ylim(-11, 11), ax.set_yticks([])

class robot:

    def __init__(self,name,force):
        self.name = name
        self.force = force
        self.position = robot_initial_positions[name]


def callback(data):
    inData =  json.loads(data.data)

    name = inData["name"]
    force = inData["force"]
 
    for r in robots:
       if r.name == name:
            r.force = force
            break

  
def update_sim(i):

    borders_x = [-10,-10,10,10,-10]
    borders_y = [-10,10,10,-10,-10]
    
    xar = []
    yar = []

    for r in robots:
        position = r.position
        fx,fy = r.force[0],r.force[1]
        x = position[0] + dt*fx
        y = position[1] + dt*fy
        r.position = [x,y]
        xar.append(x)
        yar.append(y)

    # print xar,yar

    pub_data()

    ax.clear()
    ax.scatter(xar,yar,color='red',alpha = 0.3)
    draw_resources()
    ax.plot(borders_x,borders_y)

def pub_data():
    to_send = {}
    for r in robots:
        to_send[r.name] = r.position

    json_string = json.dumps(to_send)
    pub.publish(json_string)


def initialize_sim():
    borders_x = [-10,-10,10,10,-10]
    borders_y = [-10,10,10,-10,-10]
    
    xar = []
    yar = []

    for k in robot_initial_positions.keys():
        name = k
        force = [0,0]
        r = robot(name,force)

        position = robot_initial_positions[k]
        x = position[0] 
        y = position[1]
        xar.append(x)
        yar.append(y)
        robots.append(r)
   
    print "sim initialized"

    ax.clear()
    ax.scatter(xar,yar,color='red')
    draw_resources()
    ax.plot(borders_x,borders_y)

def draw_resources():
    xr = resource_positions[0]
    yr = resource_positions[1]
    ax.scatter(xr,yr,color='green',marker ='H',s=10**3,alpha = 0.2)


    
def listener():

    rospy.init_node('sim_node', anonymous=True)

    rospy.Subscriber("sim_input", String, callback)
   
    #plot details
    initialize_sim()
    ani = animation.FuncAnimation(fig,update_sim,interval=25)
    plt.show()

    rospy.spin()


    # rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    #     pub_data()
    #     rate.sleep()

if __name__ == '__main__':
    listener()
