#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class q_learning:

    def __init__(self,nq,nu):
        self.epsilon = 0.1
        self.gamma = 0.99
        self.alpha = 0.1

        self.value_function = np.zeros([nq])
        self.policy = np.zeros([nu])
        self.q_function = np.zeros([nq,nu])
        #home position
        self.home = np.array([3,3])
        #action-position array
        self.m = np.array([[0,1],[0,-1],[-1,0],[1,0]])
        #resource position
        self.r = np.array([[1,1],[2,4],[4,6],[6,0]])



    def cost(self):
        print "calculate current cost"


    def get_state_idx(self,x):
        pos = np.array([0,1,2,3,4,5,6])
        resource = np.array([0,1])
        r1_x_idx = np.argmin((x[0]-pos)**2)
        r1_y_idx = np.argmin((x[1]-pos)**2)
        r2_x_idx = np.argmin((x[2]-pos)**2)
        r2_y_idx = np.argmin((x[3]-pos)**2)
        r1_resource = x[4]
        r2_resource = x[5]
        idx = r1_resource + r2_resource*2 + r2_y_idx*2**2 + r2_x_idx*2**2*7 + r1_y_idx*2**2*7**2 +\
            r1_x_idx*2**2*7**3
        return idx

    def get_state(self,idx):
        x = np.empty([6])
        x[0] = idx/(2**2*7**3)
        rem = idx%(2**2*7**3)
        x[1] = rem/(2**2*7**2)
        rem = rem%(2**2*7**2)
        x[2] = rem/(2**2*7)
        rem = rem%(2**2*7)
        x[3] = rem/(2**2)
        rem = rem%(2**2)
        x[4] = rem/(2)
        x[5] = rem%(2)
        return x

    def get_next_state_and_cost(self,s,a):
        cost = 0
        r_curr = self.r.copy()

        r1_action = np.argmax(a[0:6])
        r2_action = np.argmax(a[6:])
        r1_current_pos = s[0:2]
        r2_current_pos = s[2:4]
        r1_resource_status = s[4]
        r2_resource_status = s[5]


        #r1
        r1_next_pos = r1_current_pos
        if r1_action < 4:
            #movement action of r1 
            r1_next_pos =  self.m[r1_action] + r1_current_pos
            if not r1_resource_status:
                #travelling empty   
                if r1_next_pos[0] < 0 or r1_next_pos[0] > 6 or r1_next_pos[1] < 0 or r1_next_pos[1] > 6:
                    r1_next_pos = r1_current_pos
                    cost += 100
                else:
                    cost += 10
            else:
                #carrying resource
                if r1_next_pos[0] < 0 or r1_next_pos[0] > 6 or r1_next_pos[1] < 0 or r1_next_pos[1] > 6:
                    r1_next_pos = r1_current_pos
                    cost += 100
                else:
                    #swarming cost
                    if np.array_equal(r1_current_pos, r2_current_pos):
                        #traveling together
                        cost += 20
                    else:
                        #traveling alone
                        cost += 30
        else:
            #pick action of r1
            if r1_action == 4:
                #check if already contain resource
                if r1_resource_status:
                    # print "already contains resource"
                    cost += 50
                else:
                    #check if picked in resource position
                    if any((r1_current_pos==k).all() for k in self.r):
                        # print "picked up resource"
                        r1_resource_status = 1
                        #remove resource from r list
                        for i,k in enumerate(self.r):
                            if np.array_equal(r1_current_pos, k):
                                r_curr = np.delete(r_curr,i,axis=0)
                                break
                        cost -= 50
                    else:
                        # print "dud pick"
                        cost += 20                
            else:
                #drop action of r1
                #check resource status
                if r1_resource_status:
                    #check if in home position
                    if np.array_equal(r1_current_pos, self.home):
                        # print "dropped in home position - congrats"
                        r1_resource_status = 0
                        cost -=300
                    else:
                        # print "dropped at wrong spot"
                        r1_resource_status = 0
                        #add resource to new spot
                        r_curr = np.append(self.r,[r1_current_pos],axis=0)
                        cost += 50
                else:
                    # print "dropped nothing"
                    cost += 30

        #update current resource map
        self.r = r_curr


        #r2
        r2_next_pos = r2_current_pos
        if r2_action < 4:
            #movement action of r2 
            r2_next_pos =  self.m[r2_action] + r2_current_pos
            if not r2_resource_status:
                #travelling empty   
                if r2_next_pos[0] < 0 or r2_next_pos[0] > 6 or r2_next_pos[1] < 0 or r2_next_pos[1] > 6:
                    r2_next_pos = r2_current_pos
                    cost += 100
                else:
                    cost += 10
            else:
                #carrying resource
                if r2_next_pos[0] < 0 or r2_next_pos[0] > 6 or r2_next_pos[1] < 0 or r2_next_pos[1] > 6:
                    r2_next_pos = r2_current_pos
                    cost += 100
                else:
                    #swarming cost
                    if np.array_equal(r2_current_pos, r1_current_pos):
                        #traveling together
                        cost += 20
                    else:
                        #traveling alone
                        cost += 30
        else:
            #pick action of r2
            if r2_action == 4:
                #check if already contain resource
                if r2_resource_status:
                    # print "already contains resource"
                    cost += 50
                else:
                    #check if picked in resource position
                    if any((r2_current_pos==k).all() for k in self.r):
                        # print "picked up resource"
                        r2_resource_status = 1
                        #remove resource from r list
                        for i,k in enumerate(self.r):
                            if np.array_equal(r2_current_pos, k):
                                r_curr = np.delete(r_curr,i,axis=0)
                                break
                        cost -= 50
                    else:
                        # print "dud pick"
                        cost += 20                
            else:
                #drop action of r2
                #check resource status
                if r2_resource_status:
                    #check if in home position
                    if np.array_equal(r2_current_pos, self.home):
                        # print "dropped in home position - congrats"
                        r2_resource_status = 0
                        cost -=300
                    else:
                        # print "dropped at wrong spot"
                        r2_resource_status = 0
                        #add resource to new spot
                        r_curr = np.append(self.r,[r2_current_pos],axis=0)
                        cost += 50
                else:
                    # print "dropped nothing"
                    cost += 30

        #update current resource map
        self.r = r_curr

        # print "current state"
        # print r1_current_pos,r1_resource_status,r2_current_pos,r2_resource_status
        # print "action"
        # print r1_action,r2_action
        # print "next state"
        # print r1_next_pos,r1_resource_status,r2_next_pos,r2_resource_status
        # print "cost: " + str(cost)
        

        r_next_pos = np.concatenate((r1_next_pos, r2_next_pos), axis=None)
        r_resource_status = np.concatenate((r1_resource_status, r2_resource_status), axis=None)
        s_next = np.concatenate((r_next_pos, r_resource_status), axis=None)
        return s_next,cost

    def get_action_idx(self,a):
        msb = a[0:6]
        lsb = a[6:]
        a_idx =  np.argmax(msb)*6 + np.argmax(lsb)
        print a_idx

    def get_action(self,a_idx):
        msb = np.zeros(6)
        lsb = np.zeros(6)
        msb_idx = a_idx/6
        lsb_idx = a_idx%6
        msb[msb_idx] = 1
        lsb[lsb_idx] = 1
        a = np.concatenate((msb, lsb), axis=None)
        return a

    def iterate(self,iter):
        horizon = 1000
        for i in range(iter):
            Q_new = self.q_function.copy()
            J_new = self.value_function.copy()
            P_new = self.policy.copy()

            x = np.empty([6, horizon+1])
            #initial position
            x[:,0] = np.array([2,3,4,3,0,0])
            for j in range(1):
                print j


def init_q_learning():
    # print "Initializing data"
    robots = 2
    grid_size = 7

    #states - robots' position, robot resource condition
    nq = (grid_size*grid_size)**robots*2**robots
    #actions - up,down,right,left,pick,drop
    nu = 6**robots 

    student = q_learning(nq,nu) 
    student.iterate(1)



if __name__ == '__main__':
    init_q_learning()