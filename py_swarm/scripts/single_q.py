#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class q_learning:

    def __init__(self,nq,nu):
        self.epsilon = 0.1
        self.gamma = 0.99
        self.alpha = 0.1

        self.nu = nu
        self.nq = nq

        self.value_function = np.zeros([nq])
        self.policy = np.zeros([nq])
        self.q_function = np.zeros([nq,nu])
        #home position
        self.home = np.array([3,3])
        #action-position array
        self.m = np.array([[0,1],[0,-1],[-1,0],[1,0]])
        #resource position
        self.r = np.array([[1,1],[2,4],[4,6],[6,0]])
        self.r_perm = np.array([[1,1],[2,4],[4,6],[6,0]])

        self.horizon = 4

        self.total_cost = 100

        #rewards
        self.nothing_cost = 0.05
        self.travel_empty_cost = 0.2
        self.collision_with_wall_cost = 1
        self.travel_with_resource_cost = 0.5
        self.pick_when_full_cost = 1
        self.pick_when_empty_cost = -5
        self.pick_dud_cost = 1
        self.drop_success_cost = -10
        self.drop_wrong_spot_cost = 1
        self.drop_nothing_cost = 1


    def get_state_idx(self,x):
        idx = x[2] + x[1]*2 + x[0]*7*2
        return int(idx)

    def get_state(self,idx):
        x = np.empty([3])
        x[0] = idx/(2*7)
        rem = idx%(2*7)
        x[1] = rem/(2)
        rem = rem%(2)
        x[2] = rem

        return x

    def get_action(self,a_idx):
        a = np.zeros(6)
        if a_idx != 0:
            a[a_idx-1] = 1
        elif a_idx == 6:
            a[a_idx] = 1
        return a

    def get_next_state_and_cost(self,s,u):
        cost = 0
        r_curr = self.r.copy()

        pos = s[0:2]
        res = s[2]

        if u == 0:
            #doing nothing
            # print "idle"
            return s,self.nothing_cost
        else:
            #extract current action
            a = np.argmax(self.get_action(u))  
            
            if a < 4:
                #movement action
                # print "movement action"
                next_pos = self.m[a] + pos
                if next_pos[0] < 0 or next_pos[0] > 6 or next_pos[1] < 0 or next_pos[1] > 6:
                    #wall collision
                    return s,self.collision_with_wall_cost
                else:
                    s_next = np.concatenate((next_pos, res), axis=None)
                    if res:
                        cost = self.travel_with_resource_cost
                    else:
                        cost = self.travel_empty_cost
                    return s_next,cost
            else:
                # print "pick/place action"
                if a == 4:
                    #pick action
                    if res:
                        #trying to pick when full
                        return s,self.pick_when_full_cost
                    else:
                        #check if picked in resource position
                        if any((pos==k).all() for k in self.r):
                            # print "picked up resource"
                            res = 1
                            #remove resource from r list
                            for i,k in enumerate(self.r):
                                if np.array_equal(pos, k):
                                    r_curr = np.delete(r_curr,i,axis=0)
                                    break
                            cost = self.pick_when_empty_cost
                        else:
                            # print "dud pick"
                            cost = self.pick_dud_cost

                        self.r = r_curr
                        s_next = np.concatenate((s[0:2], res), axis=None)
                        return s_next,cost

                else:
                    #drop action
                    if res:
                        #drop carrying resource
                        if np.array_equal(pos, self.home):
                            # print "dropped in home position - congrats"
                            res = 0
                            cost = self.drop_success_cost
                        else:
                            # print "dropped at wrong spot"
                            res = 0
                            #add resource to new spot
                            r_curr = np.append(r_curr,[pos],axis=0)
                            # print r_curr
                            self.r = r_curr
                            cost = self.drop_wrong_spot_cost
                        s_next = np.concatenate((s[0:2], res), axis=None)
                        return s_next,cost
                    else:
                        # print "dropped nothing"
                        cost = self.drop_nothing_cost
                        return s,cost
              
    def simulate(self,x0,P):
        horizon = self.horizon
        x=np.empty([3, horizon+1])
        x[:,0] = x0
        u = np.empty([horizon])
        total_cost = 0
        for i in range(horizon):
            u[i] = P[self.get_state_idx(x[:,i])]
            x[:,i+1],cost = self.get_next_state_and_cost(x[:,i], int(u[i]))
            total_cost += cost
        return x, u, total_cost

    def learn(self,iter):
        #initiate learning episode
        horizon = self.horizon
        for i in range(iter):
            Q_new = self.q_function.copy()
            J_new = self.value_function.copy()
            P_new = self.policy.copy()

            x = np.empty([3, horizon+1])
            #initial position
            x[:,0] = np.array([2,3,0])

            for j in range(horizon):
                xt = x[:,j]
                xt_idx = self.get_state_idx(xt)
              
                #optimal policy at current step 
                u_opt = np.argmin(self.q_function[xt_idx,:])  

                #epsilion-greedy approach
                if np.random.uniform(0, 1) < self.epsilon:
                    u_rand = int(np.random.choice(self.nu,1))
                    while u_rand == u_opt:
                        u_rand = int(np.random.choice(self.nu,1))
                    u_curr = u_rand  
                else:
                    u_curr = u_opt  


                #calculate temporal difference
                x_nxt,cost = self.get_next_state_and_cost(xt,u_curr)
                total_cost =  self.total_cost + cost
                x_nxt_idx = self.get_state_idx(x_nxt)
                opt_next_state_Q = np.min(self.q_function[x_nxt_idx,:])
                current_Q = self.q_function[xt_idx,u_curr]
                TD = total_cost + self.gamma*opt_next_state_Q - current_Q
                #update Q
                Q_new[xt_idx,u_curr] += self.alpha*TD

                #update next state
                x[:,j+1] = x_nxt

                #update value function and policy
                J_new[xt_idx] = opt_next_state_Q
                P_new[xt_idx] = u_opt

            #store new J and P
            self.policy = P_new.copy()
            self.value_function = J_new.copy()
            self.total_cost = total_cost
            #repop resources
            self.r = self.r_perm.copy()

            

            #update Q table
            if ((self.q_function-Q_new)**2 < 10e-5).all():
                print("CONVERGED after iteration " + str(i))
                break
            else:
                self.q_function = Q_new.copy()

        print Q_new[self.get_state_idx(x[:,0]),:],i
        print ""
        print Q_new[self.get_state_idx(x[:,1]),:],i
        print ""
        print Q_new[self.get_state_idx(x[:,2]),:],i
        print ""
        print Q_new[self.get_state_idx(x[:,3]),:],i
        print ""

        #simulate and animate
        print "learning completed"
        print "simulating env."
        #init condition
        x0 = x[:,0]
        p = self.policy.copy()

        X, U, total_cost = self.simulate(x0,p)

        print X
        print ""
        print U

        # self.s_prev = X[:,0]

        # print "Total cost: " + str(total_cost)

        # #show simulated results
        # self.sim_result = X
        # self.sim_control = U
        # ani = animation.FuncAnimation(self.fig,self.animate,init_func=self.init_animate, frames=self.horizon,interval=200)
        # plt.show()


def init_q_learning():
    # print "Initializing data"
    robots = 1
    grid_size = 7
    bin_actions = 2

    #states - robots' position, robot resource condition
    nq = (grid_size*grid_size)*bin_actions**2
    #actions - stay,up,down,right,left,pick,drop
    nu = 7**robots 

    student = q_learning(nq,nu) 
    student.learn(100)





if __name__ == '__main__':
    init_q_learning()