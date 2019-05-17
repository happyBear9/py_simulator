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
        #decay epsilon
        self.epsilon = 1.0
        self.gamma = 0.8
        self.alpha = 0.4

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

        self.horizon = 40
        #rewards
        self.travel_empty_cost = 1
        self.collision_with_wall_cost = 10
        self.travel_with_resource_cost = 1
        self.pick_when_full_cost = 10
        self.pick_when_empty_cost = -20
        self.pick_dud_cost = 20
        self.drop_success_cost = -20
        self.drop_wrong_spot_cost = 20
        self.drop_nothing_cost = 20

        #sim parameters
        self.sim_result = None
        self.sim_control = None
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_axes([0, 0, 1, 1],autoscale_on=True, frameon=False)
        self.ax.set_xlim(-2, 9), self.ax.set_xticks([])
        self.ax.set_ylim(-2, 9), self.ax.set_yticks([])
        self.r_sim = np.array([[1,1],[2,4],[4,6],[6,0]])
        self.s_prev = np.empty([3])
        self.r_perm = np.array([[1,1],[2,4],[4,6],[6,0]])

    def get_state_idx(self,x):
        idx = x[6] + x[5]*2 + x[4]*2**2 + x[3]*2**3 +x[2]*2**4 + x[1]*2**5 + x[0]*2**5*7
        return int(idx)

    def get_state(self,idx):
        x = np.empty([7])
        x[0] = idx/(2**5*7)
        rem = idx%(2**5*7)
        x[1] = rem/(2**5)
        rem = rem%(2**5)
        x[2] = rem/(2**4)
        rem = rem%(2**4)
        x[3] = rem/(2**3)
        rem = rem%(2**3)
        x[4] = rem/(2**2)
        rem = rem%(2**2)
        x[5] = rem/(2)
        X[6] = rem%(2)
        return x

    def get_action(self,a_idx):
        a = np.zeros(6)
        a[a_idx] = 1
        return a

    def get_next_state_and_cost(self,s,u):
        cost = 0
        r_curr = self.r.copy()

        pos = s[0:2]
        res = s[2]
        s_next = s.copy()
        #extract current action
        a = np.argmax(self.get_action(u))  

        # print "current state: " + str(s)
        # print "action: " + str(a)
        # print "current resources: " + str(r_curr)
        
        if a < 4:
            #movement action
            # print "movement action"
            next_pos = self.m[a] + pos
            if next_pos[0] < 0 or next_pos[0] > 6 or next_pos[1] < 0 or next_pos[1] > 6:
                #wall collision
                return s_next,self.collision_with_wall_cost
            else:
                s_next[0:2] = next_pos
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
                    return s_next,self.pick_when_full_cost
                else:
                    #check if picked in resource position
                    if any(np.array_equal(pos, k) for k in self.r):
                        # print "picked up resource"
                        #remove resource from r list  
                        for i,k in enumerate(self.r):
                            if np.array_equal(pos, k):
                                    r_curr = np.delete(r_curr,i,axis=0)
                                    break
                        
                        s_next[3:] = np.zeros(4)
                        for i,k in enumerate(r_curr):
                            if np.array_equal(self.r_perm[0], k):
                                s_next[3] = 1
                            elif np.array_equal(self.r_perm[1], k):
                                s_next[4] = 1
                            elif np.array_equal(self.r_perm[2], k):
                                s_next[5] = 1
                            elif np.array_equal(self.r_perm[3], k):
                                s_next[6] = 1
                            else:
                                pass
   
                        cost = self.pick_when_empty_cost
                        s_next[2] = 1
                    else:
                        # print "dud pick"
                        s_next[2] = 0
                        cost = self.pick_dud_cost

                    self.r = r_curr
                    
                    return s_next,cost
            else:
                #drop action
                if res:
                    #drop carrying resource
                    if np.array_equal(pos, self.home):
                        # print "dropped in home position - congrats"
                        res = 0
                        cost = self.drop_success_cost
                        s_next[2] = 0
                    else:
                        # print "dropped at wrong spot"
                        res = 0
                        #add resource to new spot
                        r_curr = np.append(r_curr,[pos],axis=0)
                        # print r_curr
                        self.r = r_curr
                        cost = self.drop_wrong_spot_cost
                        s_next[2] = 0
                    return s_next,cost
                else:
                    # print "dropped nothing"
                    cost = self.drop_nothing_cost
                    return s_next,cost
              
    def simulate(self,x0):
        horizon = self.horizon
        P = self.policy.copy()
        x=np.empty([7, horizon+1])
        x[:,0] = x0.copy()
        u = np.empty([horizon])
        total_cost = 0
        for i in range(horizon):
            xt = x[:,i].copy()
            ut = P[self.get_state_idx(xt)]
            x[:,i+1],cost = self.get_next_state_and_cost(xt, int(ut))
            total_cost += cost
            u[i] = ut
        return x, u, total_cost

    def decay_epsilon(self,i,num_iter):
        self.epsilon = 1.0 * (1 - float(i)/num_iter)

    def init_animate(self):
        self.r_sim = self.r_perm.copy()
        self.s_prev = self.sim_result[:,0]
        print "sim restarted"

    def animate(self,i):
        #offset
        self.ax.clear()

        xo,yo = 0.5,0.5
        borders_x = [0,0,7,7,0]
        borders_y = [0,7,7,0,0]     
        r = self.r_sim.copy()
        s = self.sim_result[:,i]

        
        robot_x_empty = []
        robot_y_empty = []
        robot_x_full = []
        robot_y_full = []

        #robot1
        if self.s_prev[2] == 0 and s[2] == 1:
            #robot 1 picked up resource
            for j,k in enumerate(self.r_sim):
                if np.array_equal(s[0:2], k):
                    r = np.delete(r,j,axis=0)
                    break
            robot_x_full.append(s[0]+ xo)
            robot_y_full.append(s[1]+ yo)
        elif self.s_prev[2] == 1 and s[2] == 0:
            #robot1 dropped resource
            r = np.append(r,[s[0:2]],axis=0)
            robot_x_empty.append(s[0]+ xo)
            robot_y_empty.append(s[1]+ yo)
        elif self.s_prev[2] == 1 and s[2] == 1:
            robot_x_full.append(s[0]+ xo)
            robot_y_full.append(s[1]+ yo)
        elif self.s_prev[2] == 0 and s[2] == 0:
            robot_x_empty.append(s[0]+ xo)
            robot_y_empty.append(s[1]+ yo)

        self.s_prev = s.copy()
        self.r_sim = r.copy()

        r = r.T
        xr = r[0] + xo
        yr = r[1] + yo
        xh,yh = 3+xo,3+yo

        #text overlays
        t1 = "Step: " + str(i+1)
        t2 = "Robot Position(x,y): (" + str(s[0]) + "," + str(s[1]) + ")" 
        if s[2]:
            t3 = "Robot carrying resource - TRUE"
        else:
            t3 = "Robot carrying resource - FALSE"
        a = int(self.sim_control[i])

        if a == 0:
            t4 = "Next action - up"
        elif a == 1:
            t4 = "Next action - down"
        elif a == 1:
            t4 = "Next action - left"
        elif a == 1:
            t4 = "Next action - right"
        elif a == 1:
            t4 = "Next action - pick"
        else:
            t4 = "Next action - drop"

       
        
        self.ax.text(0.2, 6.7, t1, ha='left', wrap=True)
        self.ax.text(0.2, 6.5, t2, ha='left', wrap=True)
        self.ax.text(0.2, 6.3, t3, ha='left', wrap=True)
        self.ax.text(0.2, 6.1, t4, ha='left', wrap=True)

        
        self.ax.scatter(robot_x_full,robot_y_full,color='red',marker ='o',s=10**2.5,alpha = 0.2)
        self.ax.scatter(robot_x_empty,robot_y_empty,color='green',marker ='o',s=10**2.5,alpha = 0.2)
        self.ax.scatter(xh,yh,color='blue',marker ='s',s=10**3,alpha = 0.2)
        self.ax.scatter(xr,yr,color='red',alpha = 0.2)
        self.ax.plot(borders_x,borders_y)


    def learn(self,iter):
        #initiate learning episode
        horizon = self.horizon
        for i in range(iter):
            Q_new = self.q_function.copy()
            J_new = self.value_function.copy()
            P_new = self.policy.copy()

            x = np.empty([7, horizon+1])
            #initial position
            x[:,0] = np.array([2,3,0,1,1,1,1])


            for j in range(horizon):
                xt = x[:,j].copy()
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
                x_nxt_idx = self.get_state_idx(x_nxt)
                opt_next_state_Q = np.min(self.q_function[x_nxt_idx,:])
                current_Q = self.q_function[xt_idx,u_curr]
                TD = cost + self.gamma*opt_next_state_Q - current_Q
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
            #repop resources
            self.r = self.r_perm.copy()
            self.q_function = Q_new.copy()

            self.decay_epsilon(i,iter)
            # print x[:,0],i


        print "learning completed"
        print "simulating env."


        #init condition
        x0 = x[:,0]
        X, U, total_cost = self.simulate(x0)

        print ""
        print X
        print ""
        print U

        self.s_prev = X[:,0]

        print "Total cost: " + str(total_cost)

        #show simulated results
        self.sim_result = X
        self.sim_control = U
        ani = animation.FuncAnimation(self.fig,self.animate,init_func=self.init_animate, frames=self.horizon,interval=500)
        plt.show()


def init_q_learning():
    # print "Initializing data"
    robots = 1
    grid_size = 7
    bin_actions = 2

    #states - robots' position, robot resource condition
    nq = grid_size*grid_size*2*2*2*2*2
    #actions - stay,up,down,right,left,pick,drop
    nu = 6**robots 

    student = q_learning(nq,nu) 
    student.learn(5000)



if __name__ == '__main__':
    init_q_learning()