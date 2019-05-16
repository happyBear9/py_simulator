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
        self.horizon = 100

        #sim parameters
        self.sim_result = None
        self.sim_control = None
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_axes([0, 0, 1, 1],autoscale_on=True, frameon=False)
        self.ax.set_xlim(-2, 9), self.ax.set_xticks([])
        self.ax.set_ylim(-2, 9), self.ax.set_yticks([])
        self.r_sim = np.array([[1,1],[2,4],[4,6],[6,0]])
        self.s_prev = np.empty([6])
        self.r_perm = np.array([[1,1],[2,4],[4,6],[6,0]])

        #rewards
        self.travel_empty_cost = 20
        self.collision_with_wall_cost = 100
        self.travel_with_resource_alone_cost = 50
        self.travel_with_resource_together = 25
        self.pick_when_full_cost = 100
        self.pick_when_empty_cost = 500
        self.pick_dud_cost = 100
        self.drop_success_cost = 1000
        self.drop_wrong_spot_cost = 100
        self.drop_nothing_cost = 100

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
        return int(idx)

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
                    cost += self.collision_with_wall_cost
                else:
                    cost += self.travel_empty_cost
            else:
                #carrying resource
                if r1_next_pos[0] < 0 or r1_next_pos[0] > 6 or r1_next_pos[1] < 0 or r1_next_pos[1] > 6:
                    r1_next_pos = r1_current_pos
                    cost += self.collision_with_wall_cost
                else:
                    #swarming cost
                    if np.array_equal(r1_current_pos, r2_current_pos):
                        #traveling together
                        cost += self.travel_with_resource_together
                    else:
                        #traveling alone
                        cost += self.travel_with_resource_alone_cost
        else:
            #pick action of r1
            if r1_action == 4:
                #check if already contain resource
                if r1_resource_status:
                    # print "already contains resource"
                    cost += self.pick_when_full_cost
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
                        cost -= self.pick_when_empty_cost
                    else:
                        # print "dud pick"
                        cost += self.pick_dud_cost               
            else:
                #drop action of r1
                #check resource status
                if r1_resource_status:
                    #check if in home position
                    if np.array_equal(r1_current_pos, self.home):
                        # print "dropped in home position - congrats"
                        r1_resource_status = 0
                        cost -= self.drop_success_cost
                    else:
                        # print "dropped at wrong spot"
                        r1_resource_status = 0
                        #add resource to new spot
                        r_curr = np.append(self.r,[r1_current_pos],axis=0)
                        cost += self.drop_wrong_spot_cost
                else:
                    # print "dropped nothing"
                    cost += self.drop_nothing_cost

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
                    cost += self.collision_with_wall_cost
                else:
                    cost += self.travel_empty_cost
            else:
                #carrying resource
                if r2_next_pos[0] < 0 or r2_next_pos[0] > 6 or r2_next_pos[1] < 0 or r2_next_pos[1] > 6:
                    r2_next_pos = r2_current_pos
                    cost += self.collision_with_wall_cost
                else:
                    #swarming cost
                    if np.array_equal(r2_current_pos, r1_current_pos):
                        #traveling together
                        cost += self.travel_with_resource_together
                    else:
                        #traveling alone
                        cost += self.travel_with_resource_alone_cost
        else:
            #pick action of r2
            if r2_action == 4:
                #check if already contain resource
                if r2_resource_status:
                    # print "already contains resource"
                    cost += self.pick_when_full_cost
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
                        cost -= self.pick_when_empty_cost
                    else:
                        # print "dud pick"
                        cost += self.pick_dud_cost             
            else:
                #drop action of r2
                #check resource status
                if r2_resource_status:
                    #check if in home position
                    if np.array_equal(r2_current_pos, self.home):
                        # print "dropped in home position - congrats"
                        r2_resource_status = 0
                        cost -= self.drop_success_cost
                    else:
                        # print "dropped at wrong spot"
                        r2_resource_status = 0
                        #add resource to new spot
                        r_curr = np.append(self.r,[r2_current_pos],axis=0)
                        cost += self.drop_wrong_spot_cost
                else:
                    # print "dropped nothing"
                    cost += self.drop_nothing_cost

        #update current resource map
        self.r = r_curr

        # print "current state"
        # print r1_current_pos,r1_resource_status,r2_current_pos,r2_resource_status
        # print "action"
        # print r1_action
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
        return int(a_idx)

    def get_action(self,a_idx):
        msb = np.zeros(6)
        lsb = np.zeros(6)
        msb_idx = a_idx/6
        lsb_idx = a_idx%6
        msb[msb_idx] = 1
        lsb[lsb_idx] = 1
        a = np.concatenate((msb, lsb), axis=None)
        return a

    def simulate(self,x0,P):
        horizon = self.horizon
        x=np.empty([6, horizon+1])
        x[:,0] = x0
        u = np.empty([horizon])
        total_cost = 0
        for i in range(horizon):
            u[i] = P[self.get_state_idx(x[:,i])]
            x[:,i+1],cost = self.get_next_state_and_cost(x[:,i], self.get_action(int(u[i])))
            total_cost += cost
        return x, u, total_cost

    def init_animate(self):
        self.r_sim = self.r_perm
        print "sim restarted"

    def animate(self,i):
        #offset
        self.ax.clear()

        xo,yo = 0.5,0.5
        borders_x = [0,0,7,7,0]
        borders_y = [0,7,7,0,0]     
        r =  self.r_sim.copy()


        s = self.sim_result[:,i]
        
        robot_x_empty = []
        robot_y_empty = []
        robot_x_full = []
        robot_y_full = []

        #robot1
        if self.s_prev[4] == 0 and s[4] == 1:
            #robot 1 picked up resource
            for i,k in enumerate(r):
                if np.array_equal(s[0:2], k):
                    r = np.delete(r,i,axis=0)
                    break
            robot_x_full.append(s[0]+ xo)
            robot_y_full.append(s[1]+ yo)
        elif self.s_prev[4] == 1 and s[4] == 0:
            #robot1 dropped resource
            r = np.append(self.r,[s[0:2]],axis=0)
            robot_x_empty.append(s[0]+ xo)
            robot_y_empty.append(s[1]+ yo)
        elif self.s_prev[4] == 1 and s[4] == 1:
            robot_x_full.append(s[0]+ xo)
            robot_y_full.append(s[1]+ yo)
        elif self.s_prev[4] == 0 and s[4] == 0:
            robot_x_empty.append(s[0]+ xo)
            robot_y_empty.append(s[1]+ yo)

        #robot2 pick
        if self.s_prev[5] == 0 and s[5] == 1:
            #robot 2 picked up resource
            for i,k in enumerate(r):
                if np.array_equal(s[2:4], k):
                    r = np.delete(r,i,axis=0)
                    break
            robot_x_full.append(s[2]+ xo)
            robot_y_full.append(s[3]+ yo)
        elif self.s_prev[5] == 1 and s[5] == 0:
            #robot2 dropped resource
            r = np.append(self.r,[s[2:4]],axis=0)
            robot_x_empty.append(s[2]+ xo)
            robot_y_empty.append(s[3]+ yo)
        elif self.s_prev[5] == 1 and s[5] == 1:
            robot_x_full.append(s[2]+ xo)
            robot_y_full.append(s[3]+ yo)
        elif self.s_prev[5] == 0 and s[5] == 0:
            robot_x_empty.append(s[2]+ xo)
            robot_y_empty.append(s[3]+ yo)



        self.s_prev = s.copy()
        self.r_sim = r.copy()

        r = r.T
        xr = r[0] + xo
        yr = r[1] + yo
        xh,yh = 3+xo,3+yo



        print s,self.get_action(int(self.sim_control[i])),i
        
        self.ax.scatter(robot_x_full,robot_y_full,color='red',marker ='o',s=10**2.5,alpha = 0.2)
        self.ax.scatter(robot_x_empty,robot_y_empty,color='green',marker ='o',s=10**2.5,alpha = 0.2)
        self.ax.scatter(xh,yh,color='blue',marker ='s',s=10**3,alpha = 0.2)
        self.ax.scatter(xr,yr,color='red')
        self.ax.plot(borders_x,borders_y)

    def iterate(self,iter):
        horizon = self.horizon
        for i in range(iter):
            Q_new = self.q_function.copy()
            J_new = self.value_function.copy()
            P_new = self.policy.copy()

            x = np.empty([6, horizon+1])
            #initial position
            x[:,0] = np.array([2,4,4,3,0,0])
            for j in range(horizon):
                xt = x[:,j]
                xt_idx = self.get_state_idx(xt)
                u_opt = np.argmin(self.q_function[xt_idx,:])

                if np.random.uniform(0, 1) < self.epsilon:
                    u_rand = int(np.random.choice(self.nu,1))
                    while u_rand == u_opt:
                        u_rand = int(np.random.choice(self.nu,1))
                    u_curr = u_rand     
                else:
                    u_curr = u_opt

                a = self.get_action(u_curr)

                #calculate temporal difference
                xt_nxt,cost = self.get_next_state_and_cost(xt,a) 
                xt_nxt_idx = self.get_state_idx(xt_nxt)
                opt_next_state_Q = np.min(self.q_function[xt_nxt_idx,:])
                current_Q = self.q_function[xt_idx,u_curr]
                TD = cost + self.gamma*opt_next_state_Q - current_Q
                #update Q
                Q_new[xt_idx,u_curr] += self.alpha*TD
                
                x[:,j+1] = xt_nxt

                J_new[xt_idx] = opt_next_state_Q
                P_new[xt_idx] = u_opt

            
            self.policy = P_new.copy()
            self.value_function = J_new.copy()

            #repop resources
            self.r = self.r_perm.copy()

            #update status
            update_progress(float(i)/ float(iter),self.epsilon)

            #THIS IS WRONG
            if ((self.q_function-Q_new)**2 < 10e-5).all():
                print("CONVERGED after iteration " + str(i))
                break
            else:
                self.q_function = Q_new.copy()

        print "learning completed"
        print "simulating env."
        #init condition
        x0 = np.array([2,4,4,3,0,0])
        p = self.policy.copy()

        X, U, total_cost = self.simulate(x0,p)
        self.s_prev = X[:,0]

        print "Total cost: " + str(total_cost)

        #show simulated results
        self.sim_result = X
        self.sim_control = U
        ani = animation.FuncAnimation(self.fig,self.animate,init_func=self.init_animate, frames=self.horizon,interval=200)
        plt.show()

def update_progress(progress,e):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    os.system( 'clear' )
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100) + " e: " + str(e)
    print(text)

def init_q_learning():
    # print "Initializing data"
    robots = 2
    grid_size = 7

    #states - robots' position, robot resource condition
    nq = (grid_size*grid_size)**robots*2**robots
    #actions - up,down,right,left,pick,drop
    nu = 6**robots 

    student = q_learning(nq,nu) 
    student.iterate(10000)

if __name__ == '__main__':
    init_q_learning()