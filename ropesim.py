import os, sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
import time
import math

from numpy import cross, dot
from numpy import pi
from numpy import cos, sin, tan,sqrt,degrees,radians
from numpy.linalg import norm

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import datetime

plt.close('all')

numlinks = 10
length = .03 #m
mass = .3 #kg
g = 9.8 #m/s**2
b = 10000


modulus = 210e9 #Pa, steel wire
area = pi*(.01/2)**2

k = modulus*area/length #Newtons/meter
k = 100000

state = []
for i in range(numlinks):
    state.append(np.array([i*length,0,0,0]))
state = np.hstack(state)

def propagate(t,state,k,length,mass,g):
    
    dlinks = []
    Fprev = 0
    for index in range(int(len(state)/4)):
        link = state[index*4:index*4+4]
        
        
        if index == 0:
            nextlink = state[(index+1)*4:(index+1)*4+4]
            d = norm(nextlink[0:2] - link[0:2])
            n = (nextlink[0:2] - link[0:2])/d
            v = dot(nextlink[2:4] - link[2:4],n)
            Fnext = -(k*(length-d) - b*v)*n
            dlink = np.array([0,0,0,0])
            dlinks.append(dlink)
            Fprev = -Fnext
        elif index == int(len(state)/4 -1):
            accel = Fprev/mass + np.array([0,-g])
            dlink = np.hstack([link[2:4],accel])
            dlinks.append(dlink)
        else:
            nextlink = state[(index+1)*4:(index+1)*4+4]
            d = norm(nextlink[0:2] - link[0:2])
            n = (nextlink[0:2] - link[0:2])/d
            v = dot(nextlink[2:4] - link[2:4],n)
            Fnext = -(k*(length-d) - b*v)*n
            accel = Fnext/mass + Fprev/mass + np.array([0,-g])
            dlink = np.hstack([link[2:4],accel])
            dlinks.append(dlink)
            Fprev = -Fnext
            
    dlinks = np.hstack(dlinks)
    
    return dlinks

solver = ode(propagate)
solver.set_integrator('lsoda',atol = 10**-4)
solver.set_initial_value(state,0)
solver.set_f_params(k,length,mass,g)

tf = 20
dt = 1/60

newstate = []
t = []
#t0 = time.clock()
while solver.successful() and solver.t < tf:
    solver.integrate(solver.t+dt)
    newstate.append(solver.y)
    t.append(solver.t)
#t1 = time.clock()
#print('Integration time: '+str(t1-t0)+' s')

newstate = np.vstack(newstate)
t = np.vstack(t)

jointsx = []
jointsy = []
for i in range(numlinks):
    jointsx.append(newstate[:,i*4])
    jointsy.append(newstate[:,i*4+1])
jointsx = np.vstack(jointsx)
jointsy = np.vstack(jointsy)

def update_line(num, jointsx,jointsy, line):
    line.set_data(jointsx[:,num], jointsy[:,num])
    return line,

fig1 = plt.figure()
plt.axis('square')
plt.xlim(-.5,.5)
plt.ylim(-.5,.5)
l, = plt.plot([], [], 'r-')
line_ani = FuncAnimation(fig1, update_line, len(newstate), fargs=(jointsx,jointsy, l),
                                   interval=60, blit=True)
plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60,bitrate=-1)
line_ani.save('Rope.mp4',writer = writer)
            
        
            



    