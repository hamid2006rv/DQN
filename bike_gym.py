
# coding: utf-8

# In[1]:


import gym
from gym import error, spaces, utils
from gym.utils import seeding


# In[2]:


import os
import pybullet as p
import pybullet_data


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


import math
import random


# In[5]:


class BikeEnv(gym.Env):
    def __init__(self):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1,0,-1,0,-1,0]),np.array([1,1,1,1,1,1]))#[a,a,b,b,c,c]
        self.observation_space = spaces.Box(np.array([-1000, -1000 , 0]), np.array([1000 , 1000 , 10]))
        
    def step(self, action):
        #if action[0] >= 5 : action[0]=0 
            
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        #p.setJointMotorControl2(self.pid, 0, p.VELOCITY_CONTROL, targetVelocity=action[0], force=action[1])
        p.setJointMotorControl2(self.pid, 1, p.VELOCITY_CONTROL, targetVelocity=action[2], force=action[3])
        p.setJointMotorControl2(self.pid, 2, p.VELOCITY_CONTROL, targetVelocity=action[4], force=action[5])
        p.stepSimulation()
        
        state = p.getLinkState(self.pid,0)[0]
        if state[2] <= 0.5 :
            reward = -100
            done = True
        else :
            reward = math.sqrt((self.origin[0]-state[0])**2+(self.origin[1]-state[1])**2)
            done = False
        #self.origin = state    
        observation = p.getLinkState(self.pid,0)[0] #+ p.getJointState(self.pid,0)[:2] + p.getJointState(self.pid,1)[:2] + p.getJointState(self.pid,2)[:2]
        state_object, _ = p.getBasePositionAndOrientation(self.pid)
        info = {'x':state_object[0],'y':state_object[1],'z':state_object[2]}
        print(action)
        return observation, reward, done, info
            
        
    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-10)
        urdfRootPath = pybullet_data.getDataPath()
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,0])
        self.pid = p.loadURDF(os.path.join(urdfRootPath, "bicycle/bike.urdf"),basePosition=[0,0,1])
        self.origin = p.getLinkState(self.pid,0)[0]
        
        observation = p.getLinkState(self.pid,0)[0] #+ p.getJointState(self.pid,0)[:2] + p.getJointState(self.pid,1)[:2] + p.getJointState(self.pid,2)[:2]
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return observation
        
    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()


# In[6]:


#env = BikeEnv()
#for i_episode in range(20):
#    observation = env.reset()
#    for t in range(1000):
#        env.render()
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step(action)
#        print(info)
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
#env.close()

