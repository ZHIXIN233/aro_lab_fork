#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""
import time

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement, collision

# Following package are added
from scipy.optimize import minimize

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    def cost(q):
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        oMcubeLhook = getcubeplacement(cube, LEFT_HOOK) #placement of the left hand hook
        oMcubeRhook = getcubeplacement(cube, RIGHT_HOOK) #placement of the right hand hook

        lefthand_id = robot.model.getFrameId(LEFT_HAND)
        righthand_id = robot.model.getFrameId(RIGHT_HAND)

        oML = robot.data.oMf[lefthand_id]
        oMR = robot.data.oMf[righthand_id]

        LMcubeLhook = oML.inverse() * oMcubeLhook
        RMcubeRhook = oMR.inverse() * oMcubeRhook

        sum_dist = norm(RMcubeRhook.translation) \
                   + norm(LMcubeLhook.translation)
        sum_rotation = norm(RMcubeRhook.rotation-np.identity(3)) \
                       + norm(LMcubeLhook.rotation-np.identity(3))

        # add posture bias

        robot_facing = robot.placement(q, 1).rotation[:2, :2]

        cube2Dpos = cubetarget.translation[:2]
        s = cube2Dpos[1] / norm(cube2Dpos)
        c = cube2Dpos[0] / norm(cube2Dpos)

        cube_direction = np.asarray([[c, -s],
                                     [s, c]])

        posture_bias = norm(robot_facing-cube_direction) * 0.2

        return sum_rotation+sum_dist+posture_bias

    def collision_constraint(q):
        # If collision returns False (no collision), return a positive number (valid state)
        # If collision returns True (collision), return a negative number (invalid state)
        return -1 if collision(robot, q) else 1

    def callback(q):
        viz.display(q)
        time.sleep(0.1)

    constraint_dict = {
        'type': 'ineq',  # Inequality constraint: collision_constraint(q) >= 0
        'fun': collision_constraint
    }

    q_opt = minimize(cost, qcurrent, callback=callback).x

    if collision(robot, q_opt):
        q_opt = minimize(cost, qcurrent, constraints=constraint_dict, callback=callback).x

    try:
        updatevisuals(viz, robot, cube, q_opt)
    except:
        pass



    success = True if cost(q_opt) < 1e-3 else False




    return q_opt, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
