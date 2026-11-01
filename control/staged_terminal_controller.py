"""Staged revision: align final approach before entering the terminal disk.

Requires a verified free manoeuvring area, including 0.12 m behind target.
Original direct hybrid and all baseline evidence remain separate.
"""
import numpy as np
from control.lqr_controller import wrap
from control.terminal_controller import TerminalController,TerminalParameters,TrajectoryTerminalController


class StagedTerminalController:
    def __init__(self,goal,parameters=TerminalParameters(),stage_distance=.12,alignment_threshold=.35):
        if not np.isfinite(stage_distance) or stage_distance<=parameters.position_exit:
            raise ValueError('Stage distance must exceed position exit threshold')
        if not 0<alignment_threshold<np.pi/2:raise ValueError('Invalid alignment threshold')
        self.goal=np.asarray(goal,dtype=float).copy()
        self.final=TerminalController(goal,parameters)
        self.p=parameters;self.distance=stage_distance;self.threshold=alignment_threshold
        self.stage=None;self.started=False;self.mode='initial';self.elapsed=0.

    def command(self,estimated_pose):
        pose=np.asarray(estimated_pose,dtype=float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():
            self.mode='fault';return np.zeros(2),self.mode
        if self.mode in ('stop','fault'):return np.zeros(2),self.mode
        if self.elapsed>=self.p.timeout:
            self.mode='fault';return np.zeros(2),self.mode
        self.elapsed+=self.p.sample_time
        if not self.started:
            delta=self.goal[:2]-pose[:2]
            if np.linalg.norm(delta)>self.p.position_exit and abs(wrap(np.arctan2(delta[1],delta[0])-self.goal[2]))>self.threshold:
                point=self.goal.copy();point[:2]-=self.distance*np.array([np.cos(point[2]),np.sin(point[2])])
                self.stage=TerminalController(point,self.p)
            self.started=True
        if self.stage is not None:
            raw,mode=self.stage.command(pose)
            if mode=='fault':self.mode='fault';return np.zeros(2),self.mode
            if mode not in ('heading','coast','stop'):
                self.mode='stage_'+mode;return raw,self.mode
            # Stage is position-only: do not rotate to final heading at this point.
            self.stage=None
        raw,self.mode=self.final.command(pose)
        return raw,self.mode


class StagedTrajectoryTerminalController(TrajectoryTerminalController):
    def __init__(self,goal,end_time,parameters=TerminalParameters()):
        super().__init__(goal,end_time,parameters)
        self.terminal=StagedTerminalController(goal,parameters)
