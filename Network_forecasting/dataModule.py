# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:31:16 2023

@author: benja
"""

class data_class():
    def __init__(self,
                 all_series = [],
                 thetaParams = [],
                 BitsOrPackets = -1,
                 domain = [],
                 domainFIT = [],
                 labels = [],
                 n = -1,
                 simulated_windows = -1,
                 scope = -1,
                 Tsventana = -1):
        self.all_series = all_series
        self.thetaParams = thetaParams
        self.BitsOrPackets = BitsOrPackets
        self.domain = domain
        self.domainFIT = domainFIT
        self.labels = labels
        self.n = n
        self.simulated_windows = simulated_windows
        self.scope = scope
        self.Tsventana = Tsventana
    
    def get_theta_time_series(self,
                              week,
                              coeff_index,
                              samplingPeriod,
                              time_from,
                              time_to):
        #Function to access to certain theta series of certain week and with a sampling period and in a time range (from fime_from to time_to, both included):
        if time_to == -1:
            return self.thetaParams[time_from::samplingPeriod, week*(self.n+1)+coeff_index]
        else:
            return self.thetaParams[time_from:time_to+1:samplingPeriod, week*(self.n+1)+coeff_index]