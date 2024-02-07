# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:40:37 2023

@author: benja
"""

class GraphicalUserInterface():
    def __init__(self,
                 root = None,
                 #Data:
                 data = None,
                 timestamps = [],
                 #Widgets:
                 spinBoxWeek = None,
                 time_serie_plot = None,
                 time_serie_plot_canvas = None,
                 #Memory utilities:
                 displaying_week_was = 0,
                 displaying_theta_was = 0,
                 #Sampling dropdown menu:
                 samplingDropDownMenu = None,
                 samplePeriod = 1,
                 lastWindow = -1,
                 #Slider:
                 windowSliderSelector = None,
                 selectedWindow = 1,
                 red_lines_delimiters_plot = None,
                 window_zoom_plot = None,
                 #Spinbox for theta serie:
                 spinBoxTheta = None,
                 theta_serie_plot = None,
                 theta_serie_line_delimiter_plot = None,
                 theta_serie_plot_canvas = None,
                 #Other status data:
                 displaying_week = 0,
                 displaying_theta = 0,
                 #Plot axes objects:
                 time_serie_plot_ax = None,
                 window_plot_ax = None,
                 theta_param_plot_ax = None,
                 #Current plotting arrays:
                 current_theta_evolution = [],
                 window_test = [],
                 real_coefs=[]):
        self.root = root
        
        self.data = data
        self.timestamps = timestamps
        
        self.spinBoxWeek = spinBoxWeek
        self.displaying_week_was = displaying_week_was
        self.displaying_theta_was = displaying_theta_was
        
        self.time_serie_plot = time_serie_plot
        self.time_serie_plot_canvas = time_serie_plot_canvas
        self.time_serie_plot_ax = time_serie_plot_ax
        self.window_plot_ax = window_plot_ax
        
        self.samplingDropDownMenu = samplingDropDownMenu
        self.samplePeriod = samplePeriod
        self.windowSliderSelector = windowSliderSelector
        self.selectedWindow = selectedWindow
        self.red_lines_delimiters_plot = red_lines_delimiters_plot
        self.window_zoom_plot = window_zoom_plot
        self.lastWindow = lastWindow
        
        self.spinBoxTheta = spinBoxTheta
        
        self.theta_serie_plot = theta_serie_plot
        self.theta_serie_plot_canvas = theta_serie_plot_canvas
        self.theta_serie_line_delimiter_plot = theta_serie_line_delimiter_plot
        
        self.displaying_week = displaying_week
        self.displaying_theta = displaying_theta
        self.theta_param_plot_ax = theta_param_plot_ax
        self.current_theta_evolution = current_theta_evolution
        self.window_test = window_test
        
class GraphicalUserInterface_LSTM_configuration():
    def __init__(self,
                 root = None,
                 recurrent_forecast=False,
                 future_timesteps_recurrent=1,
                 normalization="min-max",
                 CNN = False,
                 selectedFirstWindow=-1,
                 selectedTimeStepsPast = 1,
                 selectedTimeStepsSubsequence = 1,
                 selectedKernelSize = 1,
                 epochs = 10,
                 LSTMunits = 70,
                 previous_weeks = 0,
                 #Plot widgets:
                 blue_lines_plot= None,
                 theta_serie_blue_line_plot = None,
                 #Plot axes objects:
                 time_serie_plot_ax = None,
                 theta_serie_plot_ax = None,
                 #Plot canvas:
                 time_serie_plot_canvas = None,
                 #Widgets:
                 spinBoxTimeStepsPast = None,
                 dropdownChooseSubseqsPastSteps = None,
                 spinBoxKernelSize = None,
                 spinboxChooseLSTMunits = None,
                 spinboxChooseLSTMepochs= None,
                 spinboxChoosePreviousWeeks = None
                 ):
        self.root = root
        self.recurrent_forecast = recurrent_forecast
        self.future_timesteps_recurrent = future_timesteps_recurrent
        self.normalization = normalization
        self.CNN = CNN
        self.selectedFirstWindow = selectedFirstWindow
        self.selectedTimeStepsPast = selectedTimeStepsPast
        self.epochs = epochs
        self.LSTMunits = LSTMunits
        self.previous_weeks = previous_weeks
        self.selectedKernelSize = selectedKernelSize
        
        self.blue_lines_plot = blue_lines_plot
        self.time_serie_plot_ax = time_serie_plot_ax
        self.time_serie_plot_canvas = time_serie_plot_canvas
        
        self.theta_serie_blue_line_plot = theta_serie_blue_line_plot
        self.theta_serie_plot_ax = theta_serie_plot_ax
        
        self.spinBoxTimeStepsPast = spinBoxTimeStepsPast
        self.dropdownChooseSubseqsPastSteps = dropdownChooseSubseqsPastSteps
        self.selectedTimeStepsSubsequence = selectedTimeStepsSubsequence
        self.spinBoxKernelSize = spinBoxKernelSize
        self.spinboxChooseLSTMunits = spinboxChooseLSTMunits
        self.spinboxChooseLSTMepochs = spinboxChooseLSTMepochs
        self.spinboxChoosePreviousWeeks = spinboxChoosePreviousWeeks
        
    def switchRecurrent_forecast(self):
        self.recurrent_forecast = not self.recurrent_forecast
    
    def switchCNN_layer(self):
        self.CNN = not self.CNN
            
class GUI_Result():
    def __init__(self,
                 root=None,
                 spinboxChooseThetaSerie=None,
                 displaying_theta=0,
                 displaying_theta_was=0,
                 theta_serie_plot_truth = None,
                 theta_serie_plot_predicted = None,
                 ax_result_plot=None,
                 theta_serie_plot_canvas=None
                 ):
        self.root = root
        self.spinboxChooseThetaSerie = spinboxChooseThetaSerie
        self.displaying_theta = displaying_theta
        self.displaying_theta_was = displaying_theta_was
        self.theta_serie_plot_truth = theta_serie_plot_truth
        self.theta_serie_plot_predicted = theta_serie_plot_predicted
        self.ax_result_plot = ax_result_plot
        self.theta_serie_plot_canvas = theta_serie_plot_canvas
        