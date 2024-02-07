"""
@author: Benjamín Martín Gómez
"""
import sys

import tkinter as tk
from tkinter import ttk #Para más opciones de los elementos de la interfaz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import math
import json
import dataModule
from GUIStyleModule import GraphicalUserInterface, GraphicalUserInterface_LSTM_configuration, GUI_Result
import LSTMprediction as predictorPackage
from GeneralPurposeFunctions import get_divisors, contains_nan, interpolate, map_None_to_nan, take_not_nan_values, all_nan
matplotlib.use('Agg')

def getJSONTrendDynamicsData(filenameJSON):
    #Function to get JSON data from JSON filename:
    f = open(filenameJSON)
    dataJSON = json.load(f)
    return dataJSON

def getAllSeriesMatrix(JSONobject):
    #Function to get the matrix with the time series data from JSON object:
    key_list = list(JSONobject.keys())[8:-2]
    all_series = np.zeros((len(key_list), len(JSONobject[key_list[0]])))
    for i in range(len(key_list)):
        time_serie = JSONobject[key_list[i]]
        all_series[i] = time_serie
    return all_series

def getThetaParamsMatrix(JSONobject):
    #Function to get the matrix with theta params of sliding windows from JSON object:
    n = JSONobject["n"]
    Tsventana = JSONobject["Tsventana"]
    thetaParams = np.array(JSONobject["TP"+str(int(Tsventana/60))+'_'+str(n)])
    return thetaParams
    


#Plot updates:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def update_week_network_traffic_time_serie_plot(GUIobject):
    time_serie = GUIobject.data.all_series[GUIobject.displaying_week]
    GUIobject.time_serie_plot[0].set_data([], [])
    GUIobject.time_serie_plot = GUIobject.time_serie_plot_ax.plot(time_serie, linewidth=0.1, color='blue')
    real_coefs = []
    for coef in range(GUIobject.data.n+1):
        real_coefs.append(GUIobject.data.get_theta_time_series(week=GUIobject.displaying_week,
                                                               coeff_index=coef,
                                                               samplingPeriod=1,
                                                               time_from=GUIobject.selectedWindow,
                                                               time_to=GUIobject.selectedWindow))
    GUIobject.real_coefs = real_coefs
    GUIobject.time_serie_plot_canvas.draw()
    pass
def update_week_network_traffic_time_serie_plot_redlines(GUIobject):
    #time_serie = GUIobject.data.all_series[GUIobject.displaying_week]
    x = [GUIobject.selectedWindow, GUIobject.selectedWindow+GUIobject.data.Tsventana-1, GUIobject.selectedWindow+GUIobject.data.Tsventana-1, GUIobject.selectedWindow, GUIobject.selectedWindow]
    x_axis = GUIobject.timestamps[x[0]:x[1]+1]
    #y = [0, max(take_not_nan_values(time_serie))]
    #To optimize the time response of slider widget is better to set a const value:
    y = [0, 0, 3000000, 3000000, 0]
    GUIobject.red_lines_delimiters_plot[0].set_data([], [])
    GUIobject.red_lines_delimiters_plot = GUIobject.time_serie_plot_ax.plot(x, y, color='red')
    #Configure x_ticks of window zoom plot:
    GUIobject.window_plot_ax.set_xticks([0, len(x_axis)]); GUIobject.window_plot_ax.set_xticklabels([x_axis[0], x_axis[-1]], rotation=10)
    GUIobject.time_serie_plot_canvas.draw()
    pass

def update_windowzoom_network_traffic_time_serie_plot(GUIobject):
    time_serie = GUIobject.data.all_series[GUIobject.displaying_week]
    window = list(time_serie[GUIobject.selectedWindow:GUIobject.selectedWindow+GUIobject.data.Tsventana])
    GUIobject.window_test = window
    #domain = list(range(GUIobject.selectedWindow,GUIobject.selectedWindow+GUIobject.data.Tsventana))
    GUIobject.window_zoom_plot[0].set_data([], [])
    
    GUIobject.window_zoom_plot = GUIobject.window_plot_ax.plot(window, linewidth=0.5, color='blue')
    #GUIobject.window_plot_ax.set_xlim(domain[0], domain[-1])
    GUIobject.window_plot_ax.grid(True, alpha=0.5)
    GUIobject.time_serie_plot_canvas.draw()
    pass

def update_theta_param_time_serie_plot(GUIobject):
    theta_evolution = GUIobject.data.get_theta_time_series(week=GUIobject.displaying_week, coeff_index=GUIobject.displaying_theta, samplingPeriod=GUIobject.samplePeriod, time_from=0, time_to=-1)
    theta_evolution = interpolate(theta_evolution, GUIobject.samplePeriod)
    GUIobject.theta_serie_plot[0].set_data([], [])
    GUIobject.theta_serie_plot = GUIobject.theta_param_plot_ax.plot(theta_evolution, linewidth=1, color='orange')
    if bool(take_not_nan_values(theta_evolution)):
        y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
        GUIobject.theta_param_plot_ax.set_ylim(y[0], y[1])
        #GUIobject.theta_param_plot_ax.set_xlim(GUIobject.theta_param_plot_ax.set_xlim(take_not_nan_indexes(GUIobject.data.thetaParams[::1, GUIobject.displaying_week*(GUIobject.data.n+1)+GUIobject.displaying_theta])[0], take_not_nan_indexes(GUIobject.data.thetaParams[::1, GUIobject.displaying_week*(GUIobject.data.n+1)+GUIobject.displaying_theta])[1]))
    else:
        y = [0, 1]
    x = [GUIobject.selectedWindow, GUIobject.selectedWindow]
    GUIobject.theta_serie_line_delimiter_plot[0].set_data([], [])
    GUIobject.theta_serie_line_delimiter_plot = GUIobject.theta_param_plot_ax.plot(x, y, color='red')
    GUIobject.theta_serie_plot_canvas.draw()
    GUIobject.current_theta_evolution = theta_evolution
    pass
def update_theta_param_time_serie_plot_redline(GUIobject):
    #Get the current theta serie to set the limits of the yline:
    '''
    theta_evolution = GUIobject.data.thetaParams[::GUIobject.samplePeriod, GUIobject.displaying_week*(GUIobject.data.n+1)+GUIobject.displaying_theta]
    theta_evolution_sample1 = GUIobject.data.thetaParams[::1, GUIobject.displaying_week*(GUIobject.data.n+1)+GUIobject.displaying_theta]
    theta_evolution = interpolate(theta_evolution, GUIobject.samplePeriod)
    if bool(take_not_nan_values(theta_evolution)):
        y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
        GUIobject.theta_param_plot_ax.set_ylim(y[0], y[1])
        GUIobject.theta_param_plot_ax.set_xlim(take_not_nan_indexes(theta_evolution_sample1)[0], take_not_nan_indexes(theta_evolution_sample1)[1])
    else:
        y = [0, 1]
    '''
    #To optimize the time response of slider widget is better to set a const value:
    y = [-100000, 100000]
    x = [GUIobject.selectedWindow, GUIobject.selectedWindow]
    GUIobject.theta_serie_line_delimiter_plot[0].set_data([], [])
    GUIobject.theta_serie_line_delimiter_plot = GUIobject.theta_param_plot_ax.plot(x, y, color='red')
    #Update real coefficients:
    real_coefs = []
    for coef in range(GUIobject.data.n+1):
        real_coefs.append(GUIobject.data.get_theta_time_series(week=GUIobject.displaying_week,
                                                               coeff_index=coef,
                                                               samplingPeriod=1,
                                                               time_from=GUIobject.selectedWindow,
                                                               time_to=GUIobject.selectedWindow))
    GUIobject.real_coefs = real_coefs
    GUIobject.theta_serie_plot_canvas.draw()
    
                                                            
    pass
def update_week_network_traffic_time_serie_plot_bluelines(GUIobject, GUI_LSTM_configuration):
    #time_serie = GUIobject.data.all_series[GUIobject.displaying_week]
    x = [GUI_LSTM_configuration.selectedFirstWindow, GUI_LSTM_configuration.selectedFirstWindow+GUIobject.data.Tsventana-1, GUI_LSTM_configuration.selectedFirstWindow+GUIobject.data.Tsventana-1, GUI_LSTM_configuration.selectedFirstWindow, GUI_LSTM_configuration.selectedFirstWindow]
    #y = [0, max(take_not_nan_values(time_serie))]
    #To optimize the time response of slider widget is better to set a const value:
    y = [0, 0, 150000, 150000, 0]
    GUI_LSTM_configuration.blue_lines_plot[0].set_data([], [])
    GUI_LSTM_configuration.blue_lines_plot = GUI_LSTM_configuration.time_serie_plot_ax.plot(x, y, color='blue')
    GUI_LSTM_configuration.time_serie_plot_canvas.draw()
    pass
def update_theta_param_time_serie_plot_blueline(GUI_LSTM_configuration):
    y = [-100000, 100000]
    x = [GUI_LSTM_configuration.selectedFirstWindow, GUI_LSTM_configuration.selectedFirstWindow]
    GUI_LSTM_configuration.theta_serie_blue_line_plot[0].set_data([], [])
    GUI_LSTM_configuration.theta_serie_blue_line_plot = GUI_LSTM_configuration.theta_serie_plot_ax.plot(x, y, color='blue')
    GUI_LSTM_configuration.time_serie_plot_canvas.draw()
    pass
def update_theta_param_time_serie_plot_prediction(GUI_Result_object, GUIobject, final_prediction):
    GUI_Result_object.theta_serie_plot_truth[0].set_data([], [])
    GUI_Result_object.theta_serie_plot_predicted[0].set_data([], [])
    
    theta_evolution_real = GUIobject.data.get_theta_time_series(week=GUIobject.displaying_week,
                                                                coeff_index=GUI_Result_object.displaying_theta,
                                                                samplingPeriod=GUIobject.samplePeriod,
                                                                time_from=GUIobject.lastWindow - GUIobject.data.scope*10,
                                                                time_to=GUIobject.lastWindow + GUIobject.data.scope)
    theta_evolution_prediction = GUIobject.data.get_theta_time_series(week=GUIobject.displaying_week,
                                                                      coeff_index=GUI_Result_object.displaying_theta,
                                                                      samplingPeriod=GUIobject.samplePeriod,
                                                                      time_from=GUIobject.lastWindow - GUIobject.data.scope*10,
                                                                      time_to=GUIobject.lastWindow)
    theta_evolution_prediction = np.concatenate((theta_evolution_prediction, final_prediction[GUI_Result_object.displaying_theta]))
    
    theta_evolution_real = interpolate(theta_evolution_real, GUIobject.samplePeriod)
    theta_evolution_prediction = interpolate(theta_evolution_prediction, GUIobject.samplePeriod)
    if bool(take_not_nan_values(theta_evolution_prediction)):
        y = [0.999*min(take_not_nan_values(theta_evolution_prediction)), 1.001*max(take_not_nan_values(theta_evolution_prediction))]
        GUI_Result_object.ax_result_plot.set_ylim(y[0], y[1])
        
        x_axis_vector = np.linspace(GUIobject.lastWindow - GUIobject.data.scope*10, GUIobject.lastWindow + GUIobject.data.scope+1, len(theta_evolution_real))
        #GUI_Result_object.ax_result_plot.set_xlim(x_axis_vector[0], x_axis_vector[1])
    else:
        y = [0, 1]
    GUI_Result_object.theta_serie_plot_predicted = GUI_Result_object.ax_result_plot.plot(theta_evolution_prediction, linewidth=1, color='red')
    GUI_Result_object.theta_serie_plot_truth = GUI_Result_object.ax_result_plot.plot(theta_evolution_real, linewidth=1, color='green')
    
    GUI_Result_object.theta_serie_plot_canvas.draw()
    pass
def runPrediction(GUIobject, GUILSTMobject):
    final_prediction = predictorPackage.LSTM_network_forecasting_algorithm(GUIobject, GUILSTMobject)
    createPredictionResultGUI(final_prediction, GUIobject)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_polynomial(domain_array, coefficients_list):
    result = np.zeros(domain_array.shape)
    for coef_i, coef_value in enumerate(coefficients_list):
        result = result + coef_value*pow(domain_array, coef_i)
    return result
def get_predicted_coefficients_from_dictionary(predicted_coefs_dict):
    result = []
    for coef,predicted_coefs_list in predicted_coefs_dict.items():
        result.append(predicted_coefs_list[-1])
    return list(result)
    
def createPredictionResultGUI(final_prediction, GUIobject):
    #Function to initialize GUI:
    rootPredictionResult = tk.Tk()
    rootPredictionResult.title("Prediction result")
    rootPredictionResult.geometry("1080x900")
    #Spinbox to change theta plot:
    def on_change_THETAINDEXSPINBOX():
        GUI_Result_object.displaying_theta = int(GUI_Result_object.spinboxChooseThetaSerie.get())
        if GUI_Result_object.displaying_theta != GUI_Result_object.displaying_theta_was:
            #Update theta param time serie plot:
            update_theta_param_time_serie_plot_prediction(GUI_Result_object, GUIobject, final_prediction)
            GUI_Result_object.displaying_theta_was = GUI_Result_object.displaying_theta
    labelChooseThetaSerie = tk.Label(rootPredictionResult, text="Choose theta parameter time serie:")
    labelChooseThetaSerie.pack()
    spinboxChooseThetaSerie = ttk.Spinbox(rootPredictionResult, from_=0, to=GUIobject.data.n, increment=1, command=on_change_THETAINDEXSPINBOX)
    spinboxChooseThetaSerie.set(0)
    spinboxChooseThetaSerie.pack()
    
    #Plot de la serie temporal theta:
    fig_prediction_result, ax_result_plot = plt.subplots()
    theta_evolution_real = GUIobject.data.get_theta_time_series(week=GUIobject.displaying_week,
                                                                coeff_index=0,
                                                                samplingPeriod=GUIobject.samplePeriod,
                                                                time_from=GUIobject.lastWindow - GUIobject.data.scope*10,
                                                                time_to=GUIobject.lastWindow + GUIobject.data.scope)
    theta_evolution_prediction = GUIobject.data.get_theta_time_series(week=GUIobject.displaying_week,
                                                                      coeff_index=0,
                                                                      samplingPeriod=GUIobject.samplePeriod,
                                                                      time_from=GUIobject.lastWindow - GUIobject.data.scope*10,
                                                                      time_to=GUIobject.lastWindow)
    theta_evolution_prediction = np.concatenate((theta_evolution_real, final_prediction[0]))
    theta_evolution_real = interpolate(theta_evolution_real, GUIobject.samplePeriod)
    theta_evolution_prediction = interpolate(theta_evolution_prediction, GUIobject.samplePeriod)
    theta_serie_plot_predicted = ax_result_plot.plot(theta_evolution_prediction, linewidth=1, color='red')
    theta_serie_plot_truth = ax_result_plot.plot(theta_evolution_real, linewidth=1, color='green')
    theta_serie_plot_canvas = FigureCanvasTkAgg(fig_prediction_result, master=rootPredictionResult)
    theta_serie_plot_canvas.get_tk_widget().pack()
    
    fig_window_polynomial_result, ax_window_result_plot = plt.subplots()
    #Get truth polynomial:
    pol_truth = get_polynomial(np.array(GUIobject.data.domainFIT), list(GUIobject.real_coefs))
    #Get predicted polynomial:
    predicted_coeffs = get_predicted_coefficients_from_dictionary(final_prediction)
    pol_predicted = get_polynomial(np.array(GUIobject.data.domainFIT), predicted_coeffs)
    window_x_domain = list(range(GUIobject.selectedWindow, GUIobject.selectedWindow+GUIobject.data.Tsventana))
    window_plot = ax_window_result_plot.plot(window_x_domain, GUIobject.window_test, linewidth=0.1, color='blue')
    pol_truth_plot = ax_window_result_plot.plot(window_x_domain, pol_truth, linewidth=1, color='green')
    pol_predicted_plot = ax_window_result_plot.plot(window_x_domain, pol_predicted, linewidth=1, color='red')
    
    uncertainty_limit_x = [window_x_domain[-1] - GUIobject.data.scope, window_x_domain[-1] - GUIobject.data.scope]
    uncertainty_limit_y = [0, 1.2*max(GUIobject.window_test)]
    ax_window_result_plot.plot(uncertainty_limit_x, uncertainty_limit_y, color='black', linestyle='--')
    ax_window_result_plot.autoscale(enable=True, axis='x', tight=True)
    ax_window_result_plot.set_ylim(0.8*min(GUIobject.window_test), 1.2*max(GUIobject.window_test))
    ax_window_result_plot.grid(True, alpha=0.5)
    pol_truth_plot_plot_canvas = FigureCanvasTkAgg(fig_window_polynomial_result, master=rootPredictionResult)
    pol_truth_plot_plot_canvas.get_tk_widget().pack()
    
    GUI_Result_object = GUI_Result(root=rootPredictionResult,
                                   spinboxChooseThetaSerie=spinboxChooseThetaSerie,
                                   displaying_theta=0,
                                   displaying_theta_was=0,
                                   theta_serie_plot_predicted=theta_serie_plot_predicted,
                                   theta_serie_plot_truth=theta_serie_plot_truth,
                                   ax_result_plot=ax_result_plot,
                                   theta_serie_plot_canvas=theta_serie_plot_canvas)
                            
    rootPredictionResult.mainloop()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def createDefaultLSTMconfigurationDialogueGUI(GUIobject):
    #Function to initialize GUI:
    rootLSTM = tk.Tk()
    rootLSTM.title("LSTM configuration")
    rootLSTM.geometry("1080x900")
    
    #CheckBox: checkbox to select recurrent or not recurrent forecast
    def on_change_RECURRENTFORECAST():
        GUI_LSTM_configuration.switchRecurrent_forecast()
    var = tk.BooleanVar()
    checkbox = tk.Checkbutton(rootLSTM, text="Recurrent forecast", variable=var, command=on_change_RECURRENTFORECAST)
    checkbox.pack()
    
    #Recurrent timesteps future (in case of using recurrent forecast)
    def on_change_FUTURETIMESTEPSRECURRENTFORECAST(chosen_recurrent_timesteps):
        GUI_LSTM_configuration.future_timesteps_recurrent=chosen_recurrent_timesteps
    labelChooseRecurrentTimesteps = tk.Label(rootLSTM, text="Future timesteps of recurrent forecast (only for recurrent forecast):")
    labelChooseRecurrentTimesteps.pack()
    selected_option = tk.StringVar(rootLSTM)
    selected_option.set("1")
    options = get_divisors(int(GUIobject.data.scope/GUIobject.samplePeriod))
    labelChooseRecurrentTimesteps = tk.OptionMenu(rootLSTM, selected_option, *options, command=on_change_FUTURETIMESTEPSRECURRENTFORECAST)
    labelChooseRecurrentTimesteps.pack()
    
    #Dropdown Menu to choose the normalization type:
    def on_change_NORMALIZATIONDROPDOWN(chosen_normalization):
        GUI_LSTM_configuration.normalization=chosen_normalization
    labelChooseNormalization = tk.Label(rootLSTM, text="Normalization:")
    labelChooseNormalization.pack()
    selected_option = tk.StringVar(rootLSTM)
    selected_option.set("min-max")
    options = ["min-max", "tanh", "z-score"]
    dropdownChooseSampling = tk.OptionMenu(rootLSTM, selected_option, *options, command=on_change_NORMALIZATIONDROPDOWN)
    dropdownChooseSampling.pack()
    
    #Plot of time serie: from initial second to the last test window:
    fig, (time_serie_plot_ax, theta_serie_plot_ax) = plt.subplots(1, 2, figsize=(6, 3))
    time_serie = GUIobject.data.all_series[GUIobject.displaying_week]
    window = list(time_serie[0:GUIobject.selectedWindow + GUIobject.data.Tsventana])
    time_serie_plot = time_serie_plot_ax.plot(window, linewidth=0.1, color='blue')
        #Plot de los límites de la ventana:
    x = [GUIobject.selectedWindow-GUIobject.data.scope, GUIobject.selectedWindow+GUIobject.data.Tsventana-1-GUIobject.data.scope, GUIobject.selectedWindow+GUIobject.data.Tsventana-1-GUIobject.data.scope, GUIobject.selectedWindow-GUIobject.data.scope, GUIobject.selectedWindow-GUIobject.data.scope]; y = [0, 0, max(take_not_nan_values(window)), max(take_not_nan_values(window)), 0]
    blue_lines_plot = time_serie_plot_ax.plot(x, y, color='blue')
    time_serie_plot_ax.plot(x, y, color='red')
    time_serie_plot_ax.set_xlim(1, len(window)); time_serie_plot_ax.set_ylim(1, max(take_not_nan_values(window))); time_serie_plot_ax.grid(True, alpha=0.5)
        #Plot de la serie temporal theta:
    theta_evolution = GUIobject.current_theta_evolution
    theta_serie_plot = theta_serie_plot_ax.plot(theta_evolution[0:GUIobject.selectedWindow+1], linewidth=1, color='orange')
    if bool(take_not_nan_values(theta_evolution)):
        y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
        theta_serie_plot_ax.set_ylim(y[0], y[1])
    else:
        y = [0, 1]
    x = [GUIobject.selectedWindow, GUIobject.selectedWindow]
    #theta_serie_plot_ax.set_xlim(take_not_nan_indexes(GUIobject.data.thetaParams[::1, GUIobject.displaying_week*(GUIobject.data.n+1)+GUIobject.displaying_theta])[0], GUIobject.selectedWindow)
    theta_serie_plot_ax.plot(x, y, color='red')
    x = [GUIobject.selectedWindow-GUIobject.data.scope, GUIobject.selectedWindow-GUIobject.data.scope]
    theta_serie_blue_line_plot = theta_serie_plot_ax.plot(x, y, color='blue')
    #Canvas del plot:
    canvas_Traffic = FigureCanvasTkAgg(fig, master=rootLSTM)
    canvas_Traffic.get_tk_widget().pack()
    
    #Slider to select the train duration: with the selector, user can choose the first window from which the LSTM will take the theta params for training:
    def on_change_WINDOWSLIDERTTRAIN(selectedWindow):
        GUI_LSTM_configuration.selectedFirstWindow = int(selectedWindow)
        update_week_network_traffic_time_serie_plot_bluelines(GUIobject, GUI_LSTM_configuration)
        update_theta_param_time_serie_plot_blueline(GUI_LSTM_configuration)
    labelChooseWindow = tk.Label(rootLSTM, text="Choose first window for training LSTM data:")
    labelChooseWindow.pack()
    slider = tk.Scale(rootLSTM, from_=1, to=GUIobject.lastWindow, resolution=GUIobject.samplePeriod, length=900, orient=tk.HORIZONTAL, command=on_change_WINDOWSLIDERTTRAIN)
    slider.pack()
    
    #Spinbox: choose timesteps past:
    def on_change_TIMESTEPSPASTSPINBOX():
        GUI_LSTM_configuration.selectedTimeStepsPast = int(GUI_LSTM_configuration.spinBoxTimeStepsPast.get())
        new_options = get_divisors(GUI_LSTM_configuration.selectedTimeStepsPast)
        GUI_LSTM_configuration.dropdownChooseSubseqsPastSteps['values'] = new_options
        GUI_LSTM_configuration.dropdownChooseSubseqsPastSteps.set(new_options[0])
    labelChooseTimestepsPast = tk.Label(rootLSTM, text="Choose here the timesteps past:")
    labelChooseTimestepsPast.pack()
    spinboxChooseTimestepsPast = ttk.Spinbox(rootLSTM, from_=1, to=1000000, increment=1, command=on_change_TIMESTEPSPASTSPINBOX)
    spinboxChooseTimestepsPast.set(1)
    spinboxChooseTimestepsPast.pack()
    
    #CheckBox: checkbox to select CNN or not CNN layer
    def on_change_CNNLAYER():
        GUI_LSTM_configuration.switchCNN_layer()
    var = tk.BooleanVar()
    checkbox = tk.Checkbutton(rootLSTM, text="Convolutional Layer", variable=var, command=on_change_CNNLAYER)
    checkbox.pack()
    
    #Dropdown Menu ComboBox to choose the subsequence timesteps (only for CNN layer):
    def on_change_TIMESTEPSSUBSEQUENCE(*args):
        GUI_LSTM_configuration.selectedTimeStepsSubsequence = int(dropdownChooseSubseqsPastSteps.get())
        GUI_LSTM_configuration.spinBoxKernelSize['values'] = list(range(1, GUI_LSTM_configuration.selectedTimeStepsSubsequence + 1))
    labelChooseSubseqsPastSteps = tk.Label(rootLSTM, text="Timesteps of subsequence (only for CNN layer):")
    labelChooseSubseqsPastSteps.pack()
    dropdownChooseSubseqsPastSteps = tk.ttk.Combobox(rootLSTM)
    dropdownChooseSubseqsPastSteps['values'] = [1]
    dropdownChooseSubseqsPastSteps.set(1)
    dropdownChooseSubseqsPastSteps.pack()
    dropdownChooseSubseqsPastSteps.bind("<<ComboboxSelected>>", on_change_TIMESTEPSSUBSEQUENCE)
    
    #Spinbox to choose the neighbour convolutional timesteps (also called kernel size) (only for CNN layer):
    def on_change_KERNELSIZESPINBOX():
        GUI_LSTM_configuration.selectedKernelSize = int(GUI_LSTM_configuration.spinBoxKernelSize.get())
    labelChooseKernelSize = tk.Label(rootLSTM, text="Choose here the Kernel size (only for CNN layer):")
    labelChooseKernelSize.pack()
    spinboxChooseKernelSize = ttk.Spinbox(rootLSTM, from_=1, to=1, increment=1, command=on_change_KERNELSIZESPINBOX)
    spinboxChooseKernelSize.set(1)
    spinboxChooseKernelSize.pack()
    
    #Spinbox to choose LSTM units:
    def on_change_LSTMUnitsSPINBOX():
        GUI_LSTM_configuration.LSTMunits = int(GUI_LSTM_configuration.spinboxChooseLSTMunits.get())
    labelChooseLSTMunits = tk.Label(rootLSTM, text="LSTM units:")
    labelChooseLSTMunits.pack()
    spinboxChooseLSTMunits = ttk.Spinbox(rootLSTM, from_=1, to=1000000, increment=1, command=on_change_LSTMUnitsSPINBOX)
    spinboxChooseLSTMunits.set(70)
    spinboxChooseLSTMunits.pack()
    
    #Spinbox to choose LSTM epochs:
    def on_change_LSTMEpochsSPINBOX():
        GUI_LSTM_configuration.epochs = int(GUI_LSTM_configuration.spinboxChooseLSTMepochs.get())
    labelChooseLSTMepochs = tk.Label(rootLSTM, text="LSTM epochs:")
    labelChooseLSTMepochs.pack()
    spinboxChooseLSTMepochs = ttk.Spinbox(rootLSTM, from_=1, to=1000000, increment=1, command=on_change_LSTMEpochsSPINBOX)
    spinboxChooseLSTMepochs.set(10)
    spinboxChooseLSTMepochs.pack()
    
    #Spinbox to choose LSTM epochs:
    def on_change_LSTMPreviousWeeksSPINBOX():
        GUI_LSTM_configuration.previous_weeks = int(GUI_LSTM_configuration.spinboxChoosePreviousWeeks.get())
    labelChoosePreviousWeeks = tk.Label(rootLSTM, text="Choose previous weeks for training:")
    labelChoosePreviousWeeks.pack()
    spinboxChoosePreviousWeeks = ttk.Spinbox(rootLSTM, from_=0, to=GUIobject.displaying_week, increment=1, command=on_change_LSTMPreviousWeeksSPINBOX)
    spinboxChoosePreviousWeeks.set(0)
    spinboxChoosePreviousWeeks.pack()
    
    #Next button: when user presses the button "Next", a new window must appear to configure the LSTM
    def run_simulation_BUTTON():
        runPrediction(GUIobject, GUI_LSTM_configuration)
    run_button = tk.Button(rootLSTM, text="Run simulation", command=run_simulation_BUTTON)
    run_button.pack()
    
    GUI_LSTM_configuration = GraphicalUserInterface_LSTM_configuration(root = rootLSTM,
                                                                       recurrent_forecast=False,
                                                                       future_timesteps_recurrent=1,
                                                                       normalization="min-max",
                                                                       CNN=False,
                                                                       blue_lines_plot = blue_lines_plot,
                                                                       time_serie_plot_ax = time_serie_plot_ax,
                                                                       time_serie_plot_canvas = canvas_Traffic,
                                                                       theta_serie_blue_line_plot = theta_serie_blue_line_plot,
                                                                       theta_serie_plot_ax = theta_serie_plot_ax,
                                                                       spinBoxTimeStepsPast = spinboxChooseTimestepsPast,
                                                                       dropdownChooseSubseqsPastSteps = dropdownChooseSubseqsPastSteps,
                                                                       spinBoxKernelSize = spinboxChooseKernelSize,
                                                                       spinboxChooseLSTMunits = spinboxChooseLSTMunits,
                                                                       spinboxChooseLSTMepochs = spinboxChooseLSTMepochs,
                                                                       spinboxChoosePreviousWeeks = spinboxChoosePreviousWeeks)
    return GUI_LSTM_configuration
    

def get_timestamp_list():
    result = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for weekday in weekdays:
        for hour in range(24):
            for min in range(60):
                for sec in range(60):
                    if hour < 10:
                        hour_str = '0' + str(hour)
                    else:
                        hour_str = str(hour)
                    if min < 10:
                        min_str = '0' + str(min)
                    else:
                        min_str = str(min)
                    if sec < 10:
                        sec_str = '0' + str(sec)
                    else:
                        sec_str = str(sec)
                    result.append(weekday + " " + hour_str + ":" + min_str + ":" + sec_str)
    result.pop(0)
    result.append("Monday 00:00:00")
    return result
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




def createDefaultGUI(data_object):
    #Function to initialize GUI:
    root = tk.Tk()
    root.title("Predictor")
    root.geometry("1080x900")
    
    #Next button: when user presses the button "Next", a new window must appear to configure the LSTM
    def next_button_functionality():
        GUI_LSTM_configuration = createDefaultLSTMconfigurationDialogueGUI(GUIobject)
        GUI_LSTM_configuration.root.mainloop()
        
    next_button = tk.Button(root, text="Next", command=next_button_functionality)
    next_button.pack()
    
    #Week spinbox:
    #Functionality: when user changes the week, the plot of network traffic week time series must update, as well as the plot of zoom network traffic and theta params plot
    def on_change_WEEKSPINBOX():
        #Take the chosen current week by user:
        GUIobject.displaying_week = int(GUIobject.spinBoxWeek.get())
        #If selected week is different of previous selected week, update plot of time serie and theta-params:
        if GUIobject.displaying_week != GUIobject.displaying_week_was:
            #Update network time series week plot:
            update_week_network_traffic_time_serie_plot(GUIobject)
            #Update zoom window time serie plot:
            update_windowzoom_network_traffic_time_serie_plot(GUIobject)
            #Update the theta-params plot:
            update_theta_param_time_serie_plot(GUIobject)
            #Also update the red lines delimiters to change the max value:
            update_week_network_traffic_time_serie_plot_redlines(GUIobject)
            GUIobject.displaying_week_was = GUIobject.displaying_week
    labelChooseWeek = tk.Label(root, text="Choose here the week:")
    labelChooseWeek.pack()
    spinboxChooseWeek = ttk.Spinbox(root, from_=0, to=data_object.all_series.shape[0]-1, increment=1, command=on_change_WEEKSPINBOX)
    spinboxChooseWeek.set(0)
    spinboxChooseWeek.pack()
    
    #Plot de la serie temporal inicial:
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(8, 4))
    time_serie = data_object.all_series[0]
    timestamps = get_timestamp_list()
    time_serie_plot = ax.plot(time_serie, linewidth=0.1, color='blue')
        #Plot de los límites de la ventana:
    x = [0, data_object.Tsventana-1, data_object.Tsventana-1, 0, 0]; y = [0, 0, max(take_not_nan_values(time_serie)), max(take_not_nan_values(time_serie)), 0]
    window_limits_plot = ax.plot(x, y, color='red')
    ax.set_xlim(0, len(time_serie)-1); ax.set_ylim(1, 150000); ax.grid(True, alpha=0.5)
        #Plot de la ventana zoom:
    window = time_serie[x[0]:x[1]+1]
    x_axis = timestamps[x[0]:x[1]+1]
    #Configure x_ticks:
    ax_zoom.set_xticks([0, len(x_axis)]); ax_zoom.set_xticklabels([x_axis[0], x_axis[-1]], rotation=10)
    time_serie_zoom_plot = ax_zoom.plot(x_axis, window, linewidth=0.5, color='blue'); ax_zoom.set_xlim(1, len(window))
    #Canvas del plot:
    canvas_netTraffic = FigureCanvasTkAgg(fig, master=root)
    canvas_netTraffic.get_tk_widget().pack()
    
    #DropDown Menu for sampling value:
    #Functionality: when user selects a sampling period, it must update the step of slider and the plot of theta params, as well as window zoom plot:
    def on_change_SAMPLINGDROPDOWN(chosen_sample_period):
        GUIobject.windowSliderSelector.config(resolution=chosen_sample_period)
        GUIobject.samplePeriod=chosen_sample_period
        #Update last training window:
        GUIobject.lastWindow = GUIobject.selectedWindow - GUIobject.samplePeriod*int(GUIobject.data.scope/GUIobject.samplePeriod)
        #Update theta param time serie plot:
        update_theta_param_time_serie_plot(GUIobject)
        #Update zoom window time serie plot:
        update_windowzoom_network_traffic_time_serie_plot(GUIobject)
        #The window zoom plot must update whenever user changes sample period, because the chosen window must be multiple of sample period
        
    labelChooseSampling = tk.Label(root, text="Sampling [s]:")
    labelChooseSampling.pack()
    selected_option = tk.StringVar(root)
    selected_option.set("1")
    options = get_divisors(data_object.scope)
    dropdownChooseSampling = tk.OptionMenu(root, selected_option, *options, command=on_change_SAMPLINGDROPDOWN)
    dropdownChooseSampling.pack()
    
    #Window test slider selector:
    #Functionality: when user slides the slider button of test window, it must update the red lines (delimiters of the window), the zoom plot and the last-second line delimiter in theta serie:
    def on_change_WINDOWSLIDER(selectedWindow):
        GUIobject.selectedWindow = int(selectedWindow)
        #Update last training window:
        GUIobject.lastWindow = GUIobject.selectedWindow - GUIobject.samplePeriod*int(GUIobject.data.scope/GUIobject.samplePeriod)
        #Update red lines delimiters in time serie network traffic plot:
        update_week_network_traffic_time_serie_plot_redlines(GUIobject)
        #Update zoom window time serie plot:
        update_windowzoom_network_traffic_time_serie_plot(GUIobject)
        #Update red line delimiter in theta param serie plot:
        update_theta_param_time_serie_plot_redline(GUIobject)
    labelChooseWindow = tk.Label(root, text="Choose window:")
    labelChooseWindow.pack()
    slider = tk.Scale(root, from_=0, to=len(time_serie)-(data_object.Tsventana-1), resolution=1, length=900, orient=tk.HORIZONTAL, command=on_change_WINDOWSLIDER)
    slider.pack()
    
    #Theta serie parameter spinbox:
    #Functionality: when user changes the theta index, it must update the theta series plot:
    def on_change_THETAINDEXSPINBOX():
        GUIobject.displaying_theta = int(spinboxChooseThetaSerie.get())
        if GUIobject.displaying_theta != GUIobject.displaying_theta_was:
            #Update theta param time serie plot:
            update_theta_param_time_serie_plot(GUIobject)
            GUIobject.displaying_theta_was = GUIobject.displaying_theta
    labelChooseThetaSerie = tk.Label(root, text="Choose theta parameter time serie:")
    labelChooseThetaSerie.pack()
    spinboxChooseThetaSerie = ttk.Spinbox(root, from_=0, to=data_object.n, increment=1, command=on_change_THETAINDEXSPINBOX)
    spinboxChooseThetaSerie.set(0)
    spinboxChooseThetaSerie.pack()
    
    #Plot de la serie temporal theta:
    fig_theta, ax_theta = plt.subplots()
    theta_evolution = data_object.get_theta_time_series(week=0, coeff_index=0, samplingPeriod=1, time_from=0, time_to=-1)
    theta_serie_plot = ax_theta.plot(theta_evolution, linewidth=1, color='orange')
    if bool(take_not_nan_values(theta_evolution)):
        y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
        ax_theta.set_ylim(y[0], y[1])
        #ax_theta.set_xlim(take_not_nan_indexes(theta_evolution)[0], take_not_nan_indexes(theta_evolution)[1])
    else:
        y = [0, 1]
    x = [1, 1]
    theta_delimiter = ax_theta.plot(x, y, color='red')
    canvas_thetaSerie = FigureCanvasTkAgg(fig_theta, master=root)
    canvas_thetaSerie.get_tk_widget().pack()
    
    real_coefs = []
    for coef in range(data_object.n+1):
        real_coefs.append(data_object.get_theta_time_series(week=0,
                                                            coeff_index=coef,
                                                            samplingPeriod=1,
                                                            time_from=0,
                                                            time_to=0))
    GUIobject = GraphicalUserInterface(root = root,
                                       data = data_object,
                                       spinBoxWeek = spinboxChooseWeek,
                                       time_serie_plot = time_serie_plot,
                                       time_serie_plot_canvas = canvas_netTraffic,
                                       displaying_week_was = 0,
                                       displaying_theta_was = 0,
                                       samplingDropDownMenu = dropdownChooseSampling,
                                       samplePeriod = 1,
                                       windowSliderSelector = slider,
                                       selectedWindow = 0,
                                       red_lines_delimiters_plot = window_limits_plot,
                                       window_zoom_plot = time_serie_zoom_plot,
                                       spinBoxTheta = spinboxChooseThetaSerie,
                                       theta_serie_plot = theta_serie_plot,
                                       theta_serie_line_delimiter_plot = theta_delimiter,
                                       theta_serie_plot_canvas = canvas_thetaSerie,
                                       time_serie_plot_ax = ax,
                                       window_plot_ax = ax_zoom,
                                       theta_param_plot_ax = ax_theta,
                                       current_theta_evolution = theta_evolution,
                                       window_test=window,
                                       real_coefs = real_coefs,
                                       timestamps = timestamps)
    
    return GUIobject


def createGUI(filenameJSON):
    #Paso 0: obtener datos:
    JSONobject = getJSONTrendDynamicsData(filenameJSON)
    #Crear un objeto de datos:
    data_object = dataModule.data_class(
                 all_series = getAllSeriesMatrix(JSONobject),
                 thetaParams = getThetaParamsMatrix(JSONobject),
                 BitsOrPackets = JSONobject["BitsOrPackets"],
                 domain = JSONobject["domain"],
                 domainFIT = JSONobject["domainFIT"],
                 labels = JSONobject["labels"],
                 n = JSONobject["n"],
                 simulated_windows = JSONobject["Number_of_simulated_windows"],
                 scope = JSONobject["Scope"],
                 Tsventana = JSONobject["Tsventana"])
    
    #Paso 1: mostrar la interfaz gráfica:
    GUIobject = createDefaultGUI(data_object)
    GUIobject.root.mainloop()


filenameJSON = '../Data_extraction/Data_extraction_output/'+str(sys.argv[1])
createGUI(filenameJSON)