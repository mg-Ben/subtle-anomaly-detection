# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 16:03:16 2023

@author: benja
"""

import numpy as np
#Modulos para la normalizacion y desnormalizacion
import math
from numpy import mean, std, tanh
import GeneralPurposeFunctions as utilities
#Modulos para la creacion de matrices XY
from numpy import array
#Modulos para la arquitectura de la LSTM
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Conv1D
#Modulos para la ejecucion por hilos
import threading

class threads_info():
    def __init__(self,
                 LSTM_model_dict={},
                 thread_dict={},
                 history_dict={}):
        self.LSTM_model_dict = LSTM_model_dict
        self.thread_dict = thread_dict
        self.history_dict = history_dict
        
    def add_LSTM_model(self, coef, LSTM_model_obj):
        self.LSTM_model_dict[coef] = LSTM_model_obj
    
    def add_thread(self, coef, thread_obj):
        self.thread_dict[coef] = thread_obj
        
    def add_history(self, coef, history_obj):
        self.history_dict[coef] = history_obj
        
class LSTM_input():
    def __init__(self,
                 XY_dict = {}, #Datos de entrenamiento de cada LSTM (de cada coeficiente)
                 test_sequences_dict = {},
                 validation_sequences_dict = {}):
        self.XY_dict = XY_dict
        self.test_sequences_dict = test_sequences_dict
        self.validation_sequences_dict = validation_sequences_dict
    
    def add_XY_to_dict(self,
                       coef_index,
                       XY_object):
        self.XY_dict.update(coef_index, XY_object)
        
class XY(): #Objeto que representa la matriz de caracteristicas y etiquetas del coeficiente theta_i
    def __init__(self,
                 X = [],
                 Y = [],
                 sequence_object = None,
                 normalizationType="",
                 normalization_parameters=[]):
        self.X = X
        self.Y = Y
        self.sequence_object = sequence_object
        self.normalizationType = normalizationType
        self.normalization_parameters = normalization_parameters

class sequence():
    def __init__(self,
                 week = -1,
                 coef = -1,
                 samplingPeriod = -1,
                 time_from= -1,
                 time_to= -1,
                 sequence=None, #np.array
                 normalized_sequence=None,
                 normalizationType="",
                 normalization_parameters = []
                 ):
        self.week=week
        self.coef = coef
        self.samplingPeriod = samplingPeriod
        self.time_from = time_from
        self.time_to = time_to
        self.sequence=sequence
        self.normalized_sequence = normalized_sequence
        self.normalizationType = normalizationType
        self.normalization_parameters = normalization_parameters

#Funciones de normalizacion:---------------------------------------------------------------------------------------------------------------
def get_normalization_parameters(data, normalizationType):
    if normalizationType == 'min-max':
        minimo = min(data)
        maximo = max(data)
        '''
        #Si se desea que esté entre 0 y 1, descomentar esta línea:
        #result = (data - minimo)/(maximo-minimo)
        #Si se desea que esté entre -1 y 1, descomentar estas líneas:
        valor_medio = minimo + (maximo-minimo)/2
        result = (data - valor_medio)/(maximo-valor_medio)
        '''
        return minimo, maximo
    elif normalizationType == 'tanh' or normalizationType == 'z-score':
        mu = mean(data)
        sigma = std(data)
        #result = (1/2)*(tanh(0.01*((data-mu)/sigma)) + 1)
        return mu, sigma
    else:
        print("Error: Normalization type is not min-max, z-score or tanh")

def normalize_array_using_normalization_parameters(data, normalization_parameters, normalizationType):
    if normalizationType == 'min-max':
        minimo = normalization_parameters[0]
        maximo = normalization_parameters[1]
        #Si se desea que esté entre 0 y 1, descomentar esta línea:
        #result = (data - minimo)/(maximo-minimo)
        #Si se desea que esté entre -1 y 1, descomentar estas líneas:
        valor_medio = minimo + (maximo-minimo)/2
        result = (data - valor_medio)/(maximo-valor_medio)
        return result
    elif normalizationType == 'tanh':
        mu = normalization_parameters[0]
        sigma = normalization_parameters[1]
        result = (1/2)*(tanh(0.01*((data-mu)/sigma)) + 1)
        return result
    elif normalizationType == 'z-score':
        mu = normalization_parameters[0]
        sigma = normalization_parameters[1]
        result = (data-mu)/sigma
        return result
    else:
        print("Error: Normalization type is not min-max, z-score or tanh")
        
def un_normalize_array_using_normalization_parameters(data, normalization_parameters, normalizationType):
    if normalizationType == 'min-max':
        minimo = normalization_parameters[0]
        maximo = normalization_parameters[1]
        #Si la normalización fue entre 0 y 1, descomentar esta línea:
        #result = data_normalized*(maximo-minimo) + minimo
        #Si la normalización fue entre -1 y 1, descomentar estas líneas:
        valor_medio = minimo + (maximo-minimo)/2
        result = data*(maximo-valor_medio) + valor_medio
        return result
    elif normalizationType == 'tanh':
        mu = normalization_parameters[0]
        sigma = normalization_parameters[1]
        result = np.zeros((data.shape[0],))
        for i in range(len(result)):
            result[i] = (math.atanh(2*data[i]-1)*100)*sigma + mu
        return result
    elif normalizationType == 'z-score':
        mu = normalization_parameters[0]
        sigma = normalization_parameters[1]
        result = data*sigma+mu
        return result
    else:
        print("Error: Normalization type is not min-max, z-score or tanh")
#-------------------------------------------------------------------------------------------------------------------------------------------
#Funciones para crear las matrices XY a partir de las secuencias de entrada:
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def add_list_to_list_in_dictionary(dictionary, key, lista):
    if key not in dictionary.keys():
        dictionary[key] = lista
    else:
        current_element_list = dictionary[key]
        current_element_list = np.concatenate((np.array(current_element_list), np.array(lista)))
        dictionary[key] = list(current_element_list)
    return dictionary

def add_element_to_list_in_dictionary(dictionary, key, element):
    if key not in dictionary.keys():
        dictionary[key] = [element]
    else:
        current_element_list = dictionary[key]
        current_element_list.append(element)
        dictionary[key] = current_element_list
    return dictionary

def get_week_sequences_objects(data_object,
                               samplingPeriod,
                               semana_test,
                               previous_weeks,
                               window_from,
                               window_to):
    output_dict = {}
    semana_test_inicial = semana_test-previous_weeks-1
    #Funcion que concatena las secuencias de las series temporales de cada semana. Devuelve un diccionario cuyas keys son el indice de coeficiente y cuyos valores son listas de objetos secuencia:
    for week in range(semana_test, semana_test_inicial, -1):
        #Para las semanas previas a la semana de test, se toma como ventana final hasta window_to + (window_to - window_from)
        if week < semana_test:
            window_to_final = window_to*2 - window_from
        else:
            window_to_final = window_to
        for coef in range(data_object.n+1):
            #Tomamos la serie temporal en el rango concreto para la semana week y coeficiente coef:
            theta_time_serie = data_object.get_theta_time_series(week = week,
                                                                 coeff_index = coef,
                                                                 samplingPeriod = samplingPeriod,
                                                                 time_from = window_from,
                                                                 time_to = window_to_final)
            #Eliminamos valores NoneType o nan:
            theta_time_serie = utilities.take_not_nan_values(theta_time_serie)
            #Convertimos la lista a un array:
            theta_time_serie = np.array(theta_time_serie)
            #Creamos el objeto secuencia:
            sequence_object = sequence(week=week,
                                       coef=coef,
                                       samplingPeriod=samplingPeriod,
                                       time_from=window_from,
                                       time_to=window_to_final,
                                       sequence=theta_time_serie)
            #Añadimos el objeto al diccionario:
            add_element_to_list_in_dictionary(output_dict, coef, sequence_object)
    
    return output_dict

def find_sequence_current_week_test(sequence_object_list):
    max_week = -1
    seq_result = None
    for sequence_obj in sequence_object_list:
        if sequence_obj.week > max_week:
            seq_result = sequence_obj
            max_week = sequence_obj.week
    return seq_result

def get_test_sequence_object(sequences_dict, timesteps_past):
    #Get the sequence object of the test week:
    dict_result = {}
    for coef, sequence_obj_list in sequences_dict.items():
        current_week_sequence_object = find_sequence_current_week_test(sequence_obj_list)
        current_week_sequence_object.sequence = current_week_sequence_object.sequence[-timesteps_past:]
        current_week_sequence_object.normalized_sequence = current_week_sequence_object.normalized_sequence[-timesteps_past:]
        dict_result[coef] = current_week_sequence_object
    return dict_result

def get_validation_sequence_object(data_object,
                                   coef,
                                   samplingPeriod,
                                   semana_test,
                                   window_from,
                                   window_to):
    return data_object.get_theta_time_series(week = semana_test,
                                             coeff_index = coef,
                                             samplingPeriod = samplingPeriod,
                                             time_from = window_from,
                                             time_to = window_to)
                                      
def get_validation_sequence_object_dict(data_object,
                                        samplingPeriod,
                                        semana_test,
                                        window_from,
                                        window_to):
    dict_result = {}
    for coef in range(data_object.n+1):
        validation_sequence = get_validation_sequence_object(data_object,
                                                             coef,
                                                             samplingPeriod,
                                                             semana_test,
                                                             window_from,
                                                             window_to)
        #Convertir a np.array():
        validation_sequence = np.array(list(validation_sequence))
        #Crear objeto sequence:
        validation_sequence_obj = sequence(week = semana_test,
                                           coef = coef,
                                           samplingPeriod = samplingPeriod,
                                           time_from= window_from,
                                           time_to= window_to,
                                           sequence=validation_sequence, #np.array
                                           normalized_sequence=None,
                                           normalizationType="",
                                           normalization_parameters = []
                                           )
                                
        
        #Añadimos el objeto al diccionario:
        dict_result[coef] = validation_sequence_obj
    return dict_result
    
    
#Concatenacion de secuencias:
def concatenate_week_sequences_objects_list(sequence_object_list):
    sequence_result = []
    for sequence_obj in sequence_object_list:
        #Get array:
        sequence_result = sequence_result + list(sequence_obj.sequence)
    return np.array(sequence_result)

def concatenate_week_sequences_objects(dictionary):
    dict_result = {}
    for coef, sequence_obj_list in dictionary.items():
        dict_result[coef] = concatenate_week_sequences_objects_list(sequence_obj_list)
    return dict_result


#Normalizacion de secuencias:
def normalize_week_sequences_objects_list(sequence_object_list, normalizationType, normalization_parameters):
    for sequence_obj in sequence_object_list:
        sequence_obj.normalizationType = normalizationType
        sequence_obj.normalization_parameters = normalization_parameters
        sequence_obj.normalized_sequence = normalize_array_using_normalization_parameters(sequence_obj.sequence, normalization_parameters, normalizationType)
    pass
def normalize_week_sequences_objects(dictionary_sequences, normalizationType, normalization_parameters, coef):
    normalize_week_sequences_objects_list(dictionary_sequences[coef], normalizationType, normalization_parameters)
    
def normalize_data_dictionary(dictionary_concatenated_sequences,
                              normalizationType,
                              dictionary_sequences):
    for coef, concatenated_array_sequence in dictionary_concatenated_sequences.items():
        #1: Obtener los parametros de normalizacion con la secuencia concatenada
        normalization_parameters = get_normalization_parameters(concatenated_array_sequence, normalizationType)
        #2: Con los parametros de normalizacion DE LA CONCATENACION, se normalizan las secuencias INDEPENDIENTES (no concatenadas):
        normalize_week_sequences_objects(dictionary_sequences, normalizationType, normalization_parameters, coef)
    
def normalize_data_dictionary_validation(dictionary_validation_sequences, dictionary_sequences):
    for coef,sequence_obj in dictionary_validation_sequences.items():
        sequence_obj.normalizationType = dictionary_sequences[coef][0].normalizationType
        sequence_obj.normalization_parameters = dictionary_sequences[coef][0].normalization_parameters
        sequence_obj.normalized_sequence = normalize_array_using_normalization_parameters(sequence_obj.sequence, sequence_obj.normalization_parameters, sequence_obj.normalizationType)
    pass

#Get XY:
def get_XY_objects_list(sequence_object_list, timesteps_past, timesteps_future_final):
    XY_object_list = []
    for sequence_obj in sequence_object_list:
        #Get the XY object data:
        X, y = split_sequence(sequence_obj.normalized_sequence, timesteps_past, timesteps_future_final)
        #Crear el objeto XY:
        XY_object = XY(X = X,
                       Y = y,
                       sequence_object=sequence_obj)
        #Añadirlo al array de salida:
        XY_object_list.append(XY_object)
    return XY_object_list

def get_XY_objects_dictionary(sequences_dict, timesteps_past, timesteps_future_final):
    dict_result = {}
    for coef, sequence_object_list in sequences_dict.items():
        XY_object_list = get_XY_objects_list(sequence_object_list, timesteps_past, timesteps_future_final)
        dict_result[coef] = XY_object_list
    return dict_result

#Merge XY:
def merge_XY_objects_list(XY_object_list):
    merged_X = XY_object_list[0].X
    merged_Y = XY_object_list[0].Y
    for XY_object in XY_object_list[1:]:
        if list(XY_object.X) and list(XY_object.Y): #Si hay datos, se hace merge:
            merged_X = np.concatenate((merged_X, XY_object.X))
            merged_Y = np.concatenate((merged_Y, XY_object.Y))
    #Return XY object with merged X and merged Y:
    return XY(X = merged_X,
              Y = merged_Y,
              normalizationType=XY_object_list[0].sequence_object.normalizationType,
              normalization_parameters=XY_object_list[0].sequence_object.normalization_parameters)
       
def merge_XY_objects_dictionary(XY_objects_dictionary):
    dict_result = {}
    for coef, XY_object_list in XY_objects_dictionary.items():
        dict_result[coef] = merge_XY_objects_list(XY_object_list)
    return dict_result

#Reshaping XY para LSTM y preparacion para capa convolucional si fuera necesario:
def reshape_XY_object(XY_object,
                      CNN,
                      timesteps_past,
                      TimeStepsSubsequence,
                      n_features=1):
    if CNN == 1: #Para CNN + LSTM, se debe dimensionar la matriz de entrenamiento de una forma concreta, asi como la de etiquetas Y:
        n_subseqs = int(timesteps_past/TimeStepsSubsequence) #Numero de subsecuencias
        XY_object.X = XY_object.X.reshape((XY_object.X.shape[0], n_subseqs, TimeStepsSubsequence, n_features))
    elif CNN == 0: #Para LSTM simple:
        XY_object.X = XY_object.X.reshape((XY_object.X.shape[0], XY_object.X.shape[1], n_features))
        XY_object.Y = XY_object.Y.reshape((XY_object.Y.shape[0], XY_object.Y.shape[1], n_features))
    pass
def reshape_XY_objects_dictionary(XY_objects_dictionary_merged,
                                  CNN,
                                  timesteps_past,
                                  TimeStepsSubsequence):
    for coef, XY_object in XY_objects_dictionary_merged.items():
        reshape_XY_object(XY_object, CNN, timesteps_past, TimeStepsSubsequence)
    pass
    
def reshape_test_sequence_object(sequence_obj,
                                 CNN,
                                 timesteps_past,
                                 TimeStepsSubsequence,
                                 n_features=1):
    #Para CNN + LSTM:
    if CNN == 1:
        n_subseqs = int(timesteps_past/TimeStepsSubsequence) #Numero de subsecuencias
        return sequence_obj.normalized_sequence.reshape((1, n_subseqs, TimeStepsSubsequence, n_features))
    #Para LSTM simple:
    if CNN == 0:
        return sequence_obj.normalized_sequence.reshape((1, timesteps_past, n_features))

def reshape_test_sequence_object_dict(test_sequence_dict,
                                      CNN,
                                      timesteps_past,
                                      TimeStepsSubsequence,
                                      n_features=1):
    dict_result = {}
    for coef,sequence_obj in test_sequence_dict.items():
        dict_result[coef] = reshape_test_sequence_object(sequence_obj,
                                                         CNN,
                                                         timesteps_past,
                                                         TimeStepsSubsequence,
                                                         n_features=1)
                                     
    return dict_result

def reshape_validation_sequence_object(sequence_obj,
                                       CNN,
                                       timesteps_past,
                                       TimeStepsSubsequence,
                                       n_features=1):
    #Para LSTM CNN:
    if CNN == 1:
        return sequence_obj.normalized_sequence.reshape((1, sequence_obj.normalized_sequence.shape[0], 1))
    #Para LSTM simple:
    if CNN == 0:
        return sequence_obj.normalized_sequence.reshape((1, sequence_obj.normalized_sequence.shape[0], n_features))
    
def reshape_validation_sequence_object_dict(validation_sequence_dict,
                                            CNN,
                                            timesteps_past,
                                            TimeStepsSubsequence,
                                            n_features=1):
    dict_result = {}
    for coef,sequence_obj in validation_sequence_dict.items():
        dict_result[coef] = reshape_validation_sequence_object(sequence_obj,
                                                               CNN,
                                                               timesteps_past,
                                                               TimeStepsSubsequence,
                                                               n_features=1)                          
    return dict_result











def get_X_Y_training_data(data_object,
                          samplingPeriod,
                          semana_test,
                          previous_weeks,
                          window_from,
                          window_to,
                          normalization,
                          timesteps_past,
                          recurrentForecast,
                          timesteps_future,
                          timesteps_future_recurrent,
                          CNN,
                          TimeStepsSubsequence):
    #Funcion que recibe la matriz con los parametros theta y devuelve un objeto con los datos de entrenamiento (matrices X, Y) necesarios para cada LSTM:
    XY_dictionary_output = LSTM_input()
    #Obtener las secuencias de entrada a la red neuronal: concretamente, se concatenan las secuencias de cada semana para luego normalizarlas con respecto a la concatenacion:
    #Obtencion de los objetos secuencia:
    sequences_dict = get_week_sequences_objects(data_object,
                                                samplingPeriod,
                                                semana_test,
                                                previous_weeks,
                                                window_from,
                                                window_to)
    #Concatenacion:
    sequences_dict_concatenated = concatenate_week_sequences_objects(sequences_dict)
    #Normalizacion de la concatenacion:
    normalize_data_dictionary(sequences_dict_concatenated, normalization, sequences_dict)
    #Obtencion de un diccionario de matrices XY a partir de las secuencias normalizadas:
    #Decidir que time_steps_future usar: si se eligio prediccion recurrente, se cogera timesteps_future_recurrent para crear la matriz XY. Si no, se coge timesteps_future:
    timesteps_future_final = timesteps_future_recurrent if recurrentForecast is True else timesteps_future
    XY_objects_dictionary = get_XY_objects_dictionary(sequences_dict, timesteps_past, timesteps_future_final)
    #Juntar las matrices XY:
    XY_objects_dictionary_merged = merge_XY_objects_dictionary(XY_objects_dictionary)
    #Reshaping de las matrices XY para la LSTM:
    reshape_XY_objects_dictionary(XY_objects_dictionary_merged,
                                  CNN,
                                  timesteps_past,
                                  TimeStepsSubsequence)
    
    #Tomar la ventana de test y la validacion:
    #Test:
    #1. Tomar la secuencia de input de test a la red LSTM
    test_sequences_dict = get_test_sequence_object(sequences_dict, timesteps_past)
    #2. Reshaping
    test_sequences_dict_reshaped = reshape_test_sequence_object_dict(test_sequences_dict,
                                                                     CNN,
                                                                     timesteps_past,
                                                                     TimeStepsSubsequence,
                                                                     n_features=1)
    #Validation:
    #1. Tomar la secuencia de input de validacion a la red LSTM
    window_to_validation = window_to+data_object.scope if recurrentForecast is False else window_to+timesteps_future_final*samplingPeriod
    validation_sequence_dict = get_validation_sequence_object_dict(data_object=data_object,
                                                                   samplingPeriod=samplingPeriod,
                                                                   semana_test=semana_test,
                                                                   window_from=window_to+1,
                                                                   window_to=window_to_validation)
    #2. Normalizar secuencias de validacion
    normalize_data_dictionary_validation(validation_sequence_dict, sequences_dict)
    #3. Reshaping
    validation_sequences_dict_reshaped = reshape_validation_sequence_object_dict(validation_sequence_dict,
                                                                                 CNN,
                                                                                 timesteps_past,
                                                                                 TimeStepsSubsequence,
                                                                                 n_features=1)
    LSTM_input_object = LSTM_input(XY_dict = XY_objects_dictionary_merged,
                                   test_sequences_dict = test_sequences_dict_reshaped,
                                   validation_sequences_dict = validation_sequences_dict_reshaped)
    
    return LSTM_input_object

def reshape_test_sequence_to_1D(test_sequence, CNN):
    if CNN is False:
        return test_sequence.reshape(test_sequence.shape[1],)
    else:
        return test_sequence.reshape(test_sequence.shape[1]*test_sequence.shape[2],)

#def reshape_test_sequence_to_original_shape():
    
    
def create_LSTM_architecture(CNN,
                             KernelSize,
                             LSTM_input_object,
                             recurrentForecast,
                             timesteps_future,
                             timesteps_future_recurrent,
                             timesteps_past,
                             LSTMunits,
                             n_features=1):
    timesteps_future_final = timesteps_future_recurrent if recurrentForecast is True else timesteps_future
    #CNN + LSTM:
    if CNN == 1:
        model = Sequential()
        #ENCODER:
        model.add(TimeDistributed(Conv1D(filters=16, kernel_size=KernelSize, activation='relu'), input_shape=(None, LSTM_input_object.XY_dict[0].X.shape[2], n_features)))
        model.add(TimeDistributed(Conv1D(filters=8, kernel_size=KernelSize, activation='relu')))
        #model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(LSTMunits, activation='relu'))
        #SIZE FIX:
        model.add(RepeatVector(timesteps_future_final))
        #DECODER:
        model.add(LSTM(LSTMunits, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        #optimizer = Adam(learning_rate=0.001)
        #model.compile(optimizer=optimizer, loss='mse')
        model.compile(optimizer='adam', loss='mse')
    
    #LSTM Encoder Decoder simple:
    if CNN == 0:
        model = Sequential()
        model.add(LSTM(LSTMunits, activation='relu', input_shape=(timesteps_past, n_features)))
        model.add(RepeatVector(timesteps_future_final))
        model.add(LSTM(LSTMunits, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')
        
    return model


def trainLSTM(LSTM_model, X, Y, epochs, test_sequence, validation_sequence, general_info_threads, coef):
    history = LSTM_model.fit(X, Y, epochs=epochs, verbose=1, validation_data=(test_sequence, validation_sequence)) #callbacks=[trainingStopCallback]
    general_info_threads.add_history(coef, history)
    pass



def run_forecasting_threads(data_object, LSTM_model, LSTM_input_object, epochs):
    general_info_threads = threads_info()
    for coef in range(data_object.n+1):
        X = LSTM_input_object.XY_dict[coef].X
        Y = LSTM_input_object.XY_dict[coef].Y
        test_sequence = LSTM_input_object.test_sequences_dict[coef]
        validation_sequence = LSTM_input_object.validation_sequences_dict[coef]
        
        general_info_threads.add_LSTM_model(coef, LSTM_model)
        forecasting_thread = threading.Thread(target=trainLSTM, args=(general_info_threads.LSTM_model_dict[coef], X, Y, epochs, test_sequence, validation_sequence, general_info_threads, coef))
        general_info_threads.add_thread(coef, forecasting_thread)
        forecasting_thread.start()
    
    #Await:
    for coef in range(data_object.n+1):
        general_info_threads.thread_dict[coef].join()
    
    return general_info_threads
    

def predict_coefficients(LSTM_input_object, general_info_threads, recurrentForecast, timesteps_future, timesteps_future_recurrent, CNN):
    predicted_coefs_normalized = {}
    predicted_coefs = {}
    truth_coefs = {}
    timesteps_past = LSTM_input_object.XY_dict[0].X.shape[1] if CNN is False else LSTM_input_object.XY_dict[0].X.shape[1]*LSTM_input_object.XY_dict[0].X.shape[2]
    recurrent_times = int(timesteps_future/timesteps_future_recurrent)
    for coef, test_sequence in LSTM_input_object.test_sequences_dict.items():
        if recurrentForecast is True:
            for i in range(recurrent_times):
                #Realizar prediccion para este coeficiente:
                coefs_normalized = general_info_threads.LSTM_model_dict[coef].predict(test_sequence)
                #Reshaping y conversion a np.array:
                coefs_normalized = coefs_normalized.reshape(coefs_normalized.shape[1],)
                #Unnormalize predicted coefficients and save in dict result:
                coefs_unnormalized = list(un_normalize_array_using_normalization_parameters(coefs_normalized, LSTM_input_object.XY_dict[coef].normalization_parameters, LSTM_input_object.XY_dict[coef].normalizationType))
                add_list_to_list_in_dictionary(predicted_coefs, coef, coefs_unnormalized)
                test_sequence_aux = reshape_test_sequence_to_1D(test_sequence, CNN)
                test_sequence_aux = np.concatenate((test_sequence_aux, coefs_normalized))
                test_sequence_aux = test_sequence_aux[-timesteps_past:]
                test_sequence = test_sequence_aux.reshape(test_sequence.shape)
        else:
            #Realizar prediccion para este coeficiente:
            coefs_normalized = general_info_threads.LSTM_model_dict[coef].predict(test_sequence)
            #Reshaping y conversion a np.array:
            coefs_normalized = coefs_normalized.reshape(coefs_normalized.shape[1],)
            #Unnormalize predicted coefficients and save in dict result:
            coefs_unnormalized = list(un_normalize_array_using_normalization_parameters(coefs_normalized, LSTM_input_object.XY_dict[coef].normalization_parameters, LSTM_input_object.XY_dict[coef].normalizationType))
            add_list_to_list_in_dictionary(predicted_coefs, coef, coefs_unnormalized)
 
    return predicted_coefs
        
    
    
def LSTM_network_forecasting_algorithm(GUIobject, GUILSTMobject):
    #Captura de datos de la interfaz:
    #Parametros generales:
    Tsventana = GUIobject.data.Tsventana #Tamaño de ventana que se usó para sacar los coeficientes y parámetros alpha-stable (poner mismo valor que en TrendDynamics.m)
    n = GUIobject.data.n
    scope = GUIobject.data.scope
    diezmado = GUIobject.samplePeriod
    ventana_simulacion = GUIobject.selectedWindow
    semana_test = GUIobject.displaying_week
    
    #Parametros LSTM:
    timesteps_future = int(GUIobject.data.scope/GUIobject.samplePeriod) #Número de puntos futuros usados para la predicción
    timesteps_past = GUILSTMobject.selectedTimeStepsPast
    recurrentForecast = GUILSTMobject.recurrent_forecast
    timesteps_future_recurrent = GUILSTMobject.future_timesteps_recurrent
    normalization = GUILSTMobject.normalization
    selectedFirstWindow = GUILSTMobject.selectedFirstWindow
    lastWindow = GUIobject.lastWindow
    CNN = GUILSTMobject.CNN
    TimeStepsSubsequence = GUILSTMobject.selectedTimeStepsSubsequence
    KernelSize= GUILSTMobject.selectedKernelSize
    LSTMunits = GUILSTMobject.LSTMunits
    epochs = GUILSTMobject.epochs
    previous_weeks = GUILSTMobject.previous_weeks
    
    #1: Obtencion de las series temporales para el entrenamiento:
    #El primer paso consiste en obtener las secuencias de entrada a la red neuronal. Se obtendra una secuencia por cada semana y por cada coeficiente:
    LSTM_input_object = get_X_Y_training_data(data_object = GUIobject.data,
                                              samplingPeriod = diezmado,
                                              semana_test = semana_test,
                                              previous_weeks=previous_weeks,
                                              window_from = selectedFirstWindow,
                                              window_to = lastWindow,
                                              normalization = normalization,
                                              timesteps_past = timesteps_past,
                                              recurrentForecast = recurrentForecast,
                                              timesteps_future = timesteps_future,
                                              timesteps_future_recurrent = timesteps_future_recurrent,
                                              CNN = CNN,
                                              TimeStepsSubsequence = TimeStepsSubsequence)
    #A continuacion, se crea la red neuronal:
    LSTM_model = create_LSTM_architecture(CNN,
                                          KernelSize,
                                          LSTM_input_object,
                                          recurrentForecast,
                                          timesteps_future,
                                          timesteps_future_recurrent,
                                          timesteps_past,
                                          LSTMunits,
                                          n_features=1)
    
    general_info_threads = run_forecasting_threads(GUIobject.data, LSTM_model, LSTM_input_object, epochs)
    #Ejecutar la prediccion:
    final_prediction = predict_coefficients(LSTM_input_object,
                                            general_info_threads,
                                            recurrentForecast,
                                            timesteps_future,
                                            timesteps_future_recurrent,
                                            CNN)
    return final_prediction
                         