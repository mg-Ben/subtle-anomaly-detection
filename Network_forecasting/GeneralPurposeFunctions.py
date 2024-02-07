import math
import numpy as np

def get_divisors(x):
    result = []
    for i in range(1, math.floor(x/2)+1):
        if x % i == 0:
            result.append(i)
    result.append(x)
    return result

def contains_nan(lst):
    for value in lst:
        if math.isnan(value):
            return True
    return False

def interpolate(array, Ts):
    result = []
    for i in range(1, len(array)):
        if array[i] is None or array[i-1] is None:
            adders = [None] * (Ts - 1)
        else:
            m = (array[i] - array[i-1]) / Ts
            adders = array[i-1] + m * np.arange(1, Ts)

        result.extend([array[i-1]] + list(adders))
        
    return result
'''
def interpolate(array, Ts):
    result = []
    for i in range(1, len(array)):
        adders = []
        if array[i] == None or array[i-1] == None:
            for x in range(1, Ts):
                adders.append(None)
        else:
            m = (array[i]-array[i-1])/Ts
            for x in range(1, Ts):
                adders.append(array[i-1] + m*x)
        result = result + [array[i-1]] + adders
        
    return result
'''
def map_None_to_nan(element):
    if element is None:
        return np.nan
    return element
def take_not_nan_values(array):
    #Convert all NoneType to np.nan:
    array = list(map(map_None_to_nan, array))
    array = [element for element in array if math.isnan(element) is False]
    return [element for element in array if math.isnan(element) is False]

def take_not_nan_indexes(array):
    index_from = 0
    index_to = 0
    #Convert all NoneType to np.nan:
    array = list(map(map_None_to_nan, array))
    for i in range(len(array)):
        e = array[i]
        if math.isnan(e) is False:
            index_from = i
            break
    for i in range(len(array)-1, -1, -1):
        e = array[i]
        if math.isnan(e) is False:
            index_to = i
            break
    
    return index_from, index_to

def all_nan(array):
    array = list(map(map_None_to_nan, array))
    nan_string_list = [str(x) for x in array]
    if len(list(set(nan_string_list))) == 1 and nan_string_list[0] == 'nan':
        return True
    else:
        return False