import numpy as np
import math
import sys

T = int(sys.argv[1])

def centroid(weights, data):
    dim = len(data[0]) 
    res = np.asarray([0.0 for x in range(dim)]) 
    for i in range(len(data)):
        #print(res)
        #print(weights[i], data[i])
        res += weights[i] * data[i]
    res *= 1/(np.sum(weights)) 
    return res

def perform_round(iter_num, w_list, t_list, alphas, pos_data, neg_data, pos_weights, neg_weights):
    errors = 0
    pos_errors = []
    neg_errors = []
    pos_centroid = centroid(pos_weights, pos_data)
    neg_centroid = centroid(neg_weights, neg_data)
    w = np.subtract(pos_centroid, neg_centroid)
    t = np.multiply(.5, np.dot(np.sum([pos_centroid,neg_centroid],0), w))

    for i in range(len(pos_data)):
        t_calc = np.dot(pos_data[i],w)
        if t_calc >= t: 
            pos_errors.append(0)
        else:
            errors += 1
            pos_errors.append(1)
    for i in range(len(neg_data)):
        t_calc = np.dot(neg_data[i],w)
        if t_calc < t: 
            neg_errors.append(0)
        else:
            errors += 1
            neg_errors.append(1)

    error_rate = errors / (len(pos_data) + len(neg_data)) 
    if (error_rate >= .5):
        return False
    w_list.append(w)
    t_list.append(t)

    alpha = 0.5 * np.log((1-error_rate)/error_rate)
    alphas.append(alpha)    
    increase_factor = 1 / (2 * error_rate)
    decrease_factor = 1 / (2 * (1 - error_rate))
    print("Iteration " + str(iter_num)) 
    print("Error = " + str(error_rate))
    print("Alpha = " + str(alpha))
    print("Factor to increase weights = " + str(increase_factor))
    print("Factor to decrease weights = " + str(decrease_factor))
    
    for i in range(len(pos_weights)):
        if (pos_errors[i] == 1):
            pos_weights[i] = pos_weights[i] * increase_factor 
        else:
            pos_weights[i] = pos_weights[i] * decrease_factor 
    for i in range(len(neg_weights)):
        if (neg_errors[i] == 1):
            neg_weights[i] = neg_weights[i] * increase_factor 
        else:
            neg_weights[i] = neg_weights[i] * decrease_factor 
            
    return True

pos_weights = [] 
neg_weights = [] 
pos_training_data = open(sys.argv[2])
neg_training_data = open(sys.argv[3])
pos_test_data = open(sys.argv[4])
neg_test_data = open(sys.argv[5])

pos_data = []
neg_data = []
pos_training_data.readline()
neg_training_data.readline()
for line in pos_training_data.readlines():
    data_pt = np.asarray([float(x) for x in line.split()])
    pos_weights.append(1)
    pos_data.append(data_pt)
for line in neg_training_data.readlines():
    data_pt = np.asarray([float(x) for x in line.split()])
    neg_weights.append(1)
    neg_data.append(data_pt)

w_list = []
t_list = []
alphas = []
for i in range(T):
    if (not perform_round(i+1, w_list, t_list, alphas, pos_data, neg_data, pos_weights, neg_weights)):
        T = i 
        break
t_avg = 0
w_avg = None 
first = True

for i in range(T):
    t_avg += alphas[i] * t_list[i] 
    if first:
        w_avg = w_list[i] * alphas[i]
        first = False
    else: w_avg += w_list[i] * alphas[i]

data_count = 0
pos_test_data.readline()
fp = 0
fn = 0
neg_test_data.readline()
for line in pos_test_data.readlines():
    data_pt = np.asarray([float(x) for x in line.split()])
    total = 0.0
    data_count += 1
    for i in range(T):
        t_calc = np.dot(data_pt,w_list[i])
        if (t_calc > t_list[i]):
            total += alphas[i] * 1
        else:
            total += alphas[i] * -1
    if total < 0:
        fn += 1
for line in neg_test_data.readlines():
    data_pt = np.asarray([float(x) for x in line.split()])
    total = 0.0
    data_count += 1
    for i in range(T):
        t_calc = np.dot(data_pt,w_list[i])
        if (t_calc > t_list[i]):
            total += alphas[i] * 1
        else:
            total += alphas[i] * -1
    if total >= 0:
        fp += 1
        
            
print("Testing:")
print("False positives: " + str(fp)) 
print("False negatives: " + str(fn)) 
print("Error rate: " + str((fp + fn) / data_count)) 


