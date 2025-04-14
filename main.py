#import antigravity
import numpy as np
#import pandas as pd
import csv
import math

NUM_FEATURES = 2 #2 features (x,y)
NUM_CATEGORIES = 3
examples = [[] for i in range(NUM_CATEGORIES)]

def outer_product(v1,v2): #outer product of 2 vectors
    #reformats vectors so outer product can be calculated more easily
    v1 = [v1[i][0] for i in range(len(v1))]
    v2 = v2[0]
    return [[v1[i]*v2[j] for j in range(len(v2))] for i in range(len(v1))]

def transpose(m1): #transposes matrix
    return [[m1[j][i] for j in range(len(m1))] for i in range(len(m1[0]))]

def matrix_operation(matrix,value,operation): #performs operation between each element in a matrix and a value
    return list(map(lambda x:list(map(lambda y:eval(f"{y}{operation}{value}"),x)),matrix))

def dot_product(v1,v2): #dot product of vectors
    #only works for vectors
    assert len(v1)==len(v2)
    return sum([v1[i]*v2[i] for i in range(len(v1))])

def matrix_mult(m1,m2):
    #dot product of 2 matrices
    m2 = list(zip(*m2))
    return [[dot_product(m2[i],m1[j]) for i in range(len(m2))] for j in range(len(m1))]

def matrix_add(m1,m2):
    #adds matrices together
    return [[m1[i][ii]+m2[i][ii]  for ii in range(len(m1[0]))] for i in range(len(m1))]

def vector_subtract(v1,v2):
    #subtracts two vectors (only vectors)
    assert len(v1) == len(v2)
    return [v1[i]-v2[i] for i in range(len(v1))]

def vector_to_matrix(v): #converts vector to matrix
    return [[v[i],] for i in range(len(v))]

def gaussian(mean): #generates the gaussian function for each category based on its mean
    # 1/sqrt((2pi)^num_features * |Sigma|)  *  exp(-0.5 * (x-mu)^T ⊗ (Sigma^-1 ⊗ (x-mu)) )
    def inner(x):
        deviation = vector_to_matrix(vector_subtract(x,mean))
        quadratic = matrix_mult(transpose(deviation),matrix_mult(np.linalg.inv(COVARIANCE).tolist(),deviation))[0][0]
        #print(quadratic)
        return 1/((2*math.pi)**(NUM_FEATURES/2) * np.linalg.det(np.array(COVARIANCE))**2) * math.exp(-0.5*quadratic)
    return inner

with open('generated_data.csv','r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data = (float(row[0]),float(row[1]))
        examples[int(row[2])].append(data)
    NUM_EXAMPLES = sum(len(i) for i in examples)

MEANS = [[sum(example[0] for example in category)/len(category), sum(example[1] for example in category)/len(category)] for category in examples] #mean value for each category
COVARIANCE = [[0 for j in range(NUM_FEATURES)] for i in range(NUM_FEATURES)] #sets covariance to zero
for category in range(NUM_CATEGORIES):
    for example in examples[category]:
        deviation_vec = vector_subtract(example, MEANS[category])
        deviation = vector_to_matrix(deviation_vec)
        outer = outer_product(deviation, transpose(deviation))
        #assuming all categories have the same covariance
        #adds each example's deviation to the covariance matrix
        #(x-mu) ⊗ (x-mu)^T
        COVARIANCE = matrix_add(COVARIANCE, outer)
#average value of covariance
COVARIANCE = matrix_operation(COVARIANCE,NUM_EXAMPLES,'/')
#print(COVARIANCE)

CAT_PROBS = [len(i)/NUM_EXAMPLES for i in examples] #probability of any input being in any of the categories
FUNCTIONS = [gaussian(MEANS[i]) for i in range(NUM_CATEGORIES)] #list of gaussian functions for each category

def predict(x):#predicts
    #doesnt work with extreme numbers because rounding errors make the gaussian values 0
    assert len(x)==NUM_FEATURES
    probx = sum(FUNCTIONS[i](x)*CAT_PROBS[i] for i in range(NUM_CATEGORIES)) #p(x) used in probability calculation
    predictions = [CAT_PROBS[i]*FUNCTIONS[i](x)/probx for i in range(NUM_CATEGORIES)]
    print(f"This point's probably in \33[1;32mcategory {predictions.index(max(predictions))}\33[0m, im like \33[1;32m{max(predictions)*100:.2f}%\33[0m sure")

predict([5,0])