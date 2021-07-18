# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:36:15 2021

@author: 18433
"""
import re
import numpy as np

def read_features(num_pictures):
    objects = []
    with open ('./result/classes.txt','r') as f:
        for i in range(num_pictures):
            match = re.findall(r'\d+', f.readline())
            objects.append(match)
    f.close()
    return objects

def unique(objects):
    elements = []
    for obj in objects:
        elements = list(set(elements).union(set(obj)))
    return elements

def get_vectors(objects,elements):
    vectors = np.zeros((len(objects),len(elements)))
    for i,obj in enumerate(objects):
        for j in obj:
            if int(j)<15 and vectors[i][0]<2:
                vectors[i][0] += 1
            elif int(j)<43 and vectors[i][1]<2:
                vectors[i][1] += 1
            elif int(j)<56 and vectors[i][2]<2:
                vectors[i][2] +=1
            else:
                if vectors[i][3]==0:
                    vectors[i][3] +=1 
    return vectors
'''
objects = read_features(15)
#elements = unique(objects)  # elements分为4类person,animal,food,other
elements = ['person','animal','food','other']
vectors = get_vectors(objects, elements)
print(vectors)
'''