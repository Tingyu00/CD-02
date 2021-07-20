# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:36:15 2021

@author: 18433
"""
import re
import numpy as np
import copy

def read_features(num_pictures):
    objects = [];areas = []
    with open ('./result/classes.txt','r') as f:
        for i in range(num_pictures):
            match = re.findall(r'\d+', f.readline())
            objects.append(list(map(int,match)))
    f.close()
    with open('./result/areas.txt','r') as f:
        for i in range(num_pictures):
            # calculate relative areas of boxes
            match = re.findall(r"\d+\.?\d*", f.readline()) 
            b = list(map(float,match))
            b = [x /(416*416) for x in b]
            areas.append(b)
    return objects,areas

def unique(objects):
    elements = []
    for obj in objects:
        elements =list(set(elements).union(set(obj)))
    return elements

def remove_hard(objects,areas):
    objects_new = copy.deepcopy(objects) 
    areas_new = copy.deepcopy(areas)
    index = []
    count = 0
    for i,obj in enumerate(objects):
        if not obj:
            del objects_new[i-count]
            del areas_new[i-count]
            count += 1
            index.append(i)
    #print(objects_new)
    return objects_new,areas_new,index
            
def remove_labels(index,labels):
    labels_new = copy.deepcopy(labels)
    for i,ix in enumerate(index):
        del labels_new[ix-i]
    return labels_new

def get_vectors(objects,elements,areas):
    vectors = np.zeros((len(objects),len(elements)))
    for i,obj in enumerate(objects):
        for index,j in enumerate(obj):
            # person
            if j==0 : 
                vectors[i][0] += areas[i][index]
            # traffic
            elif j>1 and j<14:
                vectors[i][1] += areas[i][index]
            # animal
            elif j>=14 and j<24 :
                vectors[i][2] += areas[i][index]
            # package
            elif j>=24 and j<28 :
                vectors[i][3] += areas[i][index]  
            # sports
            elif j>=28 and j<39 :
                vectors[i][4] += areas[i][index]
            # food and cookware and tableware
            elif (j>=39 and j<56) or( j>=68 and j<73) :
                vectors[i][5] += areas[i][index]
            # furniture
            elif j>=56 and j< 62  :
                vectors[i][6] += areas[i][index]
            # tech
            elif j>=62 and j<68 :
                vectors[i][7] += areas[i][index]
            else:
                if vectors[i][8]==0:
                    vectors[i][8] += areas[i][index]
    return vectors
'''
objects,areas = read_features(300)
#elements = unique(objects)
elements = ['person','traffic','animal','package','sports','food','furniture','tech','other']
objects_new,areas_new,index = remove_hard(objects, areas)
#vectors = get_vectors(objects, elements,areas)
#print(vectors.shape,vectors)
'''
