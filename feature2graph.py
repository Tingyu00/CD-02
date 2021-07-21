# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:36:15 2021

@author: 18433
"""
import re
import numpy as np
import copy

def read_features(num_pictures):
    objects = [];areas = [];scores=[]
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
    f.close()
    with open ('./result/scores.txt','r') as f:
        for i in range(num_pictures):
            #print(f.readline())
            match = re.findall(r"\d+\.?\d*", f.readline())
            scores.append(list(map(float,match)))
    f.close() 
    return objects,areas,scores

def unique(objects):
    elements = []
    for obj in objects:
        elements =list(set(elements).union(set(obj)))
    return elements

def remove_hard(objects,areas,scores):
    objects_new = copy.deepcopy(objects) 
    areas_new = copy.deepcopy(areas)
    scores_new = copy.deepcopy(scores)
    index = []
    count = 0
    for i,obj in enumerate(objects):
        if not obj:
            del objects_new[i-count]
            del areas_new[i-count]
            del scores_new[i-count]
            count += 1
            index.append(i)
    #print(objects_new)
            
    return objects_new,areas_new,scores_new,index
            
def remove_labels(index,labels):
    labels_new = copy.deepcopy(labels)
    for i,ix in enumerate(index):
        del labels_new[ix-i]
    return labels_new

def get_vectors(objects,elements,areas,scores):
    vectors = np.zeros((len(objects),len(elements)))
    for i,obj in enumerate(objects):
        for index,j in enumerate(obj):
            #print(len(objects[i]),len(areas[i]),len(scores[i]),scores[i])
            # person
            if j==0 : 
                vectors[i][0] += areas[i][index]*scores[i][index]
            # traffic
            elif j>1 and j<14:
                vectors[i][1] += areas[i][index]*scores[i][index]
            # animal
            elif j>=14 and j<24 :
                vectors[i][2] += areas[i][index]*scores[i][index]
            # package
            elif j>=24 and j<28 :
                vectors[i][3] += areas[i][index]*scores[i][index]  
            # sports
            elif j>=28 and j<39 :
                vectors[i][4] += areas[i][index]*scores[i][index]
            # food and cookware and tableware
            elif (j>=39 and j<56) or( j>=68 and j<73) :
                vectors[i][5] += areas[i][index]*scores[i][index]
            # furniture
            elif j>=56 and j< 62  :
                vectors[i][6] += areas[i][index]*scores[i][index]
            # tech
            elif j>=62 and j<68 :
                vectors[i][7] += areas[i][index]*scores[i][index]
            else:
                vectors[i][8] += areas[i][index]*scores[i][index]
    return vectors

def normalize(vectors):
    vect_norm = []
    for v in vectors:
        v = np.array(v)
        vect_norm.append(list(normalization(v)))
    return vect_norm
    
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    


