# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:36:15 2021

@author: 18433
"""
import re
import numpy as np
import copy
from functools import reduce
import math
import glob
categories=['animal','animation','dance','fashion','food','game','kichiku','knowledge','life','music','tech']

def read_features(num_pictures,cats):
    # read results of yolov3
    objects = [];areas = [];scores=[]
    for cat in cats:
        pos = categories.index(cat)
        with open ('./result/classes.txt','r') as f1:
            for line in f1.readlines()[pos*100:pos*100+num_pictures]:
                match = re.findall(r'\d+', line)
                objects.append(list(map(int,match)))    
        with open('./result/areas.txt','r') as f2:
            for line in f2.readlines()[pos*100:pos*100+num_pictures]:
                # calculate relative areas of boxes
                match = re.findall(r"\d+\.?\d*",line) 
                b = list(map(float,match))
                b = [x /(416*416) for x in b]
                areas.append(b)    
        with open ('./result/scores.txt','r') as f3:
            for line in f3.readlines()[pos*100:pos*100+num_pictures]:
                #print(f.readline())
                match = re.findall(r"\d+\.?\d*", line)
                scores.append(list(map(float,match)))      
    f1.close()
    f2.close()
    f3.close()
    
    # read results of ocr
    text = np.zeros([num_pictures*len(cats)])
    for i,cat in enumerate(cats):
        path = glob.glob('./ocr_result/'+cat+'_prop.txt')
        with open (path[0],'r') as f:
            for j in range(num_pictures):
                text[i*num_pictures + j] = f.readline()
        f.close()
    
    return objects,areas,scores,text

def unique(objects):
    elements = []
    for obj in objects:
        elements =list(set(elements).union(set(obj)))
    return elements

def remove_hard(objects,areas,scores,text):
    objects_new = copy.deepcopy(objects) 
    areas_new = copy.deepcopy(areas)
    scores_new = copy.deepcopy(scores)
    text_new = copy.deepcopy(text)
    index = []
    count = 0
    for i,obj in enumerate(objects):
        if not obj and text[i]==0:
            del objects_new[i-count]
            del areas_new[i-count]
            del scores_new[i-count]
            text_new = np.delete(text_new,i-count)
            count += 1
            index.append(i+1)
            
    return objects_new,areas_new,scores_new,text_new,index
            
def remove_labels(index,labels):
    labels_new = copy.deepcopy(labels)
    for i,ix in enumerate(index):
        del labels_new[ix-i]
    return labels_new


def cal_idf(objects,elements,text):
    obj_unique = unique(objects)
    document = len(objects)
    def f(x,y):
        if type(x)==type(y):
            return len(x)+len(y)
        else:
            return x+len(y)
    words = reduce(f,objects)
    
    text_no = np.count_nonzero(text)
    
    # idf和iwf原始实现
    word_count = np.zeros([len(obj_unique)])
    document_count = np.zeros([len(obj_unique)])
    for obj in objects:
        uni = list(set(obj))
        for i in uni:
            word_count[obj_unique.index(i)] += obj.count(i)
            document_count[obj_unique.index(i)] += 1
    
    # merge the labels        
    wc_t = np.zeros([len(elements)])
    dc_t = np.zeros([len(elements)])
    for i,j in enumerate(obj_unique):
            # person
            if j==0 : 
                wc_t[0] += word_count[i]
                dc_t[0] += document_count[i]
            # traffic
            elif j>1 and j<14:
                wc_t[1] += word_count[i]
                dc_t[1] += document_count[i]
            # animal
            elif j>=14 and j<24 :
                wc_t[2] += word_count[i]
                dc_t[2] += document_count[i]
            # package
            elif j>=24 and j<28 :
                wc_t[3] += word_count[i]
                dc_t[3] += document_count[i]  
            # sports
            elif j>=28 and j<39 :
                wc_t[4] += word_count[i]
                dc_t[4] += document_count[i]
            # food and cookware and tableware
            elif (j>=39 and j<56) or( j>=68 and j<73) :
                wc_t[5] += word_count[i]
                dc_t[5] += document_count[i]
            # furniture
            elif j>=56 and j< 62  :
                wc_t[6] += word_count[i]
                dc_t[6] += document_count[i]
            # tech
            elif j>=62 and j<68 :
                wc_t[7] += word_count[i]
                dc_t[7] += document_count[i]
            else:
                wc_t[8] += word_count[i]
                dc_t[8] += document_count[i]
            wc_t[9] = dc_t[9] = text_no
    
    # calculate
    idf = [math.log(words/(x+1)) for x in wc_t]
    iwf = [math.log(document/(x+1)) for x in dc_t]
    #print(idf,iwf)
    return idf,iwf
    
def get_vectors(objects,elements,areas,scores,text):
    vectors = np.zeros((len(objects),len(elements)))
    idf,iwf = cal_idf(objects, elements,text)
    for i,obj in enumerate(objects):
        for index,j in enumerate(obj):
            # person
            if j==0 : 
                vectors[i][0] += areas[i][index]*scores[i][index]*idf[0]
            # traffic
            elif j>1 and j<14:
                vectors[i][1] += areas[i][index]*scores[i][index]*idf[1]
            # animal
            elif j>=14 and j<24 :
                vectors[i][2] += areas[i][index]*scores[i][index]*idf[2]
            # package
            elif j>=24 and j<28 :
                vectors[i][3] += areas[i][index]*scores[i][index] *idf[3] 
            # sports
            elif j>=28 and j<39 :
                vectors[i][4] += areas[i][index]*scores[i][index]*idf[4]
            # food and cookware and tableware
            elif (j>=39 and j<56) or( j>=68 and j<73) :
                vectors[i][5] += areas[i][index]*scores[i][index]*idf[5]
            # furniture
            elif j>=56 and j< 62  :
                vectors[i][6] += areas[i][index]*scores[i][index]*idf[6]
            # tech
            elif j>=62 and j<68 :
                vectors[i][7] += areas[i][index]*scores[i][index]*idf[7]
            else:
                vectors[i][8] += areas[i][index]*scores[i][index]*idf[8]
        vectors[i][9] = text[i] * idf[9]
            
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



