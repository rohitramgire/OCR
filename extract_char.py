# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:05:32 2022

@author: u1374014
"""

import cv2
from scipy import ndimage
import os
import numpy as np
import matplotlib.pyplot as plt
root = os.getcwd()

def arrange_chars(gray,output,count):
    (numLabels, labels, stats, centroids) = output
    lst_centroid = []
    hashmap = {}
    # loop over the number of unique connected component labels
    for i in range(1, numLabels):
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        (cX, cY) = centroids[i]
        hashmap[str(cX)+str(cY)] = (x,y,w,h,i)
        lst_centroid.append((cX,cY))
        
    lst_centroid = sorted(lst_centroid , key=lambda k: [k[0], k[1]])
    for j in lst_centroid:
        i = hashmap[str(j[0])+str(j[1])]
        
        output = gray.copy()
        x = i[0]
        w = i[2]
        y = i[1]
        h = i[3]
        j = i[4]

        componentMask = (labels == j).astype("uint8") * 255
        componentMask = componentMask[y:y+h,x:x+w]
        filename = str(count)+'.png'
        
        cv2.imwrite(os.path.join(root,'extract_text', filename), componentMask)

        count = count+1
        
    for i in os.listdir(os.path.join(root,'extract_text')):
        
        
        img = cv2.imread(os.path.join('extract_text', i))
        img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,0)
        if((img.shape[0]<28) and (img.shape[1]<28)):
            old_image_height, old_image_width, channels = img.shape
            new_image_width = 28
            new_image_height = 28
            color = (0,0,0)
            result = np.full((new_image_height,new_image_width,channels),color, dtype=np.uint8)
            
            
    
            # compute center offset
            x_center = (new_image_width - old_image_width) // 2
            y_center = (new_image_height - old_image_height) // 2
    
            # copy img image into center of result image
            result[y_center:y_center+old_image_height, 
                    x_center:x_center+old_image_width] = img
            img = result
            
        else:
            img = cv2.resize(img, (28,28))

        cv2.imwrite(os.path.join(root,'extract_text_output', i), img)
        
        
    return count
                



def extract_text(file):    

    directory = os.listdir(os.path.join(root,'extract_text'))
    
    if(len(directory)!=0):
        for f in os.listdir(os.path.join(root,'extract_text')):
            os.remove(os.path.join('extract_text',f))
            
    directory = os.listdir(os.path.join(root,'extract_text_output'))
    if(len(directory)!=0):
        for f in os.listdir(os.path.join(root,'extract_text_output')):
            os.remove(os.path.join(root,'extract_text_output',f))
    directory = os.listdir(os.path.join(root,'test'))
    if(len(directory)!=0):
        for f in os.listdir(os.path.join(root,'test')):
            os.remove(os.path.join(root,'test',f))
    
    #Concatenate file with root path
    image = cv2.imread(os.path.join(root,'test_images', file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = ndimage.gaussian_filter(gray, 1)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (151, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    plt.imshow(morph)
    plt.show()
    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    
    
    # Draw contours
    result = image.copy()
    count = 0
    for c in cntrs[::-1]:
        box = cv2.boundingRect(c)
        x,y,w,h = box
        filename = str(count)+".png"
        cv2.imwrite(os.path.join(root,'test',filename),result[y:y+h,x:x+w])
        count = count+1
        
    # #Read the countoured files
    count_ = 0
    for i in os.listdir(os.path.join(root,'test')):
            
        image = cv2.imread(os.path.join(root,'test', i))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = ndimage.gaussian_filter(gray, 1)
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        
    ### Arrange characters into proper sentences
        count_ = arrange_chars(gray,output,count_)
        

if __name__ == "__main__":
    
    extract_text(file)
    