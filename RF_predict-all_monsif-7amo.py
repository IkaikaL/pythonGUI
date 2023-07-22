# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 20:14:54 2022

@author: mhassa9
"""

import numpy as np
import cv2
import pandas as pd
import skimage.filters
import skimage.io
import skimage.morphology
import skimage.exposure
import os

from datetime import datetime
start_time = datetime.now()

def feature_extraction (img):

    img2 = img.reshape(-1)
    df = pd.DataFrame()
    df['Original Image'] = img2
    
    #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                
                    
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
    ########################################
    #Gerate OTHER FEATURES and add them to the data frame
                    
    #CANNY EDGE
    # =============================================================================
    # edges = cv2.Canny(img, 100,200)   #Image, min and max values
    # edges1 = edges.reshape(-1)
    # df['Canny Edge'] = edges1 #Add column to original dataframe
    # =============================================================================
    
    from skimage import feature
    
    edges = feature.canny(img)
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe
    
    return df

#################################################################

import glob
import pickle
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Process
import sys
from pathlib import Path



filename = "XCT_421-A_300um-final"

#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))



#for file in glob.glob(path):

def predict_scan(input1):
    img = skimage.io.imread(r"" + str(Path().absolute()) + "\\Masked-All\\" + str(input1))
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    
    segmented = result.reshape((img.shape))
    imgg = np.array(segmented, dtype=np.uint16) # This line only change the type, not values

    plt.imshow(segmented, cmap ='plasma')
    
    name = input1.split('M-C-coreA_')
    cv2.imwrite(r"" + str(Path().absolute()) + "\\Segmented-All\\" + name[1], imgg)      #cmap ='plasma'
 
    
if __name__ == '__main__':

    start_time = datetime.now()
    if not os.path.exists(str(Path().absolute()) + "\\Segmented-All"):
        os.makedirs(str(Path().absolute()) + "\\Segmented-All")
    x = os.listdir(str(Path().absolute()) + "\\Masked-All\\")
    lst = list(x)
    processes = []
    for i in lst:
        p = mp.Process(target=predict_scan, args=(i,))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
        
        
    end_time = datetime.now()
    #print('Duration: {}'.format(end_time - start_time))

print("done")
