# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:49:17 2022

@author: mhassa9
"""


import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import sys
import numpy as np
import cv2

from pathlib import Path

from scipy import stats

import skimage

workpath = r"" + str(Path().absolute()) + "\Masked-All"
print(workpath)

#C:\Users\mhassa9\CT Scan - work\Correction\Core2_AT\15-20
# 130, 270, 410, 550, 690, 830

savepath = r"" + str(Path().absolute())
print(savepath)
for i in [500]:
    
    filename = 'M-C-coreA_300um_0' + str(i) + '.tif'
    
    im_phase = skimage.io.imread(workpath + "\\" + filename)
    
    img2 = im_phase.reshape((-1, 1))  #-1 reshape means, in this case MxN
    
    
    from sklearn.mixture import GaussianMixture as GMM
    
    #covariance choices, full, tied, diag, spherical
    # 
    k = 5
    
    gmm_model = GMM(n_components=k, random_state=2, covariance_type='full').fit(img2)  #tied works better than full, random_state=0
    
      
        
    
    
    gmm_labels = gmm_model.predict(img2)
    
    import pandas as pd
    
    from pandas import DataFrame
    
    
    d = pd.DataFrame()
    d['labels'] = gmm_labels
    
    #df['column name'] = df['column name'].replace(['old value'],'new value')
    
    
    d['labels'] = d['labels'].replace([0],22)
    d['labels'] = d['labels'].replace([1],666)
    d['labels'] = d['labels'].replace([2],33)
    d['labels'] = d['labels'].replace([3],11)
    #d['labels'] = d['labels'].replace([4],33)
   
    d['labels'] = d['labels'].replace([666],0)
    d['labels'] = d['labels'].replace([11],1)
    d['labels'] = d['labels'].replace([22],2)
    d['labels'] = d['labels'].replace([33],3)
    #d['labels'] = d['labels'].replace([44],4)
    
    
    gmm_labels = d['labels']
    
    
    
    #Put numbers back to original shape so we can reconstruct segmented image
    original_shape = im_phase.shape
    segmented = gmm_labels.values.reshape(original_shape[0], original_shape[1])
    #values.
    
    
    imgg = np.array(segmented, dtype=np.uint16) # This line only change the type, not values
    #imgg /= 256 # Now we get the good values in 16 bit format
    
    cv2.imwrite(savepath + "\\" + "S" + "-" + filename, imgg)
    
    #name = filename.replace('tif', 'jpg')
    #plt.imsave(workpath + "\\" + "S" + str(l) + "-" + name, imgg, cmap ='plasma')
        
    #################################################################
    
    
    data = img2.ravel()
    #data = data[data != 0]
    #data = data[data != 1]  #Removes background pixels (intensities 0 and 1)
    
    
    #gmm = gmm_model.fit(X=np.expand_dims(data,1))
    gmm_x = np.linspace(1,65536,65536)
    gmm_y = np.exp(gmm_model.score_samples(gmm_x.reshape(-1, 1)))
    
    gmm_model.means_
    
    gmm_model.covariances_
    
    gmm_model.weights_
    
    print (gmm_model.means_)
    print ('-.-.-.-.-.-.-.-.-.-')
    print (gmm_model.weights_)
    print ('-.-.-.-.-.-.-.-.-.-')
    print (gmm_model.covariances_)
    
    
    print ('--------------------------')
    
    
# =============================================================================
#     
#     # Plot histograms and gaussian curves
#     fig, ax = plt.subplots()
#     ax.hist(img2.ravel(),500, density=True, stacked=True, histtype = 'bar')
#     ax.plot(gmm_x, gmm_y, color="crimson", lw=2, label="GMM")
#     
#     ax.set_ylabel("Frequency")
#     ax.set_xlabel("Corrected Pixel Intensity")
#     ax.set_yscale('log')
#     minor_locator = AutoMinorLocator(2)
#     ax.xaxis.set_minor_locator(minor_locator)
#     ax.tick_params(axis="x",direction="in")
#     ax.tick_params(axis='x', which='minor', direction="in")
#     ax.tick_params(axis="y",direction="in")
#     ax.tick_params(axis='y', which='minor', direction="in")
#     
#     plt.legend()
#     plt.grid(False)
#     plt.xlim([0, 65536])
#     #plt.yscale = "log"
#     #plt.ylim(0.000001, 1)
#     plt.ylim( (10**-8,10**-3) )
#     #ax.ticklabel_format(useOffset=True)
#     plt.savefig(savepath + "\\" + "Hist_" + str(i) + ".png", dpi = 500)
#     #plt.show()
#     
#     
#     for m in range(gmm_model.n_components):
#         pdf = gmm_model.weights_[m] * stats.norm(gmm_model.means_[m, 0],
#                                                np.sqrt(gmm_model.covariances_[m, 0])).pdf(gmm_x.reshape(-1,1))
#         
#         if gmm_model.means_[m, 0] > 0:
#         
#             fig, ax = plt.subplots()
#             ax.hist(img2.ravel(),200, density=True, stacked=True, histtype = 'bar')
#             ax.plot(gmm_x, gmm_y, color="crimson", lw=2, label="GMM")
#             ax.set_yscale('log')
#             minor_locator = AutoMinorLocator(2)
#             ax.xaxis.set_minor_locator(minor_locator)
#             ax.tick_params(axis="x",direction="in")
#             ax.tick_params(axis='x', which='minor', direction="in")
#             ax.tick_params(axis="y",direction="in")
#             ax.tick_params(axis='y', which='minor', direction="in")
#             plt.fill(gmm_x, pdf, facecolor='gray',
#                      edgecolor='none')
#             plt.xlim(0, 65536)
#             plt.ylim( (10**-8,10**-3) )
#             plt.savefig(savepath + "\\" + "Fig_" + str(m) + "_" + str(i) + ".jpg", dpi = 500)
#         else:
#             
#             print ('mean = 0')
# =============================================================================
            
# =============================================================================
#     bic_value = gmm_model.bic(img2)
#     n_components = np.arange(1,11)
#     n_compo = np.arange(1,11)
#     gmm_models = [GMM(n, covariance_type='full').fit(img2) for n in n_components]
# 
#                
#     aic_value = gmm_model.aic(img2)
#     gmm_modelss = [GMM(n, covariance_type='full').fit(img2) for n in n_compo]
#     fig, ax = plt.subplots()
#     ax.plot(n_components, [m.bic(img2) for m in gmm_models], label='BIC')
#     ax.plot(n_components, [m.aic(img2) for m in gmm_modelss], label='AIC')
#     plt.legend()
#     plt.xlim([0, 10])
# =============================================================================




