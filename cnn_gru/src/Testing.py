
"""
============================================================
Project Name: Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation
Reference paper: ...
File Name: "Testing.py"
Description: This file contains the functions for evaluating the neural network (previously trained) on its test data. 
The test samples are obtained as described in the reference paper cited above (see Section 4.1). 
A function for visualizing results by comparing predictions and simulated attenuation is provided.
Author: Alice JANELA CAMEIJO
Date Created: 2025-01-15
Last Modified: 2025-01-15
License: ...
Version: 1.0.0
============================================================
"""


########################
# IMPORTs + PARAMETERs 
########################
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Run code
path = "/.../" # your path to the work space
os.makedirs(path+"Predictions/", exist_ok=True)


########################
# PREDICTIONs 
########################
# --------------------------------------------------------------------------------------------------------------------------------------------------          
# name: compute_horizontally_averaged_c_ratios_maxima         
# input param: 'c_ratios', 'inf', 'sup', 'scaler_c_ratios'
# do: restrain each 'c_ratio' slice in a given atmospheric layer (between 'inf'-'sup' km altitude)       
#     + compute mean c_ratios in this layer     
# output: 'horizontally_averaged_c_ratio_maxima'       
# --------------------------------------------------------------------------------------------------------------------------------------------------       
def compute_horizontally_averaged_c_ratios_maxima(c_ratios, inf, sup, scaler_c_ratios):
    # De-standardize c_ratios slices                                                        
    destandardized_c_ratios = np.reshape(c_ratios, (c_ratios.shape[0], c_ratios.shape[1]*c_ratios.shape[2]))
    destandardized_c_ratios = np.reshape(scaler_c_ratios.inverse_transform(destandardized_c_ratios), (c_ratios.shape[0], c_ratios.shape[1], c_ratios.shape[2]))

    # Restrain each c_ratio slice in a specific atmospheric layer                   
    c_ratio_maxima = np.empty((c_ratios.shape[0], 1, c_ratios.shape[2]))
    horizontally_averaged_c_ratio_maxima = np.empty((c_ratios.shape[0], 1))
    for sample in range(c_ratios.shape[0]):
        for column in range(c_ratios.shape[2]):
            cmpt = 0
            c_ratios_restrained = np.empty([sup-inf,])
            for line in range(inf,sup):
                c_ratios_restrained[cmpt] = destandardized_c_ratios[sample,line,column]
                cmpt+=1
            c_ratio_maxima[sample,0,column] = np.max(c_ratios_restrained)

    # Compute horizontal mean                  
    for sample in range(c_ratios.shape[0]):
        horizontally_averaged_c_ratio_maxima[sample,:] = np.mean(c_ratio_maxima[sample,0,:])
    return horizontally_averaged_c_ratio_maxima


# --------------------------------------------------------------------------------------------------------------------------------------------------
# name: make_testing_predictions
# input param: 'testing_c_ratios', 'testing_freqs', 'testing_PEs', 'cnn_gru', 'scaler_c_ratios', 'scaler_pes'
# do : make predictions on the testing samples + comparisons with simulated PEs (labels)
# --------------------------------------------------------------------------------------------------------------------------------------------------
def make_testing_predictions(nb_columns_PE, distance_PE, altitude, testing_c_ratios, testing_freqs, testing_PEs, cnn_gru, scaler_c_ratios, scaler_pes):
    # Prepare testing datasets
    testing_freqs = testing_freqs[:, np.newaxis, np.newaxis]
    freqs_test = tf.repeat(testing_freqs, repeats=5, axis=1)
    horizontally_averaged_c_ratios_maxima_30_60 = compute_horizontally_averaged_c_ratios_maxima(testing_c_ratios, (np.abs(altitude-30)).argmin(), (np.abs(altitude-60)).argmin(), scaler_c_ratios) 
 
    # Make testing predictions
    nb_samples = testing_c_ratios.shape[0]
    predicted_TLs = np.empty((nb_samples, testing_PEs.shape[1]))
    PE_simulations = np.empty((nb_samples, testing_PEs.shape[1]))
    for sample in range(nb_samples):
        # Make a prediction (destandardized)
        c_ratio = (testing_c_ratios[sample]).reshape(1, testing_c_ratios.shape[1], testing_c_ratios.shape[2])
        freq = np.reshape(freqs_test[sample], (1, 5, 1))
        prediction = np.reshape(np.array((cnn_gru([c_ratio, freq]))), (1, nb_columns_PE))
        destandardized_prediction = (scaler_pes.inverse_transform(prediction)).reshape(nb_columns_PE,) 
        predicted_TLs[sample] = destandardized_prediction

        # Get the associated PE simulation (destandardized)
        PE = testing_PEs[sample].reshape(1, nb_columns_PE)
        destandardized_PE = scaler_pes.inverse_transform(PE).reshape(nb_columns_PE,)	
        PE_simulations[sample] = destandardized_PE

        # Plot the comparison
        color=''
        if(horizontally_averaged_c_ratios_maxima_30_60[sample,:] > 1): # upwind conditions
            color='b'
        else: # downwind conditions
            color='k'
        plt.figure(figsize=(10.5,8.5), dpi=300)
        plt.plot(distance_PE, destandardized_PE, color=color, label='PE simulation (label)')
        plt.plot(distance_PE, destandardized_prediction, color='red', label='Predicted TL')
        plt.title("Simulated / predicted TL at f = "+str(np.around(freqs_test[sample,0,0],1))+" Hz \nhorizontal average of c$_{ratio}$ maxima (30-60 km) = " + str(np.round_(horizontally_averaged_c_ratios_maxima_30_60[sample,0], decimals = 1)), fontsize=23)
        plt.xlabel('Distance [km]', fontsize=23)
        plt.ylabel('Transmission loss [dB]', fontsize=23)
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.ylim(-120,1)
        plt.grid() 
        plt.legend(loc="lower left", fontsize=23)
        plt.savefig(path+"Predictions/sample_"+str(sample+1)+".pdf", format="pdf", bbox_inches="tight")
        plt.close()

    return horizontally_averaged_c_ratios_maxima_30_60, predicted_TLs, PE_simulations
       

########################
# FIGUREs 
########################
# --------------------------------------------------------------------------------------------------------------------------------------------------
# name: prepare_data_for_plot
# input param: 'altitude', 'testing_c_ratios', 'scaler_c_ratios', 'tab'
# do: sort the predicted TLs or expected PE labels according to the horizontally_averaged_c_ratios_maxima (y-axis)
# output: 'sorted_tab', 'y_axis'
# --------------------------------------------------------------------------------------------------------------------------------------------------
def prepare_data_for_plot(altitude, testing_c_ratios, scaler_c_ratios, horizontally_averaged_c_ratios_maxima_30_60, tab):
    tab_c_ratios = np.concatenate((horizontally_averaged_c_ratios_maxima_30_60, tab), axis=1) 
    sorted_tab = tab_c_ratios[tab_c_ratios[:,0].argsort()]
    y_axis = sorted_tab[:,0:1] 	                      
    return sorted_tab, np.reshape(y_axis, (y_axis.shape[0],))  


def plot_pcolormap(distance_PE, nb_columns_PE, y_axis, data, ttl):
    cmp = plt.pcolor(distance_PE, y_axis, data[:, 1:(nb_columns_PE+1)], cmap="viridis", vmin=-80, vmax=0)
    contours = plt.contour(distance_PE, y_axis, gaussian_filter(data[:, 1:(nb_columns_PE+1)], 32), levels=[-80, -60, -40, -20], colors='w')
    plt.xlabel('Distance [km]', fontsize=23)
    plt.ylabel('Horizontal average of c$_{ratio}$ maxima (30-60 km)', fontsize=23)
    plt.yticks(fontsize = 17)
    plt.xticks(fontsize = 17)
    plt.clabel(contours, inline=True, fontsize=23, colors='gold')
    cb = plt.colorbar(cmp)
    cb.set_label(label='Transmission loss [dB]', size=23)
    cb.ax.tick_params(labelsize=17)
    plt.title(ttl, fontsize=25)


def plot_results(distance_PE, nb_columns_PE, altitude, testing_c_ratios, scaler_c_ratios, horizontally_averaged_c_ratios_maxima_30_60, predicted_TLs, PE_simulations):
    PEs, y_axis = prepare_data_for_plot(altitude, testing_c_ratios, scaler_c_ratios, horizontally_averaged_c_ratios_maxima_30_60, PE_simulations)
    TLs, y_axis = prepare_data_for_plot(altitude, testing_c_ratios, scaler_c_ratios, horizontally_averaged_c_ratios_maxima_30_60, predicted_TLs)

    # Plot global results (on the whole testing dataset)
    plt.figure(figsize=(20,8), dpi=300)
    plt.subplot(1, 2, 1)
    plot_pcolormap(distance_PE, nb_columns_PE, y_axis, PEs ,"PE simulations (labels)")
    plt.subplot(1, 2, 2)
    plot_pcolormap(distance_PE, nb_columns_PE, y_axis, TLs, "Predicted TLs")
    plt.savefig(path+'Predictions/Testing_performances.pdf', format="pdf", bbox_inches="tight")
    plt.close()













