
"""
============================================================
Project Name: Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation
Reference paper: ...
File Name: "Data/Preprocessing/outputs.py"
Description: This file contains the code needed to preprocess the output data and obtain the ground-level transmission loss in order to feed the convolutional recurrent neural network.
The output data are pressure field simulated with the "ePape" solver from the National Center for Physical Acoustics (University of Mississippi, Waxler et al., 2021).
See Sections 2 of the reference paper cited above for additional details.
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import RegularGridInterpolator as rgi
from statsmodels.nonparametric.smoothers_lowess import lowess

# Run code
path = "[/.../]" # your path to the work space

nb_lines_c_ratio_slice = 434
altitudes = np.linspace(0., 129.9, nb_lines_c_ratio_slice) # pas de 0.3 km altitude
nb_columns_PE = 800 
distances = np.linspace(1., 4000., nb_columns_PE) # pas de 5 km distance


########################
# PRE-PROCESS 
########################
# --------------------------------------------------------------------------------------------------------------------------------------------------
# name: compute_interpolated_ground_level_transmission_loss
# do: load the pressure field simulated with the "ePape" solver
#     + compute the ground-level transmission loss in decibel
#     + interpolate the ground-level transmission loss with the expected format to feed the convolutional recurrent neural network
#     + plot the result
# --------------------------------------------------------------------------------------------------------------------------------------------------
def compute_interpolated_ground_level_transmission_loss():
    # Get the 2D transmission loss coming from the "ePape" solver
    map = pd.read_csv("ePape_PE/2D_PE_f0.4_azi_270.pe", sep='\s+', header=None)
    map.columns = ['x', 'z', 'Re-p', 'Im-p']
    dist, alt = map.x.unique(), map.z.unique()
    if(np.shape(np.array(np.sqrt(map['Re-p']**2 + map['Im-p']**2).values)) == (dist.size*alt.size,)):
        TL_2d = np.sqrt(map['Re-p']**2 + map['Im-p']**2).values.reshape(dist.size, alt.size).T
        
        # Interpolate the 2D transmission loss
        TL_2d_interp = rgi((alt, dist), TL_2d, bounds_error=False, fill_value=None)
        alt_dist = np.meshgrid(altitudes, distances, indexing='ij')
        alt_dist_list = np.reshape(alt_dist, (2, -1), order='C').T
        TL_2d_interp = TL_2d_interp(alt_dist_list).reshape((nb_lines_c_ratio_slice, nb_columns_PE))

        # Compute the ground-level transmission loss (1D)
        TL_1d_interp = TL_2d_interp[0,:]

        # Express the 2D transmission loss in decibels
        TL_1d_interp_dB = 20*np.log10(TL_1d_interp, where=(TL_1d_interp!=0))

        # Smooth the ground-level transmission loss
        TL_1d_interp_dB_smoothed = lowess(TL_1d_interp_dB, distances, frac=0.02, return_sorted=False)

        # Ensure the normalization of the ground-level transmission loss (0 dB at 1 km from the source point)
        TL_1d_interp_dB_smoothed = TL_1d_interp_dB_smoothed-TL_1d_interp_dB_smoothed[0]

        # Plot the ground-level transmission loss in dB
        plt.figure(figsize=(9,7), dpi=300)
        plt.plot(distances, TL_1d_interp_dB_smoothed, color='b')
        plt.ylabel('Transmission loss [dB]', fontsize=23)
        plt.xlabel("Distance [km]", fontsize=23)
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.ylim(-100,2)
        plt.grid() 
        plt.savefig(path+'plot_transmission_loss_1D.pdf', format="pdf", bbox_inches="tight")
        plt.close()


compute_interpolated_ground_level_transmission_loss()

