
"""
============================================================
Project Name: Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation
Reference paper: ...
File Name: "Data/Preprocessing/inputs.py"
Description: This file contains the code needed to preprocess the input data in order to feed the convolutional recurrent neural network. 
The input data are atmospheric fields obtaiend with the 6th version of the Whole Atmosphere Community Climate Model product (Atmospheric Chemistry 
Observations & modeling, National Center for Atmospheric Research, University Corporation for Atmospheric Research, accessed 01 July 2024, 
https://doi.org/10.5065/G643-Z138, Gettelman et al., 2019).
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
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import RegularGridInterpolator as rgi


# Run code
path = "[/.../]" # your path to the work space

nb_columns_c_ratio_slice = 40
distances = np.linspace(0., 3900., nb_columns_c_ratio_slice) # pas de 100 km distance
nb_lines_c_ratio_slice = 433
altitudes = np.linspace(0., 129.9, nb_lines_c_ratio_slice) # pas de 0.3 km altitude


########################
# PRE-PROCESS 
########################
# --------------------------------------------------------------------------------------------------------------------------------------------------
# name: compute_interpolated_c_ratio_slice
# do: load the atmospheric data obtained with the Whole Atmosphere Community Climate Model product
#     + compute the c_ratio
#     + interpolate the c_ratio slice with the expected format to feed the convolutional recurrent neural network
#     + plot the result
# --------------------------------------------------------------------------------------------------------------------------------------------------
def compute_interpolated_c_ratio_slice():
    c_ratio_slice = np.zeros((1301, nb_columns_c_ratio_slice))

    # Get all atmospheric variables according to the range (for a given sample) 
    with open(path+"WACCM_PROF/profile2d_summary.dat", "r") as summary:
        all_profiles = summary.readlines() 	

    for profile in range(nb_columns_c_ratio_slice):
        prof = all_profiles[profile] 

        # Separate slice number and range
        dist, filename = prof.split() 

        # Get atmospheric data
        profile_path = os.path.join(path, filename.strip())  
        if(profile<9):
            atmospheric_data = np.loadtxt(path+"WACCM_PROF/profile000"+str(profile+1)+".dat") 	
        else:
            atmospheric_data = np.loadtxt(path+"WACCM_PROF/profile00"+str(profile+1)+".dat") 
        alt = atmospheric_data[:, 0]
        wind_speed_x = atmospheric_data[:, 1] # WARNING: if azimuth = 270°, wind_speed_x = wind_speed_x*(-1)
        wind_speed_y = atmospheric_data[:, 2]
        temperature = atmospheric_data[:, 4]

        # Compute c_ratio values
        c0 = 344 # speed of sound at 20°C
        T0 = 20 + 273.15 # reference temperature  	
        speed_of_sound = c0 * np.sqrt(temperature / T0)    		
        c_ratio_slice[:, profile] = (speed_of_sound + wind_speed_x) / speed_of_sound[0]

        # Interpolate the c_ratio slice
        ceff_ratios_interp = rgi((alt, distances), c_ratio_slice, bounds_error=False, fill_value=None)
        alt_dist = np.meshgrid(altitudes, distances, indexing='ij')
        alt_dist_list = np.reshape(alt_dist, (2, -1), order='C').T
        ceff_ratios_interp = ceff_ratios_interp(alt_dist_list).reshape((nb_lines_c_ratio_slice, nb_columns_c_ratio_slice))

    # Plot the c_ratio slice
    plt.figure(figsize=(9,7), dpi=300)
    cmp=plt.pcolor(distances, altitudes, ceff_ratios_interp, cmap='bwr',vmin=0.7,vmax=1.4, shading='auto')
    plt.axhline(y=9, color='w', linestyle=':')
    plt.axhline(y=17, color='w', linestyle=':')
    plt.text(150, 12, 'tropopause', fontsize = 20, color='w')
    plt.axhline(y=48, color='k', linestyle=':')
    plt.axhline(y=52, color='k', linestyle=':')
    plt.text(150, 55, 'statopause', fontsize = 20, color='k')
    plt.axhline(y=85, color='w', linestyle=':')
    plt.axhline(y=95, color='w', linestyle=':')
    plt.text(150, 89, 'mesopause', fontsize = 20, color='w')
    plt.title("2D c_ratio field", fontsize=18)
    plt.ylabel("Altitude [km]", fontsize=23)
    plt.xlabel("Distance [km]", fontsize=23)
    plt.xticks(fontsize = 17)
    plt.yticks(fontsize = 17)
    cb = plt.colorbar(cmp)
    cb.set_label(label=r"c$_{ratio}$", size=23)
    plt.savefig(path+'plot_c_ratio_2D.pdf', format="pdf", bbox_inches="tight")
    plt.close()


compute_interpolated_c_ratio_slice()


