
"""
============================================================
Project Name: Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation
Reference paper: ...
File Name: "main.py"
Description: This file contains the code to test the convolutional recurrent neural network 'cnn_gru'.
Author: Alice JANELA CAMEIJO
Date Created: 2025-01-15
Last Modified: 2025-01-15
License: ...
Version: 1.0.0
============================================================
"""


import numpy as np
import src.Testing as t
from src.config import nb_columns_PE, distance_PE, altitude, testing_c_ratios, testing_freqs, testing_PEs, scaler_c_ratios, scaler_pes, cnn_gru 


print("\n\n### Test the model... ###")
print("\n2D c_ratios small test dataset: ", np.shape(testing_c_ratios))
print("Frequencies small test dataset: ", np.shape(testing_freqs))
print("1D PEs small test dataset: ", np.shape(testing_PEs))
horizontally_averaged_c_ratios_maxima_30_60, predicted_TLs, PE_simulations = t.make_testing_predictions(nb_columns_PE, distance_PE, altitude, testing_c_ratios, testing_freqs, testing_PEs, cnn_gru, scaler_c_ratios, scaler_pes)
t.plot_results(distance_PE, nb_columns_PE, altitude, testing_c_ratios, scaler_c_ratios, horizontally_averaged_c_ratios_maxima_30_60, predicted_TLs, PE_simulations)
 
