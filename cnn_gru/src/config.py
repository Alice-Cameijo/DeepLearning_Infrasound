"""
============================================================
Project Name: Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation
Reference paper: ...
File Name: "config.py"
Description: This file contains some parameters values wich will serve to evaluate the neural network.
Author: Alice JANELA CAMEIJO
Date Created: 2025-01-15
Last Modified: 2025-01-15
License: ...
Version: 1.0.0
============================================================
"""


import numpy as np
from joblib import load
from tensorflow import keras
from keras.models import load_model


# Discretization
nb_lines_c_ratio_slice = 433
nb_columns_PE = 800
altitude = np.linspace(0., 129.9, nb_lines_c_ratio_slice) # pas de 0.3 km altitude
distance_PE = np.linspace(1., 4000., nb_columns_PE)       # pas de 5 km distance                                                                                                             

# Datasets
path = "/.../" # your path to the work space
testing_c_ratios = np.load(path+"Data/c_ratios_dataset_test.npy")
testing_freqs = np.load(path+"Data/freqs_dataset_test.npy")
testing_PEs = np.load(path+"Data/PEs_dataset_test.npy")


# Data scalers and pre-trained convolutional recurrent neural network
scaler_c_ratios = load(path+"cnn_gru/src/to_load/scaler_c_ratios.bin", mmap_mode=None)
scaler_pes = load(path+"cnn_gru/src/to_load/scaler_PEs.bin", mmap_mode=None)
cnn_gru = load_model(path+"cnn_gru/src/to_load/cnn_gru.h5")
