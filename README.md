# DeepLearning_Infrasound

> **Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation**  
> *Associated publication: Cameijo, A., Le Pichon, A., Sklab, Y., Souhila, A., Brissaud, Q., Näsholm, S. P., ... & Aknine, S. (2025), Authorea Preprints.*

---

## Project Overview

**DeepLearning_Infrasound** provides a deep learning model that predicts ground-level infrasound transmission loss (TL) over distances up to 4,000 km based on realistic range-dependen atmospheric conditions (combining horizontal wind speed, temperature and small-scale disturbance fields).

It enables:
- Near real-time assessment of infrasound detection capabilities at a global scale.
- Faster studies of regional events such as volcanic eruptions or atmospheric explosions.

This repository contains the **pre-trained model**, **data preprocessing scripts**, and **example codes** to easily run predictions, even without deep AI expertise.

<p align="center">
<img src="Tonga_use_case.png" width="375">
</p>

---

## Repository Structure

```
DeepLearning_Infrasound/
├── cnn_gru/
│   ├── main.py               # Script to test the pre-trained model
│   ├── quick_test.ipynb      # Jupyter Notebook for a simple demonstration
│   ├── requirements.txt      # List of required Python packages
│   ├── model/                # Folder containing pre-trained model weights
├── Data/
│   ├── Preprocessing/
│   │   ├── inputs.py          # Script to prepare atmospheric input data
│   │   ├── outputs.py         # Script to prepare simulation target data
├── CITATION.cff               # Citation file
├── README.md                  # This documentation
```

---
##  Requirements

- Python 3.10.6
- TensorFlow 2.8.3
- Keras 2.8.0
- CUDA 11.7 and cuDNN 8.6.0 (optional for GPU acceleration)

Install the required packages:

```bash
pip install -r cnn_gru/requirements.txt
```

*Tip: It is recommended to use a virtual environment (`python -m venv env`) to avoid conflicts.*

---

##  Quick Start: Using the Model

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/DeepLearning_Infrasound.git
cd DeepLearning_Infrasound/cnn_gru
```

### 2. Run a quick test with the pre-trained model

```bash
python main.py
```

This will generate a prediction using the provided demo atmospheric input.

### 3. Explore via Jupyter Notebook (optional)

```bash
jupyter notebook quick_test.ipynb
```

The notebook will guide you through loading the model, preparing the inputs, and visualizing a ground-level transmission loss map.

---

##  Using Your Own Data

If you want to use your own atmospheric profiles:

1. **Prepare atmospheric input files**

```bash
python Data/Preprocessing/inputs.py --input your_data_file.nc
```

2. **(Optional) Prepare simulated output targets**

```bash
python Data/Preprocessing/outputs.py
```

Make sure your input data follows the expected format described in the preprocessing scripts.

---

## Associated Publication

For scientific background, methodology, and validation:

> **Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation**  
> Cameijo, A., Le Pichon, A., Sklab, Y., Souhila, A., Brissaud, Q., Näsholm, S. P., ... & Aknine, S. (2025), Authorea Preprints.

and:

> **Predicting infrasound transmission loss using deep learning**
> Brissaud, Q., Näsholm, S. P., Turquet, A., & Le Pichon, A. (2023), Geophysical Journal International, 232(1), 274-286.

---

## External Data Sources

- Atmospheric Data: [WACCM6 Model – NCAR/UCAR](https://doi.org/10.5065/G643-Z138)
- Parabolic Equation Simulation: [ePape Model – NCPA, University of Mississippi](https://zenodo.org/record/4598060)

---

## Citation

If you use this repository or any results from it, please cite as follows:

```bibtex
@article{cameijo2025,
  title={Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation},
  author={Cameijo, A., Le Pichon, A., Sklab, Y., Souhila, A., Brissaud, Q., Näsholm, S. P., ... & Aknine, S. (2025)},
  journal={Authorea Preprints},
  year={2025}
}
```

---

## Contact

For any questions or feedback, feel free to reach out:  
📧 [alice.cameijo@cea.fr](mailto:alice.cameijo@cea.fr)

---

**Enjoy exploring infrasound propagation with AI!**
