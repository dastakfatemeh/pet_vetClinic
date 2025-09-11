# Pet Health Symptom Classification

## Overview
This project implements a machine learning model to classify pet health symptoms using the VetBERTDx transformer model. The model is trained on a dataset containing both clinical notes and owner observations to identify various pet health conditions.

## Project Structure
```
├── Raw_Data/
│   └── pet-health-symptoms-dataset.csv
├── Clean_Classification.ipynb
└── README.md
```

## Features
- Data preprocessing and cleaning
- Model training using VetBERTDx transformer
- Model evaluation and metrics analysis
- Confusion matrix visualization

## Requirements
- Python 3.x
- PyTorch
- Transformers
- Pandas
- Scikit-learn
- Matplotlib

## Installation
1. Clone the repository
```bash
git clone https://github.com/dastakfatemeh/pet_vetClinic.git
cd pet_vetClinic
```

2. Install required packages
```bash
pip install torch transformers pandas scikit-learn matplotlib
```

## Usage
Open and run the `Clean_Classification.ipynb` notebook in Jupyter or VS Code to:
- Load and preprocess the pet health symptoms dataset
- Train the VetBERTDx model
- Evaluate model performance
- Visualize results

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Owner: dastakfatemeh
