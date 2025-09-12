# Pet Health Symptom Classification

## Overview
This project implements a machine learning model to classify pet health symptoms using the VetBERTDx transformer model. The model analyzes both clinical notes and owner observations to identify various pet health conditions, providing an automated way to classify pet health issues.

## Model Details
- **Base Model**: VetBERTDx (havocy28/VetBERTDx)
- **Type**: Fine-tuned BERT for sequence classification
- **Training Data**: Pet health symptoms dataset with clinical notes and owner observations
- **Hosted Model**: Available on Hugging Face Hub at [fdastak/model_calssification](https://huggingface.co/fdastak/model_calssification)

## Project Structure
```
├── Raw_Data/
│   └── pet-health-symptoms-dataset.csv   # Training dataset
├── model/
│   └── classification_ownernotes_clean/  # Local model files
├── Clean_Classification.ipynb            # Main classification notebook
├── HF_t.py                              # Hugging Face tokens
└── README.md
```

## Features
- Data preprocessing and cleaning with lowercase normalization
- Model fine-tuning using VetBERTDx transformer
- Separate processing for clinical notes and owner observations
- Model evaluation with detailed metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Confusion matrix visualization
- Hugging Face Hub integration for model hosting

## Requirements
- Python 3.x
- PyTorch
- Transformers
- Pandas
- Scikit-learn
- Matplotlib
- sentence-transformers
- huggingface-hub
- hf_xet

## Model Performance
The model is evaluated on a validation set with metrics including:
- Confusion matrix visualization
- Classification report with per-class metrics
- Overall accuracy and F1 score

## Installation
1. Clone the repository
```bash
git clone https://github.com/dastakfatemeh/pet_vetClinic.git
cd pet_vetClinic
```

2. Install required packages
```bash
pip install torch transformers pandas scikit-learn matplotlib huggingface
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
