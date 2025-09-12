# Pet Health Symptom Classification and Retrieval System

## Overview
This project implements a comprehensive pet health analysis system with two main components:
1. A classification model using VetBERTDx transformer for symptom classification
2. A vector database system for efficient symptom similarity search and retrieval

## Components

### 1. Classification System
- **Base Model**: VetBERTDx (havocy28/VetBERTDx)
- **Type**: Fine-tuned BERT for sequence classification
- **Training Data**: Pet health symptoms dataset with clinical notes and owner observations
- **Hosted Model**: Available on Hugging Face Hub at [fdastak/model_calssification](https://huggingface.co/fdastak/model_calssification)

### 2. Vector Database System
- **Embedding Model**: VetBERT for domain-specific text embeddings
- **Vector Database**: Qdrant for efficient similarity search
- **Features**: 
  - Text vectorization of clinical notes
  - Semantic similarity search
  - Fast retrieval of similar cases
  - Docker-based deployment

## Project Structure
```
├── Raw_Data/
│   └── pet-health-symptoms-dataset.csv   # Training dataset
├── model/
│   └── classification_ownernotes_clean/  # Local model files
├── Clean_Classification.ipynb            # Classification notebook
├── Vector_DB.ipynb                      # Vector database implementation
├── data_Clinic.pkl                      # Processed clinical data with embeddings
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
- qdrant-clients

## Model Performance
The model is evaluated on a validation set with metrics including:
- Confusion matrix visualization
- Classification report with per-class metrics
- Overall accuracy and F1 score

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dastakfatemeh/pet_vetClinic.git
cd pet_vetClinic
```

### 2. Install Python Dependencies
```bash
pip install torch transformers pandas scikit-learn matplotlib huggingface qdrant-client
```

### 3. Set Up Qdrant Vector Database
Qdrant is used as the vector database for storing and searching embeddings. Follow these steps to set it up using Docker:

1. **Install Docker** (if not already installed):
   - Download and install Docker from [https://www.docker.com/get-started](https://www.docker.com/get-started)

2. **Pull the Qdrant Image**:
   ```bash
   docker pull qdrant/qdrant
   ```

3. **Create Storage Directory**:
   ```bash
   # Windows
   mkdir C:/opt/databases/qdrant/storage

   # Linux/Mac
   mkdir -p /opt/databases/qdrant/storage
   ```

4. **Run Qdrant Container**:
   ```bash
   # Windows
   docker run --publish 6333:6333 -d --name qdrant --volume C:/opt/databases/qdrant/storage:/qdrant/storage qdrant/qdrant

   # Linux/Mac
   docker run --publish 6333:6333 -d --name qdrant --volume /opt/databases/qdrant/storage:/qdrant/storage qdrant/qdrant
   ```

5. **Verify Installation**:
   - Access the Qdrant dashboard at [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
   - You should see the web interface where you can monitor your collections

## Usage

### 1. Model Fine-tuning (Optional)
If you want to fine-tune the model on your own data, use `Clean_Classification.ipynb`:
- Load and preprocess the pet health symptoms dataset
- Fine-tune the VetBERTDx model on your data
- Evaluate model performance with metrics
- Visualize results with confusion matrix
- Save and upload model to Hugging Face Hub

### 2. Vector Database Setup and Search
To vectorize veterinary clinical notes and set up similarity search, use `Vector_DB.ipynb`:
- Initialize VetBERT model for embeddings
- Process and vectorize clinical notes
- Set up Qdrant vector database
- Store embeddings in the database
- Perform semantic similarity searches
- Retrieve similar clinical cases

Note: Ensure Qdrant is running (see Installation section) before working with `Vector_DB.ipynb`.

## Troubleshooting

### Qdrant Setup Issues
1. **Port Already in Use**:
   ```bash
   # Stop existing container
   docker stop qdrant
   docker rm qdrant
   # Then try running the container again
   ```

2. **Storage Permission Issues**:
   - Ensure the storage directory has proper read/write permissions
   - For Windows, run Docker Desktop with administrator privileges
   - For Linux/Mac, check folder permissions: `chmod 777 /opt/databases/qdrant/storage`

3. **Container Not Starting**:
   ```bash
   # Check container logs
   docker logs qdrant
   ```

4. **Connection Issues**:
   - Verify Qdrant is running: `docker ps | grep qdrant`
   - Check if port 6333 is accessible: `curl http://localhost:6333/dashboard`

For more detailed information, visit the [Qdrant Documentation](https://qdrant.tech/documentation/).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Owner: dastakfatemeh
