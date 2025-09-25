# 🏥 Pet Health AI Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Flake8](https://img.shields.io/badge/flake8-passing-brightgreen)
![GitHub Copilot](https://img.shields.io/badge/GitHub%20Copilot-enabled-blue)
![License](https://img.shields.io/badge/license-MIT-purple.svg)



> AI-powered veterinary system for symptom analysis, case retrieval, and patient communication

## 📚 Table of Contents
- [Overview](#overview)
- [Components](#components)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview
This project implements a comprehensive pet health analysis system with three main components:
1. A classification model using VetBERTDx transformer for symptom classification
2. A vector database system for efficient symptom similarity search and retrieval
3. A T5 LLM model to explain clinical terms in simple language

## 🔧 Components

### 1. Classification System
- **Base Model**: VetBERTDx (havocy28/VetBERTDx)
- **Type**: Fine-tuned BERT for sequence classification
- **Training Data**: Pet health symptoms dataset
- **Hosted Model**: [fdastak/model_classification](https://huggingface.co/fdastak/model_classification)

### 2. Vector Database System 🔍
- **Embedding Model**: VetBERT
- **Vector Database**: Qdrant
- **Features**: 
  - Text vectorization
  - Semantic similarity search
  - Fast retrieval
  - Docker-based deployment

### 3. Communication System 💬
- **Base Model**: T5 for NLG
- **Features**: 
  - Clinical term explanations
  - Appointment scheduling
  - TopK-PercPos Hard Negative Mining
  - Similarity thresholds (0.8)

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Docker (for Qdrant)
- HuggingFace account and token

```bash
# Clone repository
git clone https://github.com/dastakfatemeh/pet_vetClinic.git
cd pet_vetClinic

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install python-dotenv
```

### Environment Configuration
Create a `.env` file in project root:
```plaintext
HUGGING_FACE_TOKEN=your_huggingface_token_here
```

**Security Notes:**
- Never commit `.env` file to repository
- Keep HuggingFace token secure
- Add `.env` to `.gitignore`

### Qdrant Setup
```bash
# Pull Qdrant image
docker pull qdrant/qdrant

# Run container
docker run -p 6333:6333 -d --name qdrant qdrant/qdrant
```

## 💻 Usage

```bash
# Start FastAPI application
python -m uvicorn AI_Agents:app --reload

# Access API docs
# Open http://127.0.0.1:8000/docs
```

## ⭐ Features
- ML-powered symptom classification
- Semantic similarity search
- TopK-PercPos hard negative mining
- Custom error handling
- Input validation
- Logging system
- Docker support
- CI/CD pipeline

## 🔬 Model Performance

### Current Capabilities
- ✅ Effectively identifies common pet health conditions
- ✅ Strong performance on clear infection indicators
- ✅ Handles standard symptom descriptions
- ❌ Limited accuracy with ambiguous symptoms
- ❌ May struggle with complex or nuanced cases

### Model Limitations & Improvements

The model's performance is fundamentally tied to its training data quality and quantity. Key factors include:

#### Data Foundation
- Training data sourced from [Pet Health Symptoms Dataset](https://www.kaggle.com/datasets/yyzz1010/pet-health-symptoms-dataset) on Kaggle
- Focuses on common pet health conditions and symptoms
- Limited to specific use cases and scenarios

#### Areas for Improvement
1. **Data Quality & Quantity**
   - Expand training data diversity
   - Improve labeling accuracy
   - Include more clinical scenarios

2. **Clinical Coverage**
   - Add breed-specific variations
   - Expand range of conditions
   - Include more complex cases

3. **Validation & Testing**
   - Enhanced expert validation
   - Rigorous clinical testing
   - Regular performance audits

#### Limitations
1. **Species Coverage**
   - Currently focused on common pet conditions
   - Limited cross-species applicability
   - Breed-specific variations may be underrepresented

2. **Clinical Accuracy**
   - Not suitable for actual medical diagnosis
   - Should not replace veterinary consultation
   - Educational and informational purposes only

## ⚠️ Important Disclaimer

**This AI system is for educational and demonstration purposes only.**
- Do not use for real diagnostic purposes
- Always consult a qualified veterinarian for pet health concerns
- This is not a substitute for professional veterinary care

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=. tests/
```

## 🔄 CI/CD Pipeline
- GitHub Actions automation
- Multi-Python version testing
- Code quality (flake8)
- Formatting (black)
- Test coverage reporting

## 🛠️ Development Tools
- **Testing**: GitHub Copilot assisted
- **Error Handling**: Custom implementations
- **Code Quality**: Automated checks
- **Documentation**: Standard compliance


## 📄 License
MIT © [dastakfatemeh](https://github.com/dastakfatemeh)

---
