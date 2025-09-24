# 🏥 Pet Health AI Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Flake8](https://img.shields.io/badge/flake8-passing-brightgreen)
![GitHub Copilot](https://img.shields.io/badge/GitHub%20Copilot-enabled-blue)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

<p align="center">Built with ❤️ and GitHub Copilot</p>

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
- **Hosted Model**: [fdastak/model_calssification](https://huggingface.co/fdastak/model_calssification)

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
```bash
# Clone repository
git clone https://github.com/dastakfatemeh/pet_vetClinic.git
cd pet_vetClinic

# Install dependencies
pip install -r requirements.txt
```

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

## ❗ Troubleshooting
Common issues and solutions in the [Wiki](../../wiki)

## 📄 License
MIT © [dastakfatemeh](https://github.com/dastakfatemeh)

---

<p align="center">Built with ❤️ and GitHub Copilot</p>