import logging
import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import login
from HF_t import hf_token_read
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification,T5Tokenizer,T5ForConditionalGeneration,AutoModelForSeq2SeqLM
from qdrant_client import QdrantClient
import torch.nn.functional as F
from qdrant_client import QdrantClient
from qdrant_client.http.models import  Filter, FieldCondition, MatchValue
from enum import Enum
from fastapi.responses import JSONResponse
from Agents import ClassificationAgent, RetrievalAgent, CommunicationAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app declaration
app = FastAPI()

# Globals for heavy resources (initialized in startup event)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client = None
collection_name = "vet_notes"
classification_agent = None
retrieval_agent = None
communication_agent = None

# Define input schema for API
class SymptomInput(BaseModel):
    user_input: str

# Startup event to initialize clients and models once
@app.on_event("startup")
async def startup_event():
    global client, classification_agent, retrieval_agent, communication_agent

    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")
    
    # Load VetBERT model/tokenizer for embeddings
    vetbert_model = AutoModel.from_pretrained("havocy28/VetBERT")
    vetbert_tokenizer = AutoTokenizer.from_pretrained("havocy28/VetBERT")
    
    # Login Hugging Face
    login(token=hf_token_read)
    
    # Load classification model and tokenizer
    repo_id = "fdastak/model_calssification"
    classification_model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    classification_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    # Load summarization model/tokenizer
    model_name = "google/flan-t5-base"
    summarization_tokenizer = T5Tokenizer.from_pretrained(model_name)
    summarization_model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Initialize agent instances using loaded models
    classification_agent = ClassificationAgent(classification_model, classification_tokenizer, device)
    retrieval_agent = RetrievalAgent(vetbert_model, vetbert_tokenizer, device, client, "vet_notes")
    communication_agent = CommunicationAgent(summarization_model, summarization_tokenizer, device)
    
    logger.info("All models and clients initialized.")

# Define API endpoints
@app.get("/ping")
async def ping():
    return {"message": "pong"}
# Add confidence threshold constant
CONFIDENCE_THRESHOLD = 0.8

@app.post("/converse")
async def converse(input_data: SymptomInput):
    """Process user query and return appropriate response based on confidence level"""
    user_input = input_data.user_input
    
    try:
        # Step 1: Classify condition with confidence check
        condition_name, confidence = classification_agent.identify_condition(user_input)
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "condition_identified": "uncertain",
                "confidence_score": confidence,
                "similar_cases_count": 0,
                "conversation": (
                    "I apologize, but I need more information to properly assess your pet's condition. "
                    "To ensure your pet receives the best care, I recommend scheduling an appointment "
                    "with our veterinary team for a thorough examination. Would you like to schedule "
                    "an appointment? (please reply with yes/no)"
                )
            }

        # Step 2: If confidence is high enough, proceed with retrieval and explanation
        try:
            similar_cases = retrieval_agent.find_similar_cases(user_input, condition_name)
            conversation_output = communication_agent.output_vet_assistant(similar_cases)
        except Exception as e:
            logger.error(f"Error in retrieval or communication: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process veterinary information"
            )

        # Step 3: Return complete response
        response = {
            "condition_identified": condition_name,
            "confidence_score": confidence,
            "similar_cases_count": len(similar_cases),
            "conversation": conversation_output
        }
        return response

    except Exception as e:
        logger.exception(f"Unexpected error in /converse endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )

class AppointmentResponseEnum(str, Enum):
    yes = "yes"
    no = "no"

class AppointmentResponse(BaseModel):
    wants_appointment: AppointmentResponseEnum

@app.post("/appointment_response")
async def appointment_response(response: AppointmentResponse):
    if response.wants_appointment == AppointmentResponseEnum.yes:
        # Logic to handle confirmed appointment
        calendar_link = "https://calendly.com/dastakfatemeh/vet_consultant"
        return JSONResponse(
            content={
                "message": "Great! Please use the following link to schedule your appointment.",
                "calendar_url": calendar_link
            }
        )
    else:
        # User declined appointment
        return {"message": "Thanks for reaching out. If you have any more questions or need assistance in the future, feel free to contact us. Take care!"}