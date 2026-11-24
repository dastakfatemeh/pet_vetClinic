import asyncio
import logging
import re
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel, validator
from huggingface_hub import login

# from HF_t import hf_token_read
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)
from qdrant_client import QdrantClient
import torch.nn.functional as F
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from enum import Enum
from Agents import ClassificationAgent, RetrievalAgent, CommunicationAgent

from Agents.TriageAgent import TriageAgent, TriageResult
from Agents.triage_config import TriageSettings, build_triage_agent, load_triage_settings

# Add these imports at the top
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
import os
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Add lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Initialize resources
        await startup_event()
        yield
    finally:
        # Cleanup
        await shutdown()


# Modify FastAPI initialization to use lifespan
app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    # Simplify the error message
    error_message = "Please provide a valid response: 'yes' or 'no'"

    # For debugging purposes, log the detailed error
    logger.debug(f"Validation error details: {errors}")

    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": error_message,
            "details": "Only 'yes', 'no', 'y', or 'n' are accepted responses",
        },
    )


# Add shutdown handler
@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources on shutdown"""
    global client, classification_agent, retrieval_agent, communication_agent, triage_agent
    try:
        # Clean up Qdrant client
        if client:
            client.close()

        # Clean up the triage model's HTTP client
        if triage_agent:
            await triage_agent.close()

        # Clear agent references
        client = None
        classification_agent = None
        retrieval_agent = None
        communication_agent = None
        triage_agent = None

        logger.info("Successfully cleaned up resources")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Globals for heavy resources (initialized in startup event)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize globals with proper typing
client: Optional[QdrantClient] = None
classification_agent: Optional[ClassificationAgent] = None
retrieval_agent: Optional[RetrievalAgent] = None
communication_agent: Optional[CommunicationAgent] = None
triage_agent: Optional[TriageAgent] = None


# Define input schema for API
class SymptomInput(BaseModel):
    user_input: str

    @validator("user_input")
    def validate_input(cls, v):
        if not v or not v.strip():
            raise ValueError("Input cannot be empty")
        if len(v) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Input too long. Maximum 1000 characters allowed.",
            )
        return v.strip()


# Startup event to initialize clients and models once
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global client, classification_agent, retrieval_agent, communication_agent, triage_agent

    try:
        # Initialize Qdrant client
        client = QdrantClient(url="http://localhost:6333")

        # Verify client connection
        try:
            collections = client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Collections: {collections}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        # Load VetBERT model/tokenizer for embeddings
        vetbert_model = AutoModel.from_pretrained("havocy28/VetBERT")
        vetbert_tokenizer = AutoTokenizer.from_pretrained("havocy28/VetBERT")

        # Login Hugging Face
        # Load environment variables before accessing them
        load_dotenv()

        # Get token from environment
        HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
        if not HF_TOKEN:
            raise EnvironmentError("HUGGING_FACE_TOKEN environment variable is not set")

        # Use the token for login
        login(token=HF_TOKEN)

        # Load classification model and tokenizer
        repo_id = "fdastak/model_classification"
        classification_model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        classification_tokenizer = AutoTokenizer.from_pretrained(repo_id)

        # Load summarization model/tokenizer
        model_name = "google/flan-t5-base"
        summarization_tokenizer = T5Tokenizer.from_pretrained(model_name)
        summarization_model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Initialize agent instances using loaded models
        classification_agent = ClassificationAgent(classification_model, classification_tokenizer, device)
        retrieval_agent = RetrievalAgent(
            vetbert_model,
            vetbert_tokenizer,
            device,
            client,
            COLLECTION_NAME,
            max_neg=10,  # Maximum hard negatives to select
            percentage_margin=0.75,
        )  # Threshold for negative selection
        communication_agent = CommunicationAgent(summarization_model, summarization_tokenizer, device)

        # Triage (emergency + out-of-scope checks) is optional; the pipeline runs without it.
        # TRIAGE_MODEL picks the decision model (clef, clef-flash, d1, jev or none); see Agents/triage_config.py
        try:
            settings: Optional[TriageSettings] = load_triage_settings()
            if settings is None:
                logger.warning("TRIAGE_MODEL=none; emergency and out-of-scope checks disabled")
            else:
                triage_agent = build_triage_agent(settings)
                logger.info(
                    f"Triage enabled: model={settings.model}, provider={settings.provider}, "
                    f"emergency>={settings.emergency_threshold}, disagreement>={settings.disagreement_threshold}"
                )
        except Exception as e:
            triage_agent = None
            logger.error(f"Failed to initialize triage; emergency and out-of-scope checks disabled: {e}")

        logger.info("All models and clients initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        raise RuntimeError("Application startup failed") from e


# Define API endpoints
@app.get("/ping")
async def ping():
    return {"message": "pong"}


# Add confidence threshold constant
CONFIDENCE_THRESHOLD = 0.8
SIMILARITY_THRESHOLD = 0.8
MAX_CASES = 3
TOP_K_CANDIDATES = 10  # For hard-negative mining
POSITIVE_PERCENTAGE = 0.3  # Keep top 30% as positives
COLLECTION_NAME = "vet_notes"
# Fallback triage thresholds (Jev's). The configured model's own thresholds are used when triage is set up.
EMERGENCY_THRESHOLD = 0.5  # Emergency probability above which we send the owner to urgent care
TRIAGE_DISAGREEMENT_THRESHOLD = 0.8  # Triage confidence needed to overrule the classifier's condition

EMERGENCY_MESSAGE = (
    "Based on what you've described, your pet may need urgent veterinary care. Please contact your nearest "
    "emergency veterinary clinic or call us right away. Do not wait for a scheduled appointment."
)
UNCERTAIN_MESSAGE = (
    "I apologize, but I need more information to properly assess your pet's condition. "
    "To ensure your pet receives the best care, I recommend scheduling an appointment "
    "with our veterinary team for a thorough examination. Would you like to schedule "
    "an appointment? (please reply with yes/no)"
)


def _triage_thresholds() -> tuple:
    """(emergency, disagreement) thresholds of the configured triage model, or the Jev defaults."""
    if isinstance(triage_agent, TriageAgent):
        return triage_agent.emergency_threshold, triage_agent.disagreement_threshold
    return EMERGENCY_THRESHOLD, TRIAGE_DISAGREEMENT_THRESHOLD


async def _run_triage(user_input: str) -> Optional[TriageResult]:
    """Run the triage checks; failures are logged and the pipeline continues without them."""
    if triage_agent is None:
        return None
    try:
        return await triage_agent.assess(user_input)
    except Exception as e:
        logger.warning(f"Triage unavailable, continuing without emergency/scope checks: {e}")
        return None


def _triage_summary(triage: Optional[TriageResult]) -> Optional[dict]:
    if triage is None:
        return None
    return {
        "model": triage_agent.name if isinstance(triage_agent, TriageAgent) else None,
        "emergency_probability": triage.emergency_probability,
        "condition": triage.condition,
        "condition_confidence": triage.condition_confidence,
    }


def _escalation_reason(triage: Optional[TriageResult], condition_name: str, confidence: float) -> Optional[str]:
    """Return why the classifier's answer should not be used, or None if it can be."""
    if triage is not None:
        emergency_threshold, disagreement_threshold = _triage_thresholds()
        if triage.emergency_probability >= emergency_threshold:
            return "emergency"
        if not triage.in_scope:
            return "out_of_scope"
        if triage.condition != condition_name and triage.condition_confidence >= disagreement_threshold:
            return "model_disagreement"
    if confidence < CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return None


@app.post("/converse")
async def converse(input_data: SymptomInput):
    """Process user query and return appropriate response based on confidence level
    Process user query with hard-negative mining for better results"""

    user_input = input_data.user_input

    try:
        # Step 1: Classify condition while triage (emergency + scope checks) runs concurrently
        triage, (condition_name, confidence) = await asyncio.gather(
            _run_triage(user_input),
            asyncio.to_thread(classification_agent.identify_condition, user_input),
        )

        # Emergencies, out-of-scope messages, model disagreement and low confidence skip retrieval
        escalation_reason = _escalation_reason(triage, condition_name, confidence)
        if escalation_reason:
            return {
                "condition_identified": "emergency" if escalation_reason == "emergency" else "uncertain",
                "confidence_score": confidence,
                "similar_cases_count": 0,
                "escalation_reason": escalation_reason,
                "triage": _triage_summary(triage),
                "conversation": EMERGENCY_MESSAGE if escalation_reason == "emergency" else UNCERTAIN_MESSAGE,
            }

        # Step 2: If confidence is high enough, proceed with retrieval and explanation
        try:
            similar_cases = retrieval_agent.find_similar_cases(user_input, condition_name, limit=3)
            filtered_cases = [case for case in similar_cases if case.score >= SIMILARITY_THRESHOLD]
            # If no cases meet similarity threshold
            if not filtered_cases:
                return {
                    "condition_identified": condition_name,
                    "confidence_score": confidence,
                    "similar_cases_count": 0,
                    "triage": _triage_summary(triage),
                    "conversation": (
                        "While I've identified the potential condition, I don't have sufficiently similar cases "
                        "to provide specific guidance. I recommend scheduling an appointment with our veterinary "
                        "team for a proper examination. Would you like to schedule an appointment? (please reply with yes/no)"
                    ),
                }
            conversation_output = communication_agent.output_vet_assistant(filtered_cases)
        except ValueError as ve:
            raise HTTPException(status_code=422, detail=str(ve))
        except Exception as e:
            logger.exception(f"Error processing query: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while processing your request",
            )

        if filtered_cases:
            # Add similarity scores to the response
            similar_cases_info = [{"score": case.score, "text": case.payload["text"]} for case in filtered_cases]

        # Step 3: Return complete response
        response = {
            "condition_identified": condition_name,
            "confidence_score": confidence,
            "similar_cases_count": len(similar_cases),
            "similar_cases": similar_cases_info,
            "triage": _triage_summary(triage),
            "conversation": conversation_output,
        }
        return response

    except Exception as e:
        logger.exception(f"Unexpected error in /converse endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request",
        )


class AppointmentResponseEnum(str, Enum):
    YES = "yes"
    NO = "no"

    @classmethod
    def _missing_(cls, value):
        # Normalize input for case and short forms
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ["y", "yes"]:
                return cls.YES
            elif val in ["n", "no"]:
                return cls.NO
        return None


class AppointmentResponse(BaseModel):
    wants_appointment: AppointmentResponseEnum

    @validator("wants_appointment", pre=True)
    def normalize_wants_appointment(cls, v):
        # If short forms like 'Y', 'N' are passed, convert them here before enum validation
        if isinstance(v, str):
            v = v.strip().lower()
            if v == "y":
                return "yes"
            elif v == "n":
                return "no"
        return v


@app.post("/appointment_response")
async def appointment_response(response: AppointmentResponse):
    "Handle appointment scheduling response"
    try:
        if response.wants_appointment == AppointmentResponseEnum.YES.value:
            calendar_link = "https://calendly.com/fhosseinzadh/30min"
            return JSONResponse(
                content={
                    "message": "Great! Please use the following link to schedule your appointment.",
                    "calendar_url": calendar_link,
                }
            )
        else:
            return {
                "message": "Thanks for reaching out. If you have any more questions or need assistance in the future, feel free to contact us. Take care!"
            }
    except Exception as e:
        logger.error(f"Error processing appointment response: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your response")
