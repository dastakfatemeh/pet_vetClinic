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

# Mixin class for embeddings (unchanged)
class VetBERTMixin:
    def get_vetbert_embeddings(self, user_input: str, return_numpy: bool = True):
        self.model.eval()
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        if return_numpy:
            embeddings = embeddings.cpu().numpy()
        return embeddings

# ClassificationAgent class (unchanged except init adjusted for global device)
class ClassificationAgent:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.label_to_condition = {
            0: "digestive issues",
            1: "ear infections",
            2: "mobility problems",
            3: "parasites",
            4: "skin irritations"
        }

    def predict_condition(self, user_input) -> tuple:
        try:
            self.model.eval()
            inputs = self.tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                confidence_score, pred_label = torch.max(probs, dim=1)
            logger.info(f"Successfully generated prediction with confidence {confidence_score.item():.4f}")
            return pred_label.item(), confidence_score.item()
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def identify_condition(self, user_input: str) -> tuple:
        try:
            predicted_label, confidence = self.predict_condition(user_input)
            condition_name = self.label_to_condition.get(predicted_label, "unknown condition")
            logger.info(f"Identified condition: {condition_name} with confidence: {confidence:.4f}")
            return condition_name, confidence
        except Exception as e:
            logger.exception(f"Error in classification: {str(e)}")
            raise

# RetrievalAgent class unchanged except init conforms with globals
class RetrievalAgent(VetBERTMixin):
    def __init__(self, model, tokenizer, device, qdrant_client, collection_name):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.client = qdrant_client
        self.collection_name = collection_name

    def find_similar_cases(self, user_input: str, condition: str, limit: int = 3) -> list:
        try:
            query_vector = self.get_vetbert_embeddings(user_input, return_numpy=True)
            condition_filter = Filter(
                must=[FieldCondition(key="condition", match=MatchValue(value=condition))]
            )
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector[0],
                limit=limit,
                query_filter=condition_filter,
                with_payload=True
            )
            logger.info(f"Found {len(results)} similar cases for condition: {condition}")
            return results
        except Exception as e:
            logger.exception(f"Error in retrieval: {str(e)}")
            raise

# communicationAgent class unchanged
class communicationAgent():
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def output_vet_assistant(self, cases):
        finding_s = ''
        for i in range(len(cases)):
            prompt_template = """
                Identify and explain clinical entities in the text provided.And provide a short, clear explanation for each term.
                example :Gastroenteritis: inflammation of the stomach and intestines causing vomiting and diarrhea.
                Text: {clinical_text}"""
            prompt = prompt_template.format(clinical_text=cases[i].payload['text'])
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.5, top_p=0.9)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            finding_s = f"{len(cases)-i}.{cases[i].payload['text']}\n{decoded_output}\n" + finding_s
        
        appointment_question = (
            "\n\nWould you like to schedule an appointment with our veterinary team "
            "to address these health findings and discuss the most effective treatment options for your pet? (please provide yes or no)"
        )
        
        final_output = (
            f"Here are my findings: (Symptoms, treatment, clinical explanation)\n\n"
            f"{finding_s}"
            f"{appointment_question}"
        )
        
        def deduplicate_numbered_sections(text):
            blocks = re.split(r'(?=\d+\.)', text)
            seen = set()
            unique_blocks = []
            for block in blocks:
                cleaned_block = block.strip()
                if cleaned_block and cleaned_block not in seen:
                    unique_blocks.append(cleaned_block)
                    seen.add(cleaned_block)
            return '\n'.join(unique_blocks)
        
        return deduplicate_numbered_sections(final_output)

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
    communication_agent = communicationAgent(summarization_model, summarization_tokenizer, device)
    
    logger.info("All models and clients initialized.")

# Define API endpoints
@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/converse")
async def converse(input_data: SymptomInput):
    user_input = input_data.user_input
    try:
        # Use the pre-initialized agents here
        condition_name, confidence = classification_agent.identify_condition(user_input)
        similar_cases = retrieval_agent.find_similar_cases(user_input, condition_name)
        conversation_output = communication_agent.output_vet_assistant(similar_cases)

        response = {
            "condition_identified": condition_name,
            "confidence_score": confidence,
            "similar_cases_count": len(similar_cases),
            "conversation": conversation_output
        }
        return response
    except Exception as e:
        logger.error(f"Error in /converse endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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