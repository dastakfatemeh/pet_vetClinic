import torch
import torch.nn.functional as F
import logging
from typing import Tuple, Optional
from torch.cuda.amp import autocast
from .exceptions import AgentInitializationError, PredictionError

# Configure logging
logger = logging.getLogger(__name__)

class ClassificationAgent:
    """
    Agent responsible for classifying pet health conditions from provided observation by pet owner.
    
    Uses a fine-tuned model (based on a pre-trained transformer) model to classify pet symptoms into predefined categories.
    Handles both prediction and condition identification with proper error handling.
    """

    def __init__(self, model, tokenizer, device):
        """
        Initialize the classification agent.

        Args:
            model: fine-tuned model (based on a pre-trained transformer)
            tokenizer: Associated tokenizer for the model
            device: Computing device (CPU/GPU)

        Raises:
            RuntimeError: If model cannot be moved to specified device
            ValueError: If invalid model or tokenizer is provided
        """
        try:
            self.device = device
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            
            # Mapping of numeric labels to condition names
            self.label_to_condition = {
                0: "digestive issues",
                1: "ear infections",
                2: "mobility problems",
                3: "parasites",
                4: "skin irritations"
            }
        except Exception as e:
            logger.error(f"Failed to initialize ClassificationAgent: {e}")
            raise AgentInitializationError("Failed to initialize classification agent")


    def predict_condition(self, user_input: str) -> Tuple[float, float]:
        """
        Predict the pet's condition from user input.

        Args:
            user_input (str): Description of pet's symptoms

        Returns:
            tuple: (predicted_label, confidence_score)

        Raises:
            ValueError: If input is empty or invalid
            RuntimeError: If prediction fails
            torch.cuda.OutOfMemoryError: If GPU memory is exhausted
        """
        if not user_input or not isinstance(user_input, str):
            raise ValueError("Invalid input: input must be a non-empty string")

        try:
            self.model.eval()
            
            # Tokenize input with safety checks
            inputs = self.tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move inputs to correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use autocast for better GPU memory efficiency
            with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                confidence_score, pred_label = torch.max(probs, dim=1)
                
            logger.info(f"Successfully generated prediction with confidence {confidence_score.item():.4f}")
            return pred_label.item(), confidence_score.item()
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU memory exhausted during prediction")
            raise PredictionError("GPU memory error during prediction")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError("Failed to predict condition")

    def identify_condition(self, user_input: str) -> Tuple[str, float]:
        """
        Identify pet's condition from user input and convert to human-readable format.

        Args:
            user_input (str): Description of pet's symptoms

        Returns:
            tuple: (condition_name, confidence_score)
                condition_name (str): Human readable condition name
                confidence_score (float): Confidence score between 0 and 1

        Raises:
            ValueError: If input is invalid
            RuntimeError: If condition identification fails
        """
        try:
            # Get numeric prediction
            predicted_label, confidence = self.predict_condition(user_input)
            
            # Convert to condition name with fallback
            condition_name = self.label_to_condition.get(
                predicted_label, 
                "unknown condition"
            )
            
            # Validate confidence score
            if not 0 <= confidence <= 1:
                logger.warning(f"Unusual confidence score: {confidence}")
            
            logger.info(f"Identified condition: {condition_name} with confidence: {confidence:.4f}")
            return condition_name, confidence
            
        except Exception as e:
            logger.exception(f"Error in condition identification: {str(e)}")
            raise RuntimeError(f"Condition identification failed: {str(e)}") from e

    def _validate_model_output(self, confidence: float) -> bool:
        """
        Validate model output confidence score.

        Args:
            confidence (float): Model confidence score

        Returns:
            bool: True if confidence score is valid
        """
        return 0 <= confidence <= 1