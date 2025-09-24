import torch
import numpy as np
import logging
from typing import Union
from .exceptions import AgentError

# Configure logging
logger = logging.getLogger(__name__)


class VetBERTMixin:
    """
    Mixin class that provides VetBERT embedding functionality.

    This mixin adds methods for generating embeddings using the VetBERT model,
    which has been specifically trained on veterinary text data. It handles
    the conversion between PyTorch tensors and numpy arrays, with proper
    error handling for device management and model operations.
    """

    def get_vetbert_embeddings(
        self, user_input: str, return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embeddings using VetBERT model.

        Args:
            user_input (str): Text to generate embeddings for
            return_numpy (bool): If True, returns numpy array; if False, returns PyTorch tensor

        Returns:
            Union[torch.Tensor, np.ndarray]: Embedding vector of shape (768,)

        Raises:
            ValueError: If input is invalid
            AgentError: If embedding generation fails
            RuntimeError: If CUDA operations fail
        """
        # Input validation
        if not user_input or not isinstance(user_input, str):
            logger.error("Invalid input provided to get_vetbert_embeddings")
            raise ValueError("Input must be a non-empty string")

        try:
            # Ensure model is in eval mode
            self.model.eval()

            # Tokenize input
            inputs = self.tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move inputs to correct device
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except RuntimeError as e:
                logger.error(f"Failed to move inputs to device: {e}")
                raise AgentError("Device operation failed")

            # Generate embeddings
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                except RuntimeError as e:
                    logger.error(f"Model inference failed: {e}")
                    raise AgentError("Embedding generation failed")

            # Convert to numpy if requested
            if return_numpy:
                try:
                    embeddings = embeddings.cpu().numpy()
                except RuntimeError as e:
                    logger.error(f"Failed to convert embeddings to numpy: {e}")
                    raise AgentError("Numpy conversion failed")

            logger.debug(
                f"Successfully generated embeddings of shape {embeddings.shape}"
            )
            return embeddings

        except Exception as e:
            logger.exception(f"Unexpected error in get_vetbert_embeddings: {e}")
            raise AgentError("Failed to generate embeddings")

    def _validate_model_device(self) -> None:
        """
        Validate that model and device are properly configured.

        Raises:
            AgentError: If model or device configuration is invalid
        """
        if not hasattr(self, "model"):
            raise AgentError("Model not initialized")
        if not hasattr(self, "device"):
            raise AgentError("Device not initialized")
