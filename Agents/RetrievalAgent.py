from .BetBertMixin import VetBERTMixin
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import logging
from .exceptions import AgentInitializationError, SearchOperationError, DatabaseConnectionError

# Configure logging
logger = logging.getLogger(__name__)

class RetrievalAgent(VetBERTMixin):
    """
    Agent responsible for retrieving similar veterinary cases from a vector database.
    """
    
    def __init__(self, model, tokenizer, device, qdrant_client, collection_name: str):
        try:
            self.device = device
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.client = qdrant_client
            
            if not collection_name:
                raise ValueError("Collection name cannot be empty")
            self.collection_name = collection_name
            
            # Verify collection exists
            self._verify_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalAgent: {e}")
            raise AgentInitializationError("Failed to initialize retrieval agent") 

    def find_similar_cases(self, user_input: str, condition: str, limit: int = 3) -> List[Record]:
        """Find similar veterinary cases based on input text and condition."""
        # Input validation
        if not user_input or not isinstance(user_input, str):
            raise ValueError("User input must be a non-empty string")
        if not condition or not isinstance(condition, str):
            raise ValueError("Condition must be a non-empty string")
        if limit < 1:
            raise ValueError("Limit must be positive")

        try:
            # Generate embeddings for semantic search
            query_vector = self.get_vetbert_embeddings(user_input, return_numpy=True)
            
            # Set up condition filter
            condition_filter = Filter(
                must=[FieldCondition(key="condition", match=MatchValue(value=condition))]
            )
            
            # Perform similarity search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector[0],
                limit=limit,
                query_filter=condition_filter,
                with_payload=True
            )
            
            if not results:
                logger.warning(f"No cases found for condition '{condition}'")
                return []
            
            logger.info(f"Found {len(results)} similar cases for condition: {condition}")
            return results
            
        except ConnectionError as e:
            logger.error("Failed to connect to vector database")
            raise DatabaseConnectionError("Failed to connect to database")
        except Exception as e:
            logger.exception(f"Error in retrieval: {e}")
            raise SearchOperationError("Failed to search for similar cases")

    def _verify_collection(self) -> None:
        """Verify that the specified collection exists in Qdrant."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to verify collection '{self.collection_name}': {e}")
            raise AgentInitializationError("Failed to verify collection existence")