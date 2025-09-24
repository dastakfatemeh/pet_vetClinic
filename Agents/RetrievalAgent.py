from .VetBertMixin import VetBERTMixin
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import logging
from qdrant_client.models import Record
from typing import List, Tuple, Dict
import numpy as np
from .exceptions import (
    AgentInitializationError,
    SearchOperationError,
    DatabaseConnectionError,
)

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalAgent(VetBERTMixin):
    """
    Agent responsible for retrieving similar veterinary cases from a vector database.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        qdrant_client,
        collection_name: str,
        max_neg: int = 4,
        percentage_margin: float = 0.95,
    ):
        """
        Initialize with TopK and PercPos parameters.

        Args:
            model: VetBERT model
            tokenizer: Associated tokenizer
            device: Computing device
            qdrant_client: Qdrant client instance
            collection_name: Name of collection
            max_neg: Maximum number of negative examples to keep
            percentage_margin: Margin for positive example selection
        """
        try:
            self.device = device
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.client = qdrant_client
            self.max_neg = max_neg
            self.percentage_margin = percentage_margin
            if not collection_name:
                raise ValueError("Collection name cannot be empty")
            self.collection_name = collection_name

            # Verify collection exists
            self._verify_collection()

        except Exception as e:
            logger.error(f"Failed to initialize RetrievalAgent: {e}")
            raise AgentInitializationError("Failed to initialize retrieval agent")

    def _mine_hard_negatives(self, query_vector: np.ndarray, cases: List[Record]) -> List[Record]:
        """
        Apply TopK-PercPos hard negative mining.

        Args:
            query_vector: Original query embedding
            cases: List of retrieved cases

        Returns:
            List[Record]: List of cases after hard negative mining
        """
        try:
            if len(cases) == 0:
                logger.warning("No cases provided for mining")
                return []

            # Get positive case (highest similarity)
            positive_case = max(cases, key=lambda x: x.score)
            positive_score = positive_case.score

            # Calculate threshold for negative cases
            max_neg_score_threshold = positive_score * self.percentage_margin

            # Find negative candidates
            negative_candidates = [
                case
                for case in cases
                if case.score <= max_neg_score_threshold
                and case.id != positive_case.id  # Use ID comparison instead of object comparison
            ]

            # Sort negatives by similarity (descending) and take top-k
            hard_negatives = sorted(negative_candidates, key=lambda x: x.score, reverse=True)[: self.max_neg]

            # Combine positive with hard negatives
            final_cases = [positive_case] + hard_negatives

            logger.info(
                f"Mining results - Positive score: {positive_score:.4f}, " f"Hard negatives: {len(hard_negatives)}"
            )
            return final_cases

        except Exception as e:
            logger.error(f"Error in hard negative mining: {e}")
            return cases

    def find_similar_cases(self, user_input: str, condition: str, limit: int = 3) -> List[Record]:
        """Find similar cases using TopK-PercPos hard negative mining."""
        try:
            # Generate query embedding
            query_vector = self.get_vetbert_embeddings(user_input, return_numpy=True)

            # Set up condition filter
            condition_filter = Filter(must=[FieldCondition(key="condition", match=MatchValue(value=condition))])

            # Get initial results
            initial_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector[0],
                limit=20,  # Get more candidates for mining
                query_filter=condition_filter,
                with_payload=True,
            )

            if not initial_results:
                logger.warning(f"No initial results found for condition '{condition}'")
                return []

            # Apply hard negative mining
            mined_results = self._mine_hard_negatives(query_vector[0], initial_results)

            # Return requested number of results
            final_results = mined_results[:limit]

            logger.info(f"Found {len(final_results)} cases after mining for condition: {condition}")
            return final_results

        except Exception as e:
            logger.exception(f"Error in retrieval: {e}")
            raise SearchOperationError(f"Failed to search for similar cases: {str(e)}")

    def _verify_collection(self) -> None:
        """Verify that the specified collection exists in Qdrant."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to verify collection '{self.collection_name}': {e}")
            raise AgentInitializationError("Failed to verify collection existence")
