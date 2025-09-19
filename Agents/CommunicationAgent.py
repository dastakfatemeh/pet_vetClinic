import re
import logging
from typing import List, Optional
from .exceptions import AgentInitializationError, CommunicationError

# Configure logging
logger = logging.getLogger(__name__)

class CommunicationAgent:
    """
    Agent responsible for generating human-readable explanations of veterinary cases.
    
    Uses a text generation model to explain clinical terms and findings in simple language,
    making medical information more accessible to pet owners.
    """

    def __init__(self, model, tokenizer, device):
        """
        Initialize the communication agent.

        Args:
            model: Text generation model (T5 or similar)
            tokenizer: Associated tokenizer for the model
            device: Computing device (CPU/GPU)

        Raises:
            AgentInitializationError: If initialization fails
        """
        try:
            self.device = device
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            logger.info("Communication agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CommunicationAgent: {e}")
            raise AgentInitializationError("Failed to initialize communication agent")

    def output_vet_assistant(self, cases) -> str:
        """
        Generate explanations for veterinary cases.

        Args:
            cases: List of veterinary cases with their payloads

        Returns:
            str: Formatted output with explanations and appointment question

        Raises:
            CommunicationError: If text generation fails
            ValueError: If cases input is invalid
        """
        if not cases:
            logger.warning("No cases provided for explanation")
            return "No cases to explain."

        try:
            finding_s = ''
            for i, case in enumerate(cases):
                if 'text' not in case.payload:
                    logger.error(f"Invalid case format at index {i}")
                    continue

                prompt_template = """
                    Identify and explain clinical entities in the text provided.
                    And provide a short, clear explanation for each term.
                    example: Gastroenteritis: inflammation of the stomach and intestines causing vomiting and diarrhea.
                    Text: {clinical_text}"""
                
                try:
                    # Generate explanation for each case
                    prompt = prompt_template.format(clinical_text=case.payload['text'])
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        max_length=512, 
                        truncation=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.9
                    )
                    
                    decoded_output = self.tokenizer.decode(
                        outputs[0], 
                        skip_special_tokens=True
                    )
                    
                    finding_s = (
                        f"{len(cases)-i}.{case.payload['text']}\n"
                        f"{decoded_output}\n"
                    ) + finding_s
                    
                except Exception as e:
                    logger.error(f"Failed to process case {i}: {e}")
                    continue

            # Add appointment question
            appointment_question = (
                "\n\nWould you like to schedule an appointment with our veterinary team "
                "to address these health findings and discuss the most effective "
                "treatment options for your pet? (please provide yes or no)"
            )
            
            # Combine all parts
            final_output = (
                f"Here are my findings: (Symptoms, treatment, clinical explanation)"
                f"\n\n{finding_s}{appointment_question}"
            )
            
            return self._deduplicate_numbered_sections(final_output)

        except Exception as e:
            logger.exception(f"Error generating output: {e}")
            raise CommunicationError("Failed to generate explanation")

    def _deduplicate_numbered_sections(self, text: str) -> str:
        """
        Remove duplicate sections from the formatted output.

        Args:
            text (str): Text to deduplicate

        Returns:
            str: Deduplicated text with preserved formatting

        Raises:
            ValueError: If input text is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        try:
            blocks = re.split(r'(?=\d+\.)', text)
            seen = set()
            unique_blocks = []
            
            for block in blocks:
                cleaned_block = block.strip()
                if cleaned_block and cleaned_block not in seen:
                    unique_blocks.append(cleaned_block)
                    seen.add(cleaned_block)
                    
            return '\n'.join(unique_blocks)
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return text  # Return original text if deduplication fails