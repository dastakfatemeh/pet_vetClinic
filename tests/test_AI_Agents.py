import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import torch
from AI_Agents import (
    app, 
    AppointmentResponse, 
    AppointmentResponseEnum,
    CONFIDENCE_THRESHOLD,
    SIMILARITY_THRESHOLD
)

# Create test client
client = TestClient(app)

@pytest.fixture
def mock_models():
    """Mock all ML models"""
    with patch('AI_Agents.AutoModel.from_pretrained') as mock_model, \
         patch('AI_Agents.AutoTokenizer.from_pretrained') as mock_tokenizer:
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        yield mock_model, mock_tokenizer

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    with patch('AI_Agents.QdrantClient') as mock_client:
        mock_client.return_value = Mock()
        yield mock_client

class TestConversationEndpoint:
    """Test the /converse endpoint"""
    
    def test_high_confidence_response(self, mock_models, mock_qdrant):
        with patch('AI_Agents.classification_agent') as mock_class, \
             patch('AI_Agents.retrieval_agent') as mock_ret, \
             patch('AI_Agents.communication_agent') as mock_comm:
            
            # Setup complete mock chain
            mock_class.identify_condition.return_value = ("ear infections", 0.95)
            mock_ret.find_similar_cases.return_value = [Mock(score=0.9, payload={"text": "test case"})]
            mock_comm.output_vet_assistant.return_value = "Test explanation"
            
            response = client.post(
                "/converse",
                json={"user_input": "My dog has an ear infection"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["condition_identified"] == "ear infections"
            assert data["confidence_score"] > CONFIDENCE_THRESHOLD

    @pytest.mark.parametrize("invalid_input,expected_status", [
        ("", 422),
        (" ", 422),
        (None, 422),
        ("a" * 1001, 400)  # Very long input
    ])
    def test_invalid_inputs(self, invalid_input, expected_status):
        response = client.post(
            "/converse",
            json={"user_input": invalid_input}
        )
        assert response.status_code == expected_status

class TestAppointmentEndpoint:
    """Test the /appointment_response endpoint"""
    
    @pytest.mark.parametrize("input_response,expected_status", [
        ("yes", 200),
        ("YES", 200),
        ("y", 200),
        ("no", 200),
        ("NO", 200),
        ("n", 200),
        ("  Yes  ", 200),
        ("maybe", 422),
        ("", 422),
        ("invalid", 422)
    ])
    def test_appointment_responses(self, input_response, expected_status):
        response = client.post(
            "/appointment_response",
            json={"wants_appointment": input_response}
        )
        
        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            if input_response.lower() in ['yes', 'y']:
                assert "calendar_url" in data
            else:
                assert "message" in data

    def test_calendly_link(self):
        response = client.post(
            "/appointment_response",
            json={"wants_appointment": "yes"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "calendly.com" in data["calendar_url"]

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_json(self):
        response = client.post(
            "/converse",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_field(self):
        response = client.post("/converse", json={})
        assert response.status_code == 422

    def test_server_error_handling(self, mock_models, mock_qdrant):
        with patch('AI_Agents.classification_agent') as mock_class:
            mock_class.identify_condition.side_effect = Exception("Test error")
            
            response = client.post(
                "/converse",
                json={"user_input": "test input"}
            )
            
            assert response.status_code == 500