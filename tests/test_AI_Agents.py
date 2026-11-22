import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import torch
from AI_Agents import (
    app,
    AppointmentResponse,
    AppointmentResponseEnum,
    CONFIDENCE_THRESHOLD,
    SIMILARITY_THRESHOLD,
)
from Agents.TriageAgent import TriageAgent, TriageResult

# Create test client
client = TestClient(app)


@pytest.fixture
def mock_models():
    """Mock all ML models"""
    with (
        patch("AI_Agents.AutoModel.from_pretrained") as mock_model,
        patch("AI_Agents.AutoTokenizer.from_pretrained") as mock_tokenizer,
    ):
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        yield mock_model, mock_tokenizer


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    with patch("AI_Agents.QdrantClient") as mock_client:
        mock_client.return_value = Mock()
        yield mock_client


class TestConversationEndpoint:
    """Test the /converse endpoint"""

    def test_high_confidence_response(self, mock_models, mock_qdrant):
        with (
            patch("AI_Agents.classification_agent") as mock_class,
            patch("AI_Agents.retrieval_agent") as mock_ret,
            patch("AI_Agents.communication_agent") as mock_comm,
        ):
            # Setup complete mock chain
            mock_class.identify_condition.return_value = ("ear infections", 0.95)
            mock_ret.find_similar_cases.return_value = [Mock(score=0.9, payload={"text": "test case"})]
            mock_comm.output_vet_assistant.return_value = "Test explanation"

            response = client.post("/converse", json={"user_input": "My dog has an ear infection"})

            assert response.status_code == 200
            data = response.json()
            assert data["condition_identified"] == "ear infections"
            assert data["confidence_score"] > CONFIDENCE_THRESHOLD

    @pytest.mark.parametrize(
        "invalid_input,expected_status",
        [("", 422), (" ", 422), (None, 422), ("a" * 1001, 400)],  # Very long input
    )
    def test_invalid_inputs(self, invalid_input, expected_status):
        response = client.post("/converse", json={"user_input": invalid_input})
        assert response.status_code == expected_status


def make_triage(emergency=0.05, condition="ear infections", condition_confidence=0.95):
    """Build a triage result as returned by TriageAgent.assess"""
    return Mock(
        emergency_probability=emergency,
        condition=condition,
        condition_confidence=condition_confidence,
        in_scope=condition != "none of these",
    )


class TestTriageChecks:
    """Test the emergency and out-of-scope checks in /converse"""

    @pytest.fixture
    def agents(self):
        with (
            patch("AI_Agents.classification_agent") as mock_class,
            patch("AI_Agents.retrieval_agent") as mock_ret,
            patch("AI_Agents.communication_agent") as mock_comm,
            patch("AI_Agents.triage_agent") as mock_triage,
        ):
            mock_class.identify_condition.return_value = ("ear infections", 0.95)
            mock_ret.find_similar_cases.return_value = [Mock(score=0.9, payload={"text": "test case"})]
            mock_comm.output_vet_assistant.return_value = "Test explanation"
            mock_triage.assess = AsyncMock(return_value=make_triage())
            yield mock_class, mock_ret, mock_comm, mock_triage

    def test_emergency_skips_retrieval(self, agents):
        _, mock_ret, _, mock_triage = agents
        mock_triage.assess.return_value = make_triage(emergency=0.97)

        response = client.post("/converse", json={"user_input": "My cat ate a lily and is drooling"})

        assert response.status_code == 200
        data = response.json()
        assert data["condition_identified"] == "emergency"
        assert data["escalation_reason"] == "emergency"
        assert "urgent" in data["conversation"]
        mock_ret.find_similar_cases.assert_not_called()

    def test_out_of_scope_offers_appointment(self, agents):
        _, mock_ret, _, mock_triage = agents
        mock_triage.assess.return_value = make_triage(condition="none of these", condition_confidence=0.99)

        response = client.post("/converse", json={"user_input": "What are your opening hours?"})

        data = response.json()
        assert data["condition_identified"] == "uncertain"
        assert data["escalation_reason"] == "out_of_scope"
        assert "yes/no" in data["conversation"]
        mock_ret.find_similar_cases.assert_not_called()

    def test_confident_disagreement_offers_appointment(self, agents):
        _, mock_ret, _, mock_triage = agents
        mock_triage.assess.return_value = make_triage(condition="skin irritations", condition_confidence=0.9)

        data = client.post("/converse", json={"user_input": "Itchy red bumps on his belly"}).json()

        assert data["condition_identified"] == "uncertain"
        assert data["escalation_reason"] == "model_disagreement"
        mock_ret.find_similar_cases.assert_not_called()

    def test_unconfident_disagreement_keeps_classifier_answer(self, agents):
        mock_triage = agents[3]
        mock_triage.assess.return_value = make_triage(condition="skin irritations", condition_confidence=0.6)

        data = client.post("/converse", json={"user_input": "Scratching his ears"}).json()

        assert data["condition_identified"] == "ear infections"
        assert data["triage"]["condition"] == "skin irritations"

    def test_low_confidence_still_escalates_with_triage(self, agents):
        mock_class = agents[0]
        mock_class.identify_condition.return_value = ("ear infections", 0.5)

        data = client.post("/converse", json={"user_input": "Something seems off"}).json()

        assert data["condition_identified"] == "uncertain"
        assert data["escalation_reason"] == "low_confidence"

    def test_triage_failure_falls_back_to_classifier(self, agents):
        mock_triage = agents[3]
        mock_triage.assess.side_effect = Exception("TypeSafe unavailable")

        response = client.post("/converse", json={"user_input": "My dog has an ear infection"})

        assert response.status_code == 200
        data = response.json()
        assert data["condition_identified"] == "ear infections"
        assert data["triage"] is None


class TestConfiguredTriageModel:
    """The gate uses the configured model's own thresholds"""

    def make_agent(self, name, emergency_threshold, disagreement_threshold, result):
        backend = Mock()
        backend.assess = AsyncMock(return_value=result)
        return TriageAgent(
            backend, name=name, emergency_threshold=emergency_threshold, disagreement_threshold=disagreement_threshold
        )

    @pytest.fixture
    def pipeline(self):
        with (
            patch("AI_Agents.classification_agent") as mock_class,
            patch("AI_Agents.retrieval_agent") as mock_ret,
            patch("AI_Agents.communication_agent") as mock_comm,
        ):
            mock_class.identify_condition.return_value = ("ear infections", 0.95)
            mock_ret.find_similar_cases.return_value = [Mock(score=0.9, payload={"text": "test case"})]
            mock_comm.output_vet_assistant.return_value = "Test explanation"
            yield mock_ret

    def test_clef_overrules_at_lower_confidence(self, pipeline):
        # 0.6 would not overrule the classifier with Jev (0.8), but does with Clef (0.5)
        agent = self.make_agent("clef", 0.5, 0.5, TriageResult(0.1, "skin irritations", 0.6))
        with patch("AI_Agents.triage_agent", agent):
            data = client.post("/converse", json={"user_input": "Scratching a lot"}).json()

        assert data["escalation_reason"] == "model_disagreement"
        assert data["triage"]["model"] == "clef"
        pipeline.find_similar_cases.assert_not_called()

    def test_clef_flash_needs_higher_emergency_probability(self, pipeline):
        # 0.6 is an emergency at Jev's 0.5, but not at Clef-flash's 0.8
        agent = self.make_agent("clef-flash", 0.8, 0.5, TriageResult(0.6, "ear infections", 0.9))
        with patch("AI_Agents.triage_agent", agent):
            data = client.post("/converse", json={"user_input": "Shaking his head"}).json()

        assert data["condition_identified"] == "ear infections"
        assert "escalation_reason" not in data
        assert data["triage"]["model"] == "clef-flash"


class TestAppointmentEndpoint:
    """Test the /appointment_response endpoint"""

    @pytest.mark.parametrize(
        "input_response,expected_status",
        [
            ("yes", 200),
            ("YES", 200),
            ("y", 200),
            ("no", 200),
            ("NO", 200),
            ("n", 200),
            ("  Yes  ", 200),
            ("maybe", 422),
            ("", 422),
            ("invalid", 422),
        ],
    )
    def test_appointment_responses(self, input_response, expected_status):
        response = client.post("/appointment_response", json={"wants_appointment": input_response})

        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            if input_response.lower() in ["yes", "y"]:
                assert "calendar_url" in data
            else:
                assert "message" in data

    def test_calendly_link(self):
        response = client.post("/appointment_response", json={"wants_appointment": "yes"})

        assert response.status_code == 200
        data = response.json()
        assert "calendly.com" in data["calendar_url"]


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_json(self):
        response = client.post(
            "/converse",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self):
        response = client.post("/converse", json={})
        assert response.status_code == 422

    def test_server_error_handling(self, mock_models, mock_qdrant):
        with patch("AI_Agents.classification_agent") as mock_class:
            mock_class.identify_condition.side_effect = Exception("Test error")

            response = client.post("/converse", json={"user_input": "test input"})

            assert response.status_code == 500
