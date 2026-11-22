import json

import httpx
import pytest
from unittest.mock import AsyncMock, Mock

from Agents.TriageAgent import NONE_LABEL, SystemOneHTTPBackend, TriageAgent, TriageResult, TypeSafeBackend
from Agents.exceptions import AgentInitializationError, TriageError


def systemone_body(emergency=0.1, condition="parasites", confidence=0.9):
    """A SystemOne response body as Clef returns it"""
    return {
        "model": "clef",
        "answers": {
            "emergency": {"type": "noul", "noul": emergency},
            "condition": {
                "type": "choice",
                "choice": condition,
                "confidence": confidence,
                "probabilities": {condition: confidence, NONE_LABEL: round(1 - confidence, 4)},
            },
        },
        "usage": {"input_tokens": 320, "output_tokens": 0},
    }


def http_backend(handler, **kwargs):
    """SystemOneHTTPBackend whose HTTP calls go to `handler` instead of the network"""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SystemOneHTTPBackend("https://example.test/systemone", "clef", client=client, backoff=0, **kwargs)


class TestTriageAgent:
    """The model-independent wrapper"""

    @pytest.fixture
    def backend(self):
        backend = Mock()
        backend.assess = AsyncMock(return_value=TriageResult(0.2, "parasites", 0.9))
        backend.close = AsyncMock()
        return backend

    def test_requires_backend(self):
        with pytest.raises(AgentInitializationError):
            TriageAgent(None)

    def test_keeps_model_thresholds(self, backend):
        agent = TriageAgent(backend, name="clef-flash", emergency_threshold=0.8, disagreement_threshold=0.5)
        assert (agent.name, agent.emergency_threshold, agent.disagreement_threshold) == ("clef-flash", 0.8, 0.5)

    async def test_assess_delegates(self, backend):
        result = await TriageAgent(backend).assess("Dog has fleas")
        assert result.condition == "parasites"
        backend.assess.assert_awaited_once_with("Dog has fleas")

    @pytest.mark.parametrize("invalid_input", ["", None, 42])
    async def test_invalid_input(self, backend, invalid_input):
        with pytest.raises(ValueError):
            await TriageAgent(backend).assess(invalid_input)
        backend.assess.assert_not_called()

    async def test_close(self, backend):
        await TriageAgent(backend).close()
        backend.close.assert_awaited_once()

    def test_none_label_is_out_of_scope(self):
        assert not TriageResult(0.1, NONE_LABEL, 0.9).in_scope
        assert TriageResult(0.1, "parasites", 0.9).in_scope


class TestSystemOneHTTPBackend:
    """Clef / Clef-flash over Workers AI or a self-hosted SystemOne server"""

    async def test_sends_jev_compatible_request(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=systemone_body(emergency=0.97, condition="digestive issues"))

        result = await http_backend(handler).assess("Dog is retching but nothing comes up")

        assert result == TriageResult(0.97, "digestive issues", 0.9)
        assert seen["body"]["model"] == "clef"
        assert seen["body"]["state"] == {"owner_message": "Dog is retching but nothing comes up"}
        assert set(seen["body"]["questions"]) == {"emergency", "condition"}
        assert NONE_LABEL in seen["body"]["questions"]["condition"]["criteria"]

    async def test_unwraps_cloudflare_envelope(self):
        def handler(request):
            return httpx.Response(200, json={"result": systemone_body(), "success": True, "errors": []})

        result = await http_backend(handler).assess("Dog has fleas")
        assert result.condition == "parasites"

    async def test_envelope_failure_raises(self):
        def handler(request):
            return httpx.Response(200, json={"result": None, "success": False, "errors": [{"message": "bad"}]})

        with pytest.raises(TriageError):
            await http_backend(handler).assess("Dog has fleas")

    async def test_retries_server_errors(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json=systemone_body())

        result = await http_backend(handler, max_retries=2).assess("Dog has fleas")
        assert result.condition == "parasites"
        assert len(calls) == 3

    async def test_gives_up_after_retries(self):
        def handler(request):
            raise httpx.ConnectTimeout("timed out")

        with pytest.raises(TriageError):
            await http_backend(handler, max_retries=1).assess("Dog has fleas")

    async def test_client_error_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(401, text="unauthorized")

        with pytest.raises(TriageError):
            await http_backend(handler).assess("Dog has fleas")
        assert len(calls) == 1

    async def test_unexpected_shape_raises(self):
        def handler(request):
            return httpx.Response(200, json={"answers": {"emergency": {}}})

        with pytest.raises(TriageError):
            await http_backend(handler).assess("Dog has fleas")

    def test_requires_url_and_model(self):
        with pytest.raises(AgentInitializationError):
            SystemOneHTTPBackend("", "clef")


class TestTypeSafeBackend:
    """Jev through typesafe-sdk"""

    @pytest.fixture
    def typesafe_sdk(self):
        return pytest.importorskip("typesafe_sdk")  # Requires Python >= 3.10

    @pytest.fixture
    def fake_client(self):
        client = Mock()
        client.system_one = AsyncMock(
            return_value=Mock(
                nouls={"emergency": Mock(noul=0.97)},
                choices={
                    "condition": Mock(
                        choice="digestive issues", probabilities={"digestive issues": 0.8, NONE_LABEL: 0.2}
                    )
                },
            )
        )
        client.aclose = AsyncMock()
        return client

    def test_requires_client(self, typesafe_sdk):
        with pytest.raises(AgentInitializationError):
            TypeSafeBackend(None)

    async def test_maps_response(self, typesafe_sdk, fake_client):
        result = await TypeSafeBackend(fake_client).assess("Dog is retching but nothing comes up")

        assert result == TriageResult(0.97, "digestive issues", 0.8)
        state, questions = fake_client.system_one.call_args.args
        assert state == {"owner_message": "Dog is retching but nothing comes up"}
        assert set(questions) == {"emergency", "condition"}

    async def test_api_error_raises_triage_error(self, typesafe_sdk, fake_client):
        fake_client.system_one.side_effect = typesafe_sdk.TypeSafeError("boom")
        with pytest.raises(TriageError):
            await TypeSafeBackend(fake_client).assess("test input")

    async def test_close(self, typesafe_sdk, fake_client):
        await TypeSafeBackend(fake_client).close()
        fake_client.aclose.assert_awaited_once()
