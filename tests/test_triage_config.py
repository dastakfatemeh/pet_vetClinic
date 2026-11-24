import pytest

from Agents.TriageAgent import SystemOneHTTPBackend
from Agents.exceptions import AgentInitializationError
from Agents.triage_config import build_backend, build_triage_agent, load_triage_settings

CLOUDFLARE = {"CLOUDFLARE_ACCOUNT_ID": "acct123", "CLOUDFLARE_API_TOKEN": "cf-token"}


class TestLoadSettings:
    def test_defaults_to_clef_on_workers_ai(self):
        settings = load_triage_settings({})
        assert (settings.model, settings.provider) == ("clef", "workers_ai")
        assert (settings.emergency_threshold, settings.disagreement_threshold) == (0.5, 0.5)

    def test_jev_preset(self):
        settings = load_triage_settings({"TRIAGE_MODEL": "jev"})
        assert (settings.model, settings.provider) == ("jev", "typesafe")
        assert (settings.emergency_threshold, settings.disagreement_threshold) == (0.5, 0.8)

    @pytest.mark.parametrize(
        "model,emergency,disagreement",
        [("clef", 0.5, 0.5), ("clef-flash", 0.8, 0.5), ("CLEF-Flash", 0.8, 0.5)],
    )
    def test_clef_presets(self, model, emergency, disagreement):
        settings = load_triage_settings({"TRIAGE_MODEL": model})
        assert settings.provider == "workers_ai"
        assert (settings.emergency_threshold, settings.disagreement_threshold) == (emergency, disagreement)

    def test_d1_preset(self):
        settings = load_triage_settings({"TRIAGE_MODEL": "d1", "LIQUID_API_KEY": "lq-key"})
        assert (settings.model, settings.provider, settings.liquid_api_key) == ("d1", "liquid", "lq-key")
        assert (settings.emergency_threshold, settings.disagreement_threshold) == (0.5, 0.5)

    @pytest.mark.parametrize("value", ["none", "off", "disabled"])
    def test_can_turn_off(self, value):
        assert load_triage_settings({"TRIAGE_MODEL": value}) is None

    def test_threshold_overrides(self):
        settings = load_triage_settings(
            {"TRIAGE_MODEL": "clef", "TRIAGE_EMERGENCY_THRESHOLD": "0.7", "TRIAGE_DISAGREEMENT_THRESHOLD": "0.6"}
        )
        assert (settings.emergency_threshold, settings.disagreement_threshold) == (0.7, 0.6)

    @pytest.mark.parametrize(
        "env",
        [
            {"TRIAGE_MODEL": "gpt"},
            {"TRIAGE_MODEL": "jev", "TRIAGE_PROVIDER": "workers_ai"},
            {"TRIAGE_MODEL": "clef", "TRIAGE_PROVIDER": "typesafe"},
            {"TRIAGE_MODEL": "d1", "TRIAGE_PROVIDER": "workers_ai"},
            {"TRIAGE_MODEL": "jev", "TRIAGE_EMERGENCY_THRESHOLD": "high"},
            {"TRIAGE_DISAGREEMENT_THRESHOLD": "1.5"},
        ],
    )
    def test_rejects_bad_config(self, env):
        with pytest.raises(AgentInitializationError):
            load_triage_settings(env)


class TestBuildBackend:
    def test_workers_ai_url_and_auth(self):
        settings = load_triage_settings({"TRIAGE_MODEL": "clef-flash", **CLOUDFLARE})
        backend = build_backend(settings, {})
        assert isinstance(backend, SystemOneHTTPBackend)
        assert backend.url == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef-flash"
        assert backend.model == "clef-flash"
        assert backend.client.headers["Authorization"] == "Bearer cf-token"

    def test_workers_ai_needs_credentials(self):
        settings = load_triage_settings({"TRIAGE_MODEL": "clef"})
        with pytest.raises(AgentInitializationError):
            build_backend(settings, {})

    def test_self_hosted(self):
        settings = load_triage_settings(
            {"TRIAGE_MODEL": "clef", "TRIAGE_PROVIDER": "self_hosted", "CLEF_BASE_URL": "http://gpu-box:30000/"}
        )
        backend = build_backend(settings, {})
        assert backend.url == "http://gpu-box:30000/v1/systemone"
        assert "Authorization" not in backend.client.headers

    def test_self_hosted_needs_url(self):
        settings = load_triage_settings({"TRIAGE_MODEL": "clef", "TRIAGE_PROVIDER": "self_hosted"})
        with pytest.raises(AgentInitializationError):
            build_backend(settings, {})

    def test_liquid_url_and_auth(self):
        settings = load_triage_settings({"TRIAGE_MODEL": "d1", "LIQUID_API_KEY": "lq-key"})
        backend = build_backend(settings, {})
        assert isinstance(backend, SystemOneHTTPBackend)
        assert backend.url == "https://api.liquid.ai/decisions/v1/systemone"
        assert backend.model == "d1"
        assert backend.client.headers["Authorization"] == "Bearer lq-key"

    def test_liquid_needs_api_key(self):
        with pytest.raises(AgentInitializationError):
            build_backend(load_triage_settings({"TRIAGE_MODEL": "d1"}), {})

    def test_jev_needs_api_key(self):
        with pytest.raises(AgentInitializationError):
            build_backend(load_triage_settings({"TRIAGE_MODEL": "jev"}), {})

    def test_agent_carries_model_thresholds(self):
        settings = load_triage_settings({"TRIAGE_MODEL": "clef-flash", **CLOUDFLARE})
        agent = build_triage_agent(settings, {})
        assert (agent.name, agent.emergency_threshold, agent.disagreement_threshold) == ("clef-flash", 0.8, 0.5)
