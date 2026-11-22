"""
Configuration for the triage layer: which decision model runs behind TriageAgent, where it is served,
and the thresholds that go with it.

Environment variables (read from .env at startup):

    TRIAGE_MODEL                  clef (default) | clef-flash | jev | none
    TRIAGE_PROVIDER               For clef / clef-flash: workers_ai (default) | self_hosted

    TYPESAFE_API_KEY              jev: TypeSafe API key (read by typesafe-sdk)
    CLOUDFLARE_ACCOUNT_ID         workers_ai: Cloudflare account ID
    CLOUDFLARE_API_TOKEN          workers_ai: API token with Workers AI permission
    CLEF_BASE_URL                 self_hosted: base URL of a SystemOne-compatible server (POST {url}/v1/systemone)
    CLEF_API_KEY                  self_hosted: optional bearer token

    TRIAGE_EMERGENCY_THRESHOLD    Optional override of the model's emergency threshold
    TRIAGE_DISAGREEMENT_THRESHOLD Optional override of the model's disagreement threshold
    TRIAGE_TIMEOUT_SECONDS        Per-request timeout (default 3)

Default thresholds come from experiments/decider/README.md. The Clef values were chosen on the same
445 evaluation messages, so treat them as provisional until they are checked on fresh traffic.
"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from .exceptions import AgentInitializationError
from .TriageAgent import SystemOneHTTPBackend, TriageAgent, TypeSafeBackend

DEFAULT_MODEL = "clef"  # free weights; cheap on Workers AI. Set TRIAGE_MODEL=jev to use TypeSafe instead
WORKERS_AI_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/{model}"


@dataclass(frozen=True)
class ModelPreset:
    emergency_threshold: float
    disagreement_threshold: float
    providers: tuple  # first one is the default


PRESETS = {
    # Jev: production defaults the triage layer shipped with
    "jev": ModelPreset(emergency_threshold=0.5, disagreement_threshold=0.8, providers=("typesafe",)),
    # Clef 27B: matched Jev at a 0.5 disagreement threshold; 0.7-0.8 emergency also tested clean (vet review pending)
    "clef": ModelPreset(emergency_threshold=0.5, disagreement_threshold=0.5, providers=("workers_ai", "self_hosted")),
    # Clef-flash 9.4B: flags too many emergencies at 0.5; 0.8 kept 12/12 recall with no false alarms
    "clef-flash": ModelPreset(
        emergency_threshold=0.8, disagreement_threshold=0.5, providers=("workers_ai", "self_hosted")
    ),
}


@dataclass(frozen=True)
class TriageSettings:
    model: str
    provider: str
    emergency_threshold: float
    disagreement_threshold: float
    timeout: float = 3.0
    cloudflare_account_id: Optional[str] = None
    cloudflare_api_token: Optional[str] = None
    clef_base_url: Optional[str] = None
    clef_api_key: Optional[str] = None


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as e:
        raise AgentInitializationError(f"{key} must be a number, got {raw!r}") from e
    if key != "TRIAGE_TIMEOUT_SECONDS" and not 0.0 <= value <= 1.0:
        raise AgentInitializationError(f"{key} must be between 0 and 1, got {value}")
    return value


def load_triage_settings(env: Optional[Mapping[str, str]] = None) -> Optional[TriageSettings]:
    """Read triage settings from the environment. Returns None when triage is turned off (TRIAGE_MODEL=none)."""
    env = os.environ if env is None else env
    model = env.get("TRIAGE_MODEL", DEFAULT_MODEL).strip().lower() or DEFAULT_MODEL
    if model in ("none", "off", "disabled"):
        return None
    if model not in PRESETS:
        raise AgentInitializationError(f"TRIAGE_MODEL must be one of {', '.join(PRESETS)} or none, got {model!r}")

    preset = PRESETS[model]
    provider = env.get("TRIAGE_PROVIDER", "").strip().lower() or preset.providers[0]
    if provider not in preset.providers:
        raise AgentInitializationError(f"TRIAGE_PROVIDER for {model} must be one of {', '.join(preset.providers)}")

    return TriageSettings(
        model=model,
        provider=provider,
        emergency_threshold=_float(env, "TRIAGE_EMERGENCY_THRESHOLD", preset.emergency_threshold),
        disagreement_threshold=_float(env, "TRIAGE_DISAGREEMENT_THRESHOLD", preset.disagreement_threshold),
        timeout=_float(env, "TRIAGE_TIMEOUT_SECONDS", 3.0),
        cloudflare_account_id=env.get("CLOUDFLARE_ACCOUNT_ID") or None,
        cloudflare_api_token=env.get("CLOUDFLARE_API_TOKEN") or None,
        clef_base_url=env.get("CLEF_BASE_URL") or None,
        clef_api_key=env.get("CLEF_API_KEY") or None,
    )


def build_backend(settings: TriageSettings, env: Optional[Mapping[str, str]] = None):
    """Create the backend for the configured model and provider. Raises AgentInitializationError if misconfigured."""
    env = os.environ if env is None else env
    if settings.provider == "typesafe":
        if not env.get("TYPESAFE_API_KEY"):
            raise AgentInitializationError("TYPESAFE_API_KEY is not set")
        try:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
        except ImportError as e:
            raise AgentInitializationError("typesafe-sdk is not installed (requires Python >= 3.10)") from e
        return TypeSafeBackend(
            AsyncTypeSafeClient(), retry=RetryPolicy(max_retries=2, backoff_max=0.5, timeout=settings.timeout)
        )

    if settings.provider == "workers_ai":
        if not settings.cloudflare_account_id or not settings.cloudflare_api_token:
            raise AgentInitializationError(
                f"CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set to use {settings.model} on Workers AI "
                "(or set TRIAGE_PROVIDER=self_hosted, or TRIAGE_MODEL=jev)"
            )
        url = WORKERS_AI_URL.format(account_id=settings.cloudflare_account_id, model=settings.model)
        headers = {"Authorization": f"Bearer {settings.cloudflare_api_token}"}
        return SystemOneHTTPBackend(url, settings.model, headers=headers, timeout=settings.timeout)

    if settings.provider == "self_hosted":
        if not settings.clef_base_url:
            raise AgentInitializationError("CLEF_BASE_URL must be set for a self-hosted Clef server")
        url = settings.clef_base_url.rstrip("/") + "/v1/systemone"
        headers = {"Authorization": f"Bearer {settings.clef_api_key}"} if settings.clef_api_key else None
        return SystemOneHTTPBackend(url, settings.model, headers=headers, timeout=settings.timeout)

    raise AgentInitializationError(f"Unknown triage provider {settings.provider!r}")


def build_triage_agent(settings: TriageSettings, env: Optional[Mapping[str, str]] = None) -> TriageAgent:
    return TriageAgent(
        build_backend(settings, env),
        name=settings.model,
        emergency_threshold=settings.emergency_threshold,
        disagreement_threshold=settings.disagreement_threshold,
    )
