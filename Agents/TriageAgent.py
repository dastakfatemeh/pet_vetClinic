import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .exceptions import AgentInitializationError, TriageError

# Configure logging
logger = logging.getLogger(__name__)

NONE_LABEL = "none of these"

# Generic veterinary descriptions of the classifier's categories, plus an explicit no-match option
CONDITION_CRITERIA = {
    "digestive issues": "Stomach or intestinal problems: vomiting, diarrhea, constipation, bloating, gas, "
    "appetite loss tied to the gut, swallowed objects.",
    "ear infections": "Problems of the ear canal or flap: head shaking, ear scratching, discharge, odor, "
    "redness or swelling of the ear, head tilt.",
    "mobility problems": "Problems moving: limping, stiffness, weakness or paralysis of limbs, reluctance to "
    "jump or climb stairs, joint or back pain.",
    "parasites": "Fleas, ticks, mites, lice, or intestinal worms and protozoa, including signs such as visible "
    "worms, flea dirt, or scooting.",
    "skin irritations": "Skin and coat problems not caused by the ear: itching, rashes, redness, hot spots, "
    "hair loss, flaking, lumps, hives or allergic swelling.",
    NONE_LABEL: "The message does not describe any of the conditions above, for example a different body "
    "system (eyes, teeth, urinary, respiratory), a poisoning, or a non-medical question.",
}

EMERGENCY_INSTRUCTIONS = (
    "A pet owner wrote `owner_message`. Does it describe a situation where the pet "
    "should be seen by a veterinarian within the next few hours (for example trouble breathing, "
    "suspected poisoning, unproductive retching with a swollen belly, inability to urinate, "
    "seizures, sudden paralysis, major trauma, or collapse)?"
)
CONDITION_INSTRUCTIONS = (
    "A pet owner wrote `owner_message`. Which condition category best matches the "
    "pet's problem? Choose the single best fit."
)


def build_questions() -> Dict[str, Dict[str, Any]]:
    """The two triage questions in SystemOne JSON format (the format Jev and Clef share)."""
    return {
        "emergency": {"type": "noul", "instructions": EMERGENCY_INSTRUCTIONS},
        "condition": {"type": "choice", "instructions": CONDITION_INSTRUCTIONS, "criteria": CONDITION_CRITERIA},
    }


@dataclass
class TriageResult:
    """A decision model's judgments about a single owner message."""

    emergency_probability: float
    condition: str
    condition_confidence: float

    @property
    def in_scope(self) -> bool:
        return self.condition != NONE_LABEL


def _validate(user_input: Any) -> None:
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Invalid input: input must be a non-empty string")


class TypeSafeBackend:
    """Jev through the typesafe-sdk client (paid API, Python >= 3.10)."""

    def __init__(self, client, retry=None):
        if client is None:
            raise AgentInitializationError("TypeSafe backend needs an AsyncTypeSafeClient")
        try:
            from typesafe_sdk import Choice, Noul, RetryPolicy
        except ImportError as e:
            raise AgentInitializationError("typesafe-sdk is not installed (requires Python >= 3.10)") from e

        self.client = client
        self.retry = retry or RetryPolicy(max_retries=2, backoff_max=0.5, timeout=3.0)
        questions = build_questions()
        self.questions = {
            "emergency": Noul(instructions=questions["emergency"]["instructions"]),
            "condition": Choice(
                instructions=questions["condition"]["instructions"], criteria=questions["condition"]["criteria"]
            ),
        }

    async def assess(self, user_input: str) -> TriageResult:
        from typesafe_sdk import TypeSafeError

        try:
            response = await self.client.system_one({"owner_message": user_input}, self.questions, retry=self.retry)
        except TypeSafeError as e:
            raise TriageError(f"TypeSafe request failed: {e}") from e

        condition = response.choices["condition"]
        return TriageResult(
            emergency_probability=response.nouls["emergency"].noul,
            condition=condition.choice,
            condition_confidence=condition.probabilities[condition.choice],
        )

    async def close(self) -> None:
        await self.client.aclose()


class SystemOneHTTPBackend:
    """
    Any decision model that serves the SystemOne JSON API over HTTP.

    Used for Clef and Clef-flash, either on Cloudflare Workers AI or self-hosted (for example SGLang's
    `/v1/systemone` route). The request is `{"model", "state", "questions"}` and the response is
    `{"model", "answers", "usage"}`, optionally wrapped in Cloudflare's `{"result": ..., "success": ...}` envelope.
    """

    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        url: str,
        model: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 3.0,
        max_retries: int = 2,
        backoff: float = 0.25,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not url or not model:
            raise AgentInitializationError("SystemOne backend needs a URL and a model name")
        self.url = url
        self.model = model
        self.max_retries = max_retries
        self.backoff = backoff
        self.questions = build_questions()
        self.client = client or httpx.AsyncClient(
            timeout=timeout, headers={"Content-Type": "application/json", **(headers or {})}
        )

    async def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                await asyncio.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                response = await self.client.post(self.url, json=body)
            except httpx.HTTPError as e:  # timeouts and connection errors
                last_error = e
                continue
            if response.status_code in self.RETRY_STATUS:
                last_error = TriageError(f"HTTP {response.status_code}: {response.text[:200]}")
                continue
            if response.status_code >= 400:
                raise TriageError(f"HTTP {response.status_code}: {response.text[:200]}")
            try:
                return response.json()
            except ValueError as e:
                raise TriageError("Response was not JSON") from e
        raise TriageError(f"Request failed after {self.max_retries + 1} attempts: {last_error}")

    async def assess(self, user_input: str) -> TriageResult:
        payload = await self._post(
            {"model": self.model, "state": {"owner_message": user_input}, "questions": self.questions}
        )

        if "result" in payload and "answers" not in payload:  # Cloudflare REST envelope
            if payload.get("success") is False:
                raise TriageError(f"Workers AI error: {payload.get('errors')}")
            payload = payload["result"] or {}
        try:
            answers = payload["answers"]
            condition = answers["condition"]
            return TriageResult(
                emergency_probability=float(answers["emergency"]["noul"]),
                condition=condition["choice"],
                condition_confidence=float(condition["probabilities"][condition["choice"]]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise TriageError(f"Unexpected response shape: {str(payload)[:200]}") from e

    async def close(self) -> None:
        await self.client.aclose()


class TriageAgent:
    """
    Agent responsible for safety checks on the owner's message before the classifier's answer is used.

    Asks a decision model, in a single request, whether the message describes a veterinary emergency and
    which of the supported condition categories (if any) it matches. The model is pluggable: Jev through
    TypeSafe, or Clef / Clef-flash through any SystemOne-compatible HTTP endpoint. Each model comes with its
    own thresholds, because their probabilities are calibrated differently (see experiments/decider/README.md).
    """

    def __init__(
        self,
        backend,
        name: str = "jev",
        emergency_threshold: float = 0.5,
        disagreement_threshold: float = 0.8,
    ):
        """
        Args:
            backend: Object with `async assess(text) -> TriageResult` and `async close()`
            name: Model name, reported in responses and logs
            emergency_threshold: Emergency probability at or above which the owner is sent to urgent care
            disagreement_threshold: Confidence needed in a different condition to overrule the classifier

        Raises:
            AgentInitializationError: If no backend is provided
        """
        if backend is None:
            logger.error("Failed to initialize TriageAgent: no backend")
            raise AgentInitializationError("Failed to initialize triage agent")
        self.backend = backend
        self.name = name
        self.emergency_threshold = emergency_threshold
        self.disagreement_threshold = disagreement_threshold

    async def assess(self, user_input: str) -> TriageResult:
        """
        Judge emergency likelihood and condition category for the owner's message.

        Raises:
            ValueError: If input is empty or invalid
            TriageError: If the decision model request fails
        """
        _validate(user_input)
        result = await self.backend.assess(user_input)
        logger.info(
            f"Triage ({self.name}): emergency={result.emergency_probability:.2f}, "
            f"condition={result.condition} ({result.condition_confidence:.2f})"
        )
        return result

    async def close(self) -> None:
        """Close the underlying client."""
        await self.backend.close()
