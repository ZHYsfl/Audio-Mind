"""
Abstract base class for frontend modules.

The frontend is responsible for initial audio understanding,
producing question-guided evidence/captions from raw audio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from audio_agent.core.errors import FrontendError
from audio_agent.core.schemas import FrontendOutput


class BaseFrontend(ABC):
    """
    Abstract base class for audio frontends.

    A frontend takes a question and audio path, and produces initial evidence.
    Concrete implementations might use:
    - Local LALM models
    - Remote API-based audio understanding services
    - Hybrid approaches
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this frontend for logging and identification."""
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str, audio_paths: list[str], question_oriented_prompt: str | None = None) -> FrontendOutput:
        """
        Process audio(s) with the given question and produce initial evidence.

        Args:
            question: The user's question about the audio
            audio_paths: List of paths to audio files (one or more)
            question_oriented_prompt: Optional customized prompt from the planner

        Returns:
            FrontendOutput with question-guided caption covering all audios

        Raises:
            FrontendError: If processing fails or input is invalid
        """
        raise NotImplementedError

    def generate_final_answer(
        self,
        question: str,
        audio_paths: list[str],
        context: dict[str, Any],
    ) -> str:
        """
        Generate the final answer using the frontend audio model.

        The frontend receives the original audio(s) and all accumulated context
        (evidence, planner trace, tool history, etc.) so the answer is grounded
        directly in the audio content.

        Args:
            question: The original user question about the audio
            audio_paths: List of paths to audio files
            context: Dictionary containing accumulated evidence and metadata

        Returns:
            The final answer string

        Raises:
            FrontendError: If generation fails or input is invalid
            NotImplementedError: If this frontend doesn't support final answer generation
        """
        raise NotImplementedError(
            f"Frontend {self.name} does not support final answer generation"
        )

    def run_followup(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> FrontendOutput:
        """
        Re-perceive selected audio artifact(s) for a planner-authored follow-up request.

        This is distinct from run(), which produces an initial question-guided caption.
        Follow-up calls should answer the planner's specific request directly.
        """
        raise NotImplementedError(
            f"Frontend {self.name} does not support follow-up re-perception"
        )

    def validate_inputs(self, question: str, audio_paths: list[str]) -> None:
        """
        Validate inputs before processing. Called by subclasses.

        Raises:
            FrontendError: If inputs are invalid
        """
        if not question or not question.strip():
            raise FrontendError(
                "Question must be a non-empty string",
                details={"question": question},
            )
        if not audio_paths or len(audio_paths) == 0:
            raise FrontendError(
                "Audio paths must contain at least one path",
                details={"audio_paths": audio_paths},
            )
        for i, path in enumerate(audio_paths):
            if not path or not path.strip():
                raise FrontendError(
                    f"Audio path at index {i} is empty",
                    details={"index": i},
                )


# Backward-compatible re-exports for existing imports.
from audio_agent.frontend.model_frontend import (  # noqa: E402
    BaseModelFrontend,
    FrontendInputFormat,
    UnifiedFrontendInput,
)

__all__ = [
    "BaseFrontend",
    "BaseModelFrontend",
    "FrontendInputFormat",
    "UnifiedFrontendInput",
]
