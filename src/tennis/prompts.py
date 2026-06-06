"""Prompt builders for tennis temporal QA."""

from __future__ import annotations


def build_tennis_prompt(context: str, question: str) -> str:
    """Build the TISER-style tennis temporal QA prompt."""
    return "\n".join(
        [
            "You are given a temporal context from the tennis domain and a question.",
            "Reason only over the events explicitly stated in the context.",
            "Use the temporal order, durations, overlaps, or tournament sequence when needed.",
            "",
            "Temporal context:",
            context.strip(),
            "",
            "Question:",
            question.strip(),
            "",
            "Return the final answer in the required format.",
        ]
    )


def build_standard_prompt(context: str, question: str) -> str:
    """Build a direct-answer prompt for later prompt-format comparisons."""
    return "\n".join(
        [
            "You are given a temporal context from the tennis domain and a question.",
            "Reason only over the events explicitly stated in the context.",
            "",
            "Temporal context:",
            context.strip(),
            "",
            "Question:",
            question.strip(),
            "",
            "Return only the concise final answer.",
        ]
    )
