"""Small deterministic smoke checks for tennis answer normalization/scoring."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tennis.eval import score_prediction_rows
from src.tennis.normalize import (
    normalize_tennis_answer,
    tennis_exact_match,
    tennis_token_f1,
)


CASES = [
    ("8 minutes", "8 minute"),
    ("The semifinal", "semifinal"),
    ("No. 1", "#1"),
    ("Yes", "yes."),
]


def main() -> None:
    for pred, gold in CASES:
        em = tennis_exact_match(pred, gold)
        f1 = tennis_token_f1(pred, gold)
        print(
            f"{pred!r} vs {gold!r}: "
            f"normalized=({normalize_tennis_answer(pred)!r}, {normalize_tennis_answer(gold)!r}), "
            f"EM={em}, F1={f1:.3f}"
        )
        assert em == 1 or f1 >= 0.8

    metrics = score_prediction_rows(
        [
            {
                "question_id": "smoke_a",
                "gold": "Yes",
                "pred_answer": "yes.",
                "category": "yes_no_before_after",
            },
            {
                "question_id": "smoke_b",
                "answer": "No. 1",
                "raw_generation": "<reasoning>x</reasoning><answer>#1</answer>",
                "category": "tennis_ranking_or_date",
            },
        ]
    )
    assert metrics["overall"]["n"] == 2
    assert metrics["overall"]["em"] == 1.0
    assert metrics["overall"]["malformed_count"] == 0
    print(
        "Aggregate smoke: "
        f"n={metrics['overall']['n']}, "
        f"EM={metrics['overall']['em']:.3f}, "
        f"F1={metrics['overall']['f1']:.3f}, "
        f"malformed={metrics['overall']['malformed_count']}"
    )


if __name__ == "__main__":
    main()
