# Tennis Smoke Results

These results summarize completed Base Qwen tennis smoke evaluations from
`results/tennis_domain_adaptation/scored/`.

| Condition | Model | Prompt style | Limit | EM | F1 | Malformed count | Malformed rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `base_qwen_standard_smoke_10` | `Qwen/Qwen2.5-0.5B-Instruct` | standard | 10 | 0.60 | 0.68 | 0 | 0.00 |
| `base_qwen_standard_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | standard | 100 | 0.38 | 0.45 | 0 | 0.00 |
| `base_qwen_tiser_smoke_10` | `Qwen/Qwen2.5-0.5B-Instruct` | tiser | 10 | 0.60 | 0.74 | 0 | 0.00 |
| `base_qwen_tiser_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | tiser | 100 | 0.38 | 0.40 | 4 | 0.04 |

## Interpretation

The evaluation pipeline is functional. On the 100-example smoke setting,
standard prompting reaches 0.38 EM / 0.45 F1, while TISER-style prompting
reaches 0.38 EM / 0.40 F1 with 4 invalid or malformed predictions.

TISER-style prompting alone does not improve the 0.5B base model in this smoke
setting. These results are preliminary and should not be interpreted as final
TISER performance.

## Strict-Evaluation Note

The tennis evaluator uses strict answer-type expectations. Yes/no questions
require explicit `Yes` or `No` answers; relational outputs such as `after` are
counted wrong even when they are semantically interpretable from the question.
