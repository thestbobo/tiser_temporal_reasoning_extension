# Tennis Dataset Split Report

- Total examples: 1122
- Seed: 42
- Requested ratios: train 0.7, dev 0.1, test 0.2

## Split Counts

| Split | Examples |
| --- | ---: |
| train | 785 |
| dev | 113 |
| test | 224 |

## Category Distribution Per Split

### train

| Category | Examples |
| --- | ---: |
| duration_minutes | 103 |
| immediate_before_after | 137 |
| other_temporal | 12 |
| overlap_while_during | 81 |
| tennis_injury_or_medical | 1 |
| tournament_round_sequence | 13 |
| which_first_last | 165 |
| yes_no_before_after | 273 |

### dev

| Category | Examples |
| --- | ---: |
| duration_minutes | 15 |
| immediate_before_after | 20 |
| other_temporal | 2 |
| overlap_while_during | 11 |
| tennis_injury_or_medical | 1 |
| tournament_round_sequence | 2 |
| which_first_last | 23 |
| yes_no_before_after | 39 |

### test

| Category | Examples |
| --- | ---: |
| duration_minutes | 29 |
| immediate_before_after | 39 |
| other_temporal | 3 |
| overlap_while_during | 23 |
| tennis_injury_or_medical | 1 |
| tournament_round_sequence | 4 |
| which_first_last | 47 |
| yes_no_before_after | 78 |

## Answer Distribution Per Split

### train

| Answer | Examples |
| --- | ---: |
| Yes | 211 |
| No | 147 |
| 45 minutes | 14 |
| 7 minutes | 9 |
| 9 minutes | 8 |
| 12 minutes | 7 |
| 14 minutes | 5 |
| 15 minutes | 5 |
| Nadal broke serve | 5 |
| The semifinal | 5 |
| 20 minutes | 4 |
| 21 minutes | 4 |
| 25 minutes | 4 |
| Djokovic broke serve | 4 |
| 0 minutes | 3 |
| 13 minutes | 3 |
| 3 minutes | 3 |
| 30 minutes | 3 |
| 40 minutes | 3 |
| Alcaraz broke serve | 3 |
| Alcaraz saved a break point | 3 |
| Djokovic called the trainer | 3 |
| Djokovic saved a break point | 3 |
| Medvedev broke serve | 3 |
| Rune broke serve | 3 |

### dev

| Answer | Examples |
| --- | ---: |
| Yes | 32 |
| No | 18 |
| 12 minutes | 3 |
| 15 minutes | 3 |
| Rublev broke serve | 2 |
| Sinner broke serve | 2 |
| The semifinal | 2 |
| 18 minutes | 1 |
| 2 | 1 |
| 20 minutes | 1 |
| 25 minutes | 1 |
| 3 minutes | 1 |
| 40 minutes | 1 |
| 45 minutes | 1 |
| 5 minutes | 1 |
| 60 minutes | 1 |
| 9 minutes | 1 |
| A rain delay stopped play | 1 |
| De Minaur took a medical timeout | 1 |
| Djokovic held serve | 1 |
| Djokovic saved two break points | 1 |
| Fritz forced a tie-break | 1 |
| Fritz held his own serve | 1 |
| Fritz requested new shoes | 1 |
| He sat down for an extended changeover | 1 |

### test

| Answer | Examples |
| --- | ---: |
| Yes | 66 |
| No | 35 |
| 45 minutes | 4 |
| 11 minutes | 3 |
| 12 minutes | 3 |
| 10 minutes | 2 |
| 15 minutes | 2 |
| 30 minutes | 2 |
| 35 minutes | 2 |
| 5 minutes | 2 |
| 7 minutes | 2 |
| 8 minutes | 2 |
| Sinner won the tie-break | 2 |
| 2 hours | 1 |
| 25 minutes | 1 |
| 3 minutes | 1 |
| 34 minutes | 1 |
| 4 minutes | 1 |
| 9 minutes | 1 |
| A crowd interruption caused a replay | 1 |
| A crowd interruption delayed play | 1 |
| A five-minute evaluation | 1 |
| Alcaraz advanced directly to the final | 1 |
| Alcaraz broke serve | 1 |
| Alcaraz entered the stadium | 1 |

## Duplicate Leakage Check

- Exact duplicate leakage detected: False
- Leaked duplicate groups: 0

## Warnings

- Category other_temporal has only 17 examples; per-category conclusions will be weak.
- Category tennis_injury_or_medical has only 3 examples; per-category conclusions will be weak.
- Category tournament_round_sequence has only 19 examples; per-category conclusions will be weak.

## Recommendation

More data is recommended for low-count categories before final domain-adaptation conclusions.
