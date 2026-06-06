# Tennis Raw Dataset Audit

## Inputs and Outputs

- Input: `data/tennis/raw/tennis_raw.json`
- Audited dataset: `data/tennis/processed/tennis_raw_audited.json`
- Summary JSON: `results/tennis_domain_adaptation/raw_audit/tennis_raw_audit_summary.json`

## Record Counts

- Total examples: 1122
- Valid examples: 1122
- Malformed examples: 0
- Assigned question ids: 1122
- Exact duplicate records after first occurrence: 1
- Near-duplicate records after first match: 8
- Near-duplicate pairs: 14

## Category Distribution

| Category | Count |
| --- | ---: |
| duration_minutes | 147 |
| immediate_before_after | 196 |
| other_temporal | 17 |
| overlap_while_during | 115 |
| tennis_injury_or_medical | 3 |
| tournament_round_sequence | 19 |
| which_first_last | 235 |
| yes_no_before_after | 390 |

## Answer Distribution

| Answer | Count |
| --- | ---: |
| Yes | 309 |
| No | 200 |
| 45 minutes | 19 |
| 12 minutes | 13 |
| 7 minutes | 11 |
| 15 minutes | 10 |
| 9 minutes | 10 |
| The semifinal | 8 |
| 25 minutes | 6 |
| 14 minutes | 5 |
| 20 minutes | 5 |
| 3 minutes | 5 |
| 30 minutes | 5 |
| 5 minutes | 5 |
| Nadal broke serve | 5 |
| Sinner broke serve | 5 |
| 21 minutes | 4 |
| 40 minutes | 4 |
| 8 minutes | 4 |
| Alcaraz broke serve | 4 |
| Alcaraz saved a break point | 4 |
| Djokovic broke serve | 4 |
| Djokovic saved a break point | 4 |
| Medvedev broke serve | 4 |
| Rublev broke serve | 4 |

## Length Statistics

Length statistics are reported as characters and whitespace-token counts.

```json
{
  "answer": {
    "characters": {
      "count": 1122,
      "max": 75,
      "mean": 13.32,
      "median": 9.0,
      "min": 1,
      "p25": 3,
      "p75": 24
    },
    "tokens": {
      "count": 1122,
      "max": 12,
      "mean": 2.59,
      "median": 2.0,
      "min": 1,
      "p25": 1,
      "p75": 4
    }
  },
  "context": {
    "characters": {
      "count": 1122,
      "max": 389,
      "mean": 180.54,
      "median": 173.0,
      "min": 96,
      "p25": 157,
      "p75": 199
    },
    "tokens": {
      "count": 1122,
      "max": 61,
      "mean": 31.14,
      "median": 30.0,
      "min": 17,
      "p25": 27,
      "p75": 34
    }
  },
  "question": {
    "characters": {
      "count": 1122,
      "max": 121,
      "mean": 59.7,
      "median": 60.0,
      "min": 20,
      "p25": 50,
      "p75": 70
    },
    "tokens": {
      "count": 1122,
      "max": 20,
      "mean": 9.85,
      "median": 10.0,
      "min": 3,
      "p25": 8,
      "p75": 12
    }
  }
}
```

## Malformed Records

No malformed records were detected.

## Main Limitations

- Raw records only provide the minimal context/question/answer schema; provenance/source fields are not present.
- Categories and tags are rule-based labels, so ambiguous questions may need manual review before training.
- Near-duplicate detection is based on text similarity only and does not know template ids, match ids, or source clusters.
- The audit does not verify factual tennis correctness or whether each answer is entailed by the context.
