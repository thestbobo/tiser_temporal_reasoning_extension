# Tennis Adapter Comparison

- Generated at: `2026-06-07T16:05:46`
- Delta baseline: `base_qwen_standard_smoke_100`

## Overall

| Condition | Model | Prompt | No Adapter | Adapter | N | EM | F1 | Malformed | Malformed Rate | Delta EM | Delta F1 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_qwen_standard_smoke_10` | `Qwen/Qwen2.5-0.5B-Instruct` | `standard` | true | `` | 10 | 0.6000 | 0.6821 | 0 | 0.0000 | +0.2200 | +0.2316 |
| `base_qwen_standard_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | `standard` | true | `` | 100 | 0.3800 | 0.4504 | 0 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_tiser_smoke_10` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | true | `` | 10 | 0.6000 | 0.7417 | 0 | 0.0000 | +0.2200 | +0.2913 |
| `base_qwen_tiser_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | true | `` | 100 | 0.3800 | 0.4014 | 4 | 0.0400 | +0.0000 | -0.0490 |
| `tennis_only_trace50_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | false | `C:\Users\aless\OneDrive\Desktop\dnlp\tiser_temporal_reasoning_extension\tiser_temporal_reasoning_extension\model\tiser_tennis_smoke\adapter` | 100 | 0.3300 | 0.3767 | 1 | 0.0100 | -0.0500 | -0.0737 |

## Per Category

| Condition | Category | N | EM | F1 | Malformed Rate | Delta EM | Delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_qwen_standard_smoke_10` | `duration_minutes` | 1 | 0.0000 | 0.0000 | 0.0000 | -0.2000 | -0.2000 |
| `base_qwen_standard_smoke_10` | `immediate_before_after` | 2 | 0.5000 | 0.8333 | 0.0000 | +0.3667 | +0.3549 |
| `base_qwen_standard_smoke_10` | `overlap_while_during` | 1 | 1.0000 | 1.0000 | 0.0000 | +0.4286 | +0.4286 |
| `base_qwen_standard_smoke_10` | `which_first_last` | 2 | 0.0000 | 0.0769 | 0.0000 | -0.0667 | -0.1029 |
| `base_qwen_standard_smoke_10` | `yes_no_before_after` | 4 | 1.0000 | 1.0000 | 0.0000 | +0.4091 | +0.4091 |
| `base_qwen_standard_smoke_100` | `duration_minutes` | 15 | 0.2000 | 0.2000 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_100` | `immediate_before_after` | 15 | 0.1333 | 0.4784 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_100` | `other_temporal` | 1 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_100` | `overlap_while_during` | 7 | 0.5714 | 0.5714 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_100` | `tournament_round_sequence` | 3 | 0.6667 | 0.7222 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_100` | `which_first_last` | 15 | 0.0667 | 0.1798 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_100` | `yes_no_before_after` | 44 | 0.5909 | 0.5909 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_tiser_smoke_10` | `duration_minutes` | 1 | 1.0000 | 1.0000 | 0.0000 | +0.8000 | +0.8000 |
| `base_qwen_tiser_smoke_10` | `immediate_before_after` | 2 | 0.0000 | 0.7083 | 0.0000 | -0.1333 | +0.2299 |
| `base_qwen_tiser_smoke_10` | `overlap_while_during` | 1 | 1.0000 | 1.0000 | 0.0000 | +0.4286 | +0.4286 |
| `base_qwen_tiser_smoke_10` | `which_first_last` | 2 | 0.0000 | 0.0000 | 0.0000 | -0.0667 | -0.1798 |
| `base_qwen_tiser_smoke_10` | `yes_no_before_after` | 4 | 1.0000 | 1.0000 | 0.0000 | +0.4091 | +0.4091 |
| `base_qwen_tiser_smoke_100` | `duration_minutes` | 15 | 0.3333 | 0.3333 | 0.0000 | +0.1333 | +0.1333 |
| `base_qwen_tiser_smoke_100` | `immediate_before_after` | 15 | 0.0667 | 0.1792 | 0.0000 | -0.0667 | -0.2992 |
| `base_qwen_tiser_smoke_100` | `other_temporal` | 1 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_tiser_smoke_100` | `overlap_while_during` | 7 | 0.7143 | 0.7143 | 0.0000 | +0.1429 | +0.1429 |
| `base_qwen_tiser_smoke_100` | `tournament_round_sequence` | 3 | 0.6667 | 0.7222 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_tiser_smoke_100` | `which_first_last` | 15 | 0.0000 | 0.0190 | 0.0000 | -0.0667 | -0.1608 |
| `base_qwen_tiser_smoke_100` | `yes_no_before_after` | 44 | 0.5682 | 0.5682 | 0.0909 | -0.0227 | -0.0227 |
| `tennis_only_trace50_smoke_100` | `duration_minutes` | 15 | 0.3333 | 0.3778 | 0.0000 | +0.1333 | +0.1778 |
| `tennis_only_trace50_smoke_100` | `immediate_before_after` | 15 | 0.0667 | 0.2402 | 0.0000 | -0.0667 | -0.2382 |
| `tennis_only_trace50_smoke_100` | `other_temporal` | 1 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| `tennis_only_trace50_smoke_100` | `overlap_while_during` | 7 | 0.7143 | 0.7143 | 0.0000 | +0.1429 | +0.1429 |
| `tennis_only_trace50_smoke_100` | `tournament_round_sequence` | 3 | 0.3333 | 0.5952 | 0.0000 | -0.3333 | -0.1270 |
| `tennis_only_trace50_smoke_100` | `which_first_last` | 15 | 0.0667 | 0.1076 | 0.0000 | +0.0000 | -0.0723 |
| `tennis_only_trace50_smoke_100` | `yes_no_before_after` | 44 | 0.4545 | 0.4545 | 0.0227 | -0.1364 | -0.1364 |
