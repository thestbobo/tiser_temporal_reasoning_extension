# Tennis Adapter Comparison

- Generated at: `2026-06-07T16:58:45`
- Delta baseline: `base_qwen_standard_fulltest`

## Overall

| Condition | Model | Prompt | No Adapter | Adapter | N | EM | F1 | Malformed | Malformed Rate | Delta EM | Delta F1 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_qwen_standard_fulltest` | `Qwen/Qwen2.5-0.5B-Instruct` | `standard` | true | `` | 20 | 0.4000 | 0.4494 | 0 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_10` | `Qwen/Qwen2.5-0.5B-Instruct` | `standard` | true | `` | 10 | 0.6000 | 0.6821 | 0 | 0.0000 | +0.2000 | +0.2327 |
| `base_qwen_standard_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | `standard` | true | `` | 100 | 0.3800 | 0.4504 | 0 | 0.0000 | -0.0200 | +0.0010 |
| `base_qwen_tiser_fulltest` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | true | `` | 20 | 0.5000 | 0.5785 | 0 | 0.0000 | +0.1000 | +0.1292 |
| `base_qwen_tiser_smoke_10` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | true | `` | 10 | 0.6000 | 0.7417 | 0 | 0.0000 | +0.2000 | +0.2923 |
| `base_qwen_tiser_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | true | `` | 100 | 0.3800 | 0.4014 | 4 | 0.0400 | -0.0200 | -0.0480 |
| `tennis_only_full600_fulltest` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | false | `C:\Users\aless\OneDrive\Desktop\dnlp\tiser_temporal_reasoning_extension\tiser_temporal_reasoning_extension\model\tiser_tennis_full600_smoke\adapter` | 20 | 0.4500 | 0.4944 | 0 | 0.0000 | +0.0500 | +0.0451 |
| `tennis_only_full600_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | false | `C:\Users\aless\OneDrive\Desktop\dnlp\tiser_temporal_reasoning_extension\tiser_temporal_reasoning_extension\model\tiser_tennis_full600_smoke\adapter` | 100 | 0.4700 | 0.5158 | 0 | 0.0000 | +0.0700 | +0.0664 |
| `tennis_only_trace50_smoke_100` | `Qwen/Qwen2.5-0.5B-Instruct` | `tiser` | false | `C:\Users\aless\OneDrive\Desktop\dnlp\tiser_temporal_reasoning_extension\tiser_temporal_reasoning_extension\model\tiser_tennis_smoke\adapter` | 100 | 0.3300 | 0.3767 | 1 | 0.0100 | -0.0700 | -0.0727 |

## Per Category

| Condition | Category | N | EM | F1 | Malformed Rate | Delta EM | Delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_qwen_standard_fulltest` | `duration_minutes` | 3 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_fulltest` | `immediate_before_after` | 3 | 0.3333 | 0.6111 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_fulltest` | `overlap_while_during` | 4 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_fulltest` | `which_first_last` | 4 | 0.0000 | 0.0385 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_fulltest` | `yes_no_before_after` | 6 | 0.8333 | 0.8333 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_10` | `duration_minutes` | 1 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_standard_smoke_10` | `immediate_before_after` | 2 | 0.5000 | 0.8333 | 0.0000 | +0.1667 | +0.2222 |
| `base_qwen_standard_smoke_10` | `overlap_while_during` | 1 | 1.0000 | 1.0000 | 0.0000 | +0.5000 | +0.5000 |
| `base_qwen_standard_smoke_10` | `which_first_last` | 2 | 0.0000 | 0.0769 | 0.0000 | +0.0000 | +0.0385 |
| `base_qwen_standard_smoke_10` | `yes_no_before_after` | 4 | 1.0000 | 1.0000 | 0.0000 | +0.1667 | +0.1667 |
| `base_qwen_standard_smoke_100` | `duration_minutes` | 15 | 0.2000 | 0.2000 | 0.0000 | +0.2000 | +0.2000 |
| `base_qwen_standard_smoke_100` | `immediate_before_after` | 15 | 0.1333 | 0.4784 | 0.0000 | -0.2000 | -0.1327 |
| `base_qwen_standard_smoke_100` | `other_temporal` | 1 | 0.0000 | 0.0000 | 0.0000 |  |  |
| `base_qwen_standard_smoke_100` | `overlap_while_during` | 7 | 0.5714 | 0.5714 | 0.0000 | +0.0714 | +0.0714 |
| `base_qwen_standard_smoke_100` | `tournament_round_sequence` | 3 | 0.6667 | 0.7222 | 0.0000 |  |  |
| `base_qwen_standard_smoke_100` | `which_first_last` | 15 | 0.0667 | 0.1798 | 0.0000 | +0.0667 | +0.1414 |
| `base_qwen_standard_smoke_100` | `yes_no_before_after` | 44 | 0.5909 | 0.5909 | 0.0000 | -0.2424 | -0.2424 |
| `base_qwen_tiser_fulltest` | `duration_minutes` | 3 | 0.6667 | 0.6667 | 0.0000 | +0.6667 | +0.6667 |
| `base_qwen_tiser_fulltest` | `immediate_before_after` | 3 | 0.0000 | 0.5235 | 0.0000 | -0.3333 | -0.0876 |
| `base_qwen_tiser_fulltest` | `overlap_while_during` | 4 | 0.7500 | 0.7500 | 0.0000 | +0.2500 | +0.2500 |
| `base_qwen_tiser_fulltest` | `which_first_last` | 4 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | -0.0385 |
| `base_qwen_tiser_fulltest` | `yes_no_before_after` | 6 | 0.8333 | 0.8333 | 0.0000 | +0.0000 | +0.0000 |
| `base_qwen_tiser_smoke_10` | `duration_minutes` | 1 | 1.0000 | 1.0000 | 0.0000 | +1.0000 | +1.0000 |
| `base_qwen_tiser_smoke_10` | `immediate_before_after` | 2 | 0.0000 | 0.7083 | 0.0000 | -0.3333 | +0.0972 |
| `base_qwen_tiser_smoke_10` | `overlap_while_during` | 1 | 1.0000 | 1.0000 | 0.0000 | +0.5000 | +0.5000 |
| `base_qwen_tiser_smoke_10` | `which_first_last` | 2 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | -0.0385 |
| `base_qwen_tiser_smoke_10` | `yes_no_before_after` | 4 | 1.0000 | 1.0000 | 0.0000 | +0.1667 | +0.1667 |
| `base_qwen_tiser_smoke_100` | `duration_minutes` | 15 | 0.3333 | 0.3333 | 0.0000 | +0.3333 | +0.3333 |
| `base_qwen_tiser_smoke_100` | `immediate_before_after` | 15 | 0.0667 | 0.1792 | 0.0000 | -0.2667 | -0.4319 |
| `base_qwen_tiser_smoke_100` | `other_temporal` | 1 | 0.0000 | 0.0000 | 0.0000 |  |  |
| `base_qwen_tiser_smoke_100` | `overlap_while_during` | 7 | 0.7143 | 0.7143 | 0.0000 | +0.2143 | +0.2143 |
| `base_qwen_tiser_smoke_100` | `tournament_round_sequence` | 3 | 0.6667 | 0.7222 | 0.0000 |  |  |
| `base_qwen_tiser_smoke_100` | `which_first_last` | 15 | 0.0000 | 0.0190 | 0.0000 | +0.0000 | -0.0194 |
| `base_qwen_tiser_smoke_100` | `yes_no_before_after` | 44 | 0.5682 | 0.5682 | 0.0909 | -0.2652 | -0.2652 |
| `tennis_only_full600_fulltest` | `duration_minutes` | 3 | 0.3333 | 0.3333 | 0.0000 | +0.3333 | +0.3333 |
| `tennis_only_full600_fulltest` | `immediate_before_after` | 3 | 0.3333 | 0.6296 | 0.0000 | +0.0000 | +0.0185 |
| `tennis_only_full600_fulltest` | `overlap_while_during` | 4 | 0.7500 | 0.7500 | 0.0000 | +0.2500 | +0.2500 |
| `tennis_only_full600_fulltest` | `which_first_last` | 4 | 0.5000 | 0.5000 | 0.0000 | +0.5000 | +0.4615 |
| `tennis_only_full600_fulltest` | `yes_no_before_after` | 6 | 0.3333 | 0.3333 | 0.0000 | -0.5000 | -0.5000 |
| `tennis_only_full600_smoke_100` | `duration_minutes` | 15 | 0.2667 | 0.2667 | 0.0000 | +0.2667 | +0.2667 |
| `tennis_only_full600_smoke_100` | `immediate_before_after` | 15 | 0.3333 | 0.5514 | 0.0000 | +0.0000 | -0.0597 |
| `tennis_only_full600_smoke_100` | `other_temporal` | 1 | 1.0000 | 1.0000 | 0.0000 |  |  |
| `tennis_only_full600_smoke_100` | `overlap_while_during` | 7 | 0.8571 | 0.8571 | 0.0000 | +0.3571 | +0.3571 |
| `tennis_only_full600_smoke_100` | `tournament_round_sequence` | 3 | 0.6667 | 0.7619 | 0.0000 |  |  |
| `tennis_only_full600_smoke_100` | `which_first_last` | 15 | 0.4000 | 0.4681 | 0.0000 | +0.4000 | +0.4297 |
| `tennis_only_full600_smoke_100` | `yes_no_before_after` | 44 | 0.5227 | 0.5227 | 0.0000 | -0.3106 | -0.3106 |
| `tennis_only_trace50_smoke_100` | `duration_minutes` | 15 | 0.3333 | 0.3778 | 0.0000 | +0.3333 | +0.3778 |
| `tennis_only_trace50_smoke_100` | `immediate_before_after` | 15 | 0.0667 | 0.2402 | 0.0000 | -0.2667 | -0.3709 |
| `tennis_only_trace50_smoke_100` | `other_temporal` | 1 | 0.0000 | 0.0000 | 0.0000 |  |  |
| `tennis_only_trace50_smoke_100` | `overlap_while_during` | 7 | 0.7143 | 0.7143 | 0.0000 | +0.2143 | +0.2143 |
| `tennis_only_trace50_smoke_100` | `tournament_round_sequence` | 3 | 0.3333 | 0.5952 | 0.0000 |  |  |
| `tennis_only_trace50_smoke_100` | `which_first_last` | 15 | 0.0667 | 0.1076 | 0.0000 | +0.0667 | +0.0691 |
| `tennis_only_trace50_smoke_100` | `yes_no_before_after` | 44 | 0.4545 | 0.4545 | 0.0227 | -0.3788 | -0.3788 |
