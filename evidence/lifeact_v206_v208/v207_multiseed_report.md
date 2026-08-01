# LifeAct-MDCK v207 multiseed confirmation

Frozen state table: `<AIRI_SOURCE_ROOT>/outputs/lifeact_mdck_mechanochemical_state_gate_v207_center60_seed207001_2026-08-01/v207_cell_state.parquet`.
Seeds: `7,42,123`.

## Decision

| protocol               | model   |   seeds |   real_rmse_mean |   gain_vs_coord_percent_mean |   gain_vs_coord_percent_std |   gain_vs_best_control_percent_mean |   soft_pass_seeds |   hard_pass_seeds |
|:-----------------------|:--------|--------:|-----------------:|-----------------------------:|----------------------------:|------------------------------------:|------------------:|------------------:|
| chronological          | hgbdt   |       3 |         9.977104 |                     0.850073 |                    0.018047 |                            0.557521 |                 0 |                 0 |
| chronological          | ridge   |       3 |        10.052231 |                    -0.004930 |                    0.000000 |                           -0.030267 |                 0 |                 0 |
| leave_one_sequence_out | hgbdt   |       3 |        11.082042 |                    -0.125009 |                    0.080633 |                           -0.125009 |                 0 |                 0 |
| leave_one_sequence_out | ridge   |       3 |        11.423967 |                    -1.580367 |                    0.000000 |                           -1.580367 |                 0 |                 0 |

## Full aggregate

| protocol               | packet             | model   |   seeds |   component_rmse_mean |   component_rmse_seed_std |   component_r2_mean |   gain_vs_coord_percent_mean |   gain_vs_coord_percent_std |
|:-----------------------|:-------------------|:--------|--------:|----------------------:|--------------------------:|--------------------:|-----------------------------:|----------------------------:|
| chronological          | coord_actin        | hgbdt   |       3 |             10.046365 |                  0.001559 |            0.000408 |                     0.161777 |                    0.020443 |
| chronological          | coord_actin        | ridge   |       3 |             10.050577 |                  0.000000 |           -0.000431 |                     0.011525 |                    0.000000 |
| chronological          | coord_contact      | hgbdt   |       3 |             10.042378 |                  0.005830 |            0.001201 |                     0.201398 |                    0.053335 |
| chronological          | coord_contact      | ridge   |       3 |             10.038175 |                  0.000000 |            0.002037 |                     0.134914 |                    0.000000 |
| chronological          | coord_only         | hgbdt   |       3 |             10.062644 |                  0.001131 |           -0.002834 |                     0.000000 |                    0.000000 |
| chronological          | coord_only         | ridge   |       3 |             10.051736 |                  0.000000 |           -0.000661 |                     0.000000 |                    0.000000 |
| chronological          | coord_reliability  | hgbdt   |       3 |             10.009625 |                  0.003419 |            0.007705 |                     0.526888 |                    0.027966 |
| chronological          | coord_reliability  | ridge   |       3 |             10.046517 |                  0.000000 |            0.000378 |                     0.051923 |                    0.000000 |
| chronological          | coord_shape        | hgbdt   |       3 |             10.010339 |                  0.007227 |            0.007564 |                     0.519801 |                    0.063714 |
| chronological          | coord_shape        | ridge   |       3 |             10.059848 |                  0.000000 |           -0.002277 |                    -0.080707 |                    0.000000 |
| chronological          | full_real          | hgbdt   |       3 |              9.977104 |                  0.001721 |            0.014143 |                     0.850073 |                    0.018047 |
| chronological          | full_real          | ridge   |       3 |             10.052231 |                  0.000000 |           -0.000760 |                    -0.004930 |                    0.000000 |
| chronological          | full_row_shuffled  | hgbdt   |       3 |             10.077967 |                  0.002733 |           -0.005891 |                    -0.152268 |                    0.018475 |
| chronological          | full_row_shuffled  | ridge   |       3 |             10.058923 |                  0.003990 |           -0.002093 |                    -0.071502 |                    0.039697 |
| chronological          | full_time_shuffled | hgbdt   |       3 |             10.040923 |                  0.004971 |            0.001490 |                     0.215864 |                    0.042171 |
| chronological          | full_time_shuffled | ridge   |       3 |             10.049190 |                  0.000000 |           -0.000154 |                     0.025329 |                    0.000000 |
| chronological          | full_wrong_cell    | hgbdt   |       3 |             10.033451 |                  0.008863 |            0.002975 |                     0.290112 |                    0.094192 |
| chronological          | full_wrong_cell    | ridge   |       3 |             10.059080 |                  0.004613 |           -0.002124 |                    -0.073064 |                    0.045888 |
| chronological          | full_zero_state    | hgbdt   |       3 |             10.062644 |                  0.001131 |           -0.002834 |                     0.000000 |                    0.000000 |
| chronological          | full_zero_state    | ridge   |       3 |             10.051736 |                  0.000000 |           -0.000661 |                     0.000000 |                    0.000000 |
| leave_one_sequence_out | coord_actin        | hgbdt   |       3 |             11.090865 |                  0.007808 |           -0.088268 |                    -0.204738 |                    0.029538 |
| leave_one_sequence_out | coord_actin        | ridge   |       3 |             11.286435 |                  0.000000 |           -0.213452 |                    -0.357452 |                    0.000000 |
| leave_one_sequence_out | coord_contact      | hgbdt   |       3 |             11.094055 |                  0.009838 |           -0.091369 |                    -0.233560 |                    0.053203 |
| leave_one_sequence_out | coord_contact      | ridge   |       3 |             11.246991 |                  0.000000 |           -0.184510 |                    -0.006726 |                    0.000000 |
| leave_one_sequence_out | coord_only         | hgbdt   |       3 |             11.068203 |                  0.005161 |           -0.084528 |                     0.000000 |                    0.000000 |
| leave_one_sequence_out | coord_only         | ridge   |       3 |             11.246235 |                  0.000000 |           -0.184073 |                     0.000000 |                    0.000000 |
| leave_one_sequence_out | coord_reliability  | hgbdt   |       3 |             11.083057 |                  0.015776 |           -0.091556 |                    -0.134180 |                    0.103985 |
| leave_one_sequence_out | coord_reliability  | ridge   |       3 |             11.300004 |                  0.000000 |           -0.201034 |                    -0.478109 |                    0.000000 |
| leave_one_sequence_out | coord_shape        | hgbdt   |       3 |             11.086503 |                  0.027227 |           -0.087342 |                    -0.165273 |                    0.199304 |
| leave_one_sequence_out | coord_shape        | ridge   |       3 |             11.295364 |                  0.000000 |           -0.194863 |                    -0.436855 |                    0.000000 |
| leave_one_sequence_out | full_real          | hgbdt   |       3 |             11.082042 |                  0.013060 |           -0.086736 |                    -0.125009 |                    0.080633 |
| leave_one_sequence_out | full_real          | ridge   |       3 |             11.423967 |                  0.000000 |           -0.268870 |                    -1.580367 |                    0.000000 |
| leave_one_sequence_out | full_row_shuffled  | hgbdt   |       3 |             11.103427 |                  0.013770 |           -0.094931 |                    -0.318222 |                    0.080112 |
| leave_one_sequence_out | full_row_shuffled  | ridge   |       3 |             11.277802 |                  0.026243 |           -0.197868 |                    -0.280692 |                    0.233347 |
| leave_one_sequence_out | full_time_shuffled | hgbdt   |       3 |             11.126062 |                  0.014367 |           -0.100491 |                    -0.522720 |                    0.083780 |
| leave_one_sequence_out | full_time_shuffled | ridge   |       3 |             11.344541 |                  0.000000 |           -0.221702 |                    -0.874130 |                    0.000000 |
| leave_one_sequence_out | full_wrong_cell    | hgbdt   |       3 |             11.106002 |                  0.006949 |           -0.096052 |                    -0.341518 |                    0.076723 |
| leave_one_sequence_out | full_wrong_cell    | ridge   |       3 |             11.304670 |                  0.021204 |           -0.195302 |                    -0.519599 |                    0.188540 |
| leave_one_sequence_out | full_zero_state    | hgbdt   |       3 |             11.068203 |                  0.005161 |           -0.084528 |                     0.000000 |                    0.000000 |
| leave_one_sequence_out | full_zero_state    | ridge   |       3 |             11.246235 |                  0.000000 |           -0.184073 |                     0.000000 |                    0.000000 |

A robust pass requires the real state to beat coordinates and every hard control across seeds; seed variation alone is not treated as independent biological replication.
