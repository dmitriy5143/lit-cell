# LaChance h1 evidence bundle v205

## Decision

The h1 target is difficult but not shown to be irreducible noise. The frozen model exposes a continuous h1--h6 utility trade-off; h1 remains an explicit guarded endpoint rather than being dismissed.

Only `lambda_00` is the original confirmatory h1 operating point. Intermediate profiles are descriptive. `lambda_10` was frozen only before the separate movies 10--16 evaluation.

## Pareto endpoints

| objective_name   |   lambda |   h1_guard_percent | status                        |   h1_component_rmse |   h1_component_r2 |   h1_normalized_rmse |   h2_component_rmse |   h2_component_r2 |   h2_normalized_rmse |   h4_component_rmse |   h4_component_r2 |   h4_normalized_rmse |   h6_component_rmse |   h6_component_r2 |   h6_normalized_rmse | pareto_nondominated   |
|:-----------------|---------:|-------------------:|:------------------------------|--------------------:|------------------:|---------------------:|--------------------:|------------------:|---------------------:|--------------------:|------------------:|---------------------:|--------------------:|------------------:|---------------------:|:----------------------|
| lambda_00        | 0.000000 |           0.500000 | confirmatory_h1               |            3.474374 |          0.506282 |             0.702152 |            4.386432 |          0.756597 |             0.492764 |            5.733410 |          0.871134 |             0.358075 |            6.784638 |          0.903985 |             0.308710 | True                  |
| lambda_10        | 1.000000 |          10.000000 | frozen_later_on_unseen_movies |            3.807965 |          0.406798 |             0.769779 |            4.180735 |          0.779077 |             0.469705 |            4.924978 |          0.905181 |             0.307348 |            5.500749 |          0.937096 |             0.250057 | True                  |

## Dimensionless error

| objective_name   |   horizon |   movies |   component_rmse_mean |   target_component_sd_mean |   normalized_rmse_mean |   normalized_rmse_std |   variance_explained_mean |   skill_vs_cv_mean |
|:-----------------|----------:|---------:|----------------------:|---------------------------:|-----------------------:|----------------------:|--------------------------:|-------------------:|
| lambda_00        |         1 |        6 |              3.474374 |                   4.943698 |               0.702152 |              0.021529 |                  0.506596 |           0.340574 |
| lambda_00        |         2 |        6 |              4.386432 |                   8.884738 |               0.492764 |              0.025408 |                  0.756646 |           0.182592 |
| lambda_00        |         4 |        6 |              5.733410 |                  15.988892 |               0.358075 |              0.027001 |                  0.871175 |           0.069261 |
| lambda_00        |         6 |        6 |              6.784638 |                  21.963488 |               0.308710 |              0.027196 |                  0.904082 |          -0.043223 |
| lambda_10        |         1 |        6 |              3.807965 |                   4.943698 |               0.769779 |              0.031253 |                  0.406627 |           0.207705 |
| lambda_10        |         2 |        6 |              4.180735 |                   8.884738 |               0.469705 |              0.019323 |                  0.779066 |           0.257110 |
| lambda_10        |         4 |        6 |              4.924978 |                  15.988892 |               0.307348 |              0.022511 |                  0.905115 |           0.314037 |
| lambda_10        |         6 |        6 |              5.500749 |                  21.963488 |               0.250057 |              0.021556 |                  0.937084 |           0.314558 |

## Paired controls

| objective_name   |   horizon | comparator   |   movies |   mean_rmse_delta_comparator_minus_real |   movies_real_better |   exact_two_sided_sign_flip_p |   bootstrap_ci_low |   bootstrap_ci_high |
|:-----------------|----------:|:-------------|---------:|----------------------------------------:|---------------------:|------------------------------:|-------------------:|--------------------:|
| lambda_00        |         1 | no_update    |        6 |                                0.008640 |                    4 |                      0.843750 |          -0.039551 |            0.057518 |
| lambda_00        |         1 | wrong_cell   |        6 |                                0.050306 |                    6 |                      0.031250 |           0.035296 |            0.067037 |
| lambda_00        |         1 | stale_time   |        6 |                                0.091439 |                    6 |                      0.031250 |           0.055378 |            0.142862 |
| lambda_00        |         2 | no_update    |        6 |                                0.305716 |                    6 |                      0.031250 |           0.261373 |            0.349122 |
| lambda_00        |         2 | wrong_cell   |        6 |                                0.434084 |                    6 |                      0.031250 |           0.337116 |            0.532155 |
| lambda_00        |         2 | stale_time   |        6 |                                0.380920 |                    6 |                      0.031250 |           0.296590 |            0.468366 |
| lambda_00        |         4 | no_update    |        6 |                                0.746471 |                    6 |                      0.031250 |           0.613290 |            0.874438 |
| lambda_00        |         4 | wrong_cell   |        6 |                                0.952496 |                    6 |                      0.031250 |           0.742191 |            1.169574 |
| lambda_00        |         4 | stale_time   |        6 |                                0.357816 |                    6 |                      0.031250 |           0.294785 |            0.421993 |
| lambda_00        |         6 | no_update    |        6 |                                1.027402 |                    6 |                      0.031250 |           0.819446 |            1.216200 |
| lambda_00        |         6 | wrong_cell   |        6 |                                1.318483 |                    6 |                      0.031250 |           1.028244 |            1.617004 |
| lambda_00        |         6 | stale_time   |        6 |                                0.323575 |                    6 |                      0.031250 |           0.266121 |            0.375988 |
| lambda_10        |         1 | no_update    |        6 |                               -0.324951 |                    0 |                      0.031250 |          -0.387164 |           -0.264074 |
| lambda_10        |         1 | wrong_cell   |        6 |                                0.049296 |                    6 |                      0.031250 |           0.025515 |            0.073076 |
| lambda_10        |         1 | stale_time   |        6 |                               -0.037327 |                    2 |                      0.375000 |          -0.109322 |            0.034670 |
| lambda_10        |         2 | no_update    |        6 |                                0.511413 |                    6 |                      0.031250 |           0.474902 |            0.553232 |
| lambda_10        |         2 | wrong_cell   |        6 |                                0.926249 |                    6 |                      0.031250 |           0.869992 |            0.985646 |
| lambda_10        |         2 | stale_time   |        6 |                                0.984167 |                    6 |                      0.031250 |           0.889029 |            1.064832 |
| lambda_10        |         4 | no_update    |        6 |                                1.554903 |                    6 |                      0.031250 |           1.493729 |            1.611078 |
| lambda_10        |         4 | wrong_cell   |        6 |                                2.120065 |                    6 |                      0.031250 |           1.974659 |            2.276167 |
| lambda_10        |         4 | stale_time   |        6 |                                0.884902 |                    6 |                      0.031250 |           0.802861 |            0.965939 |
| lambda_10        |         6 | no_update    |        6 |                                2.311291 |                    6 |                      0.031250 |           2.204304 |            2.412328 |
| lambda_10        |         6 | wrong_cell   |        6 |                                2.975063 |                    6 |                      0.031250 |           2.750964 |            3.209884 |
| lambda_10        |         6 | stale_time   |        6 |                                0.829056 |                    6 |                      0.031250 |           0.749610 |            0.910704 |

## Localization evidence

| dataset            | audit                                               | quantity                        |     value | unit   | independent_localization_reference   | decision     |
|:-------------------|:----------------------------------------------------|:--------------------------------|----------:|:-------|:-------------------------------------|:-------------|
| LaChance MDCK Bulk | Cellpose current-query reliability, six movie folds | NLL gain over best hard control | -1.146874 | %      | False                                | FAIL         |
| C2C12 F0009        | manual/automatic same-frame forensic match          | median one-step disagreement    |  0.738692 | px     | True                                 | context_only |
| C2C12 F0009        | manual/automatic same-frame forensic match          | p90 one-step disagreement       |  1.880452 | px     | True                                 | context_only |

No LaChance-specific irreducible h1 noise floor was established. C2C12 disagreement is an external scale reference only.

The LaChance Cellpose reliability packet failed its movie-level hard controls and independent causal retracking was not completed. The C2C12 manual/automatic disagreement therefore supplies context, not a LaChance measurement floor and not a subtraction from h1 RMSE.

Elapsed: `41.97` minutes.
