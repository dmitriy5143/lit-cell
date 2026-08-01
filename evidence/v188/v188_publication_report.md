# v188 Publication Evidence Bundle

## Frozen matched causal-online benchmark

| method            |   horizon |   movies |   component_rmse |   vector_rmse |       r2 |
|:------------------|----------:|---------:|-----------------:|--------------:|---------:|
| constant_velocity |         1 |        6 |         4.291708 |      6.069392 | 0.250136 |
| constant_velocity |         6 |        6 |         6.656919 |      9.414305 | 0.909186 |
| gru_track         |         1 |        6 |         3.628207 |      5.131060 | 0.464135 |
| gru_track         |         6 |        6 |         8.412099 |     11.896505 | 0.855164 |
| hgbdt_v52         |         1 |        6 |         3.492363 |      4.938947 | 0.503431 |
| hgbdt_v52         |         6 |        6 |         8.404088 |     11.885176 | 0.855278 |
| kalmannet         |         1 |        6 |         3.926258 |      5.552568 | 0.371938 |
| kalmannet         |         6 |        6 |         8.794405 |     12.437168 | 0.840182 |
| v166_h1_strict    |         1 |        6 |         3.474374 |      4.913506 | 0.507733 |
| v166_h1_strict    |         6 |        6 |         6.784638 |      9.594927 | 0.905043 |
| v166_h6_utility   |         1 |        6 |         3.807965 |      5.385276 | 0.407988 |
| v166_h6_utility   |         6 |        6 |         5.500749 |      7.779233 | 0.937721 |
| v97_no_update     |         1 |        6 |         3.483014 |      4.925725 | 0.505919 |
| v97_no_update     |         6 |        6 |         7.812040 |     11.047893 | 0.874604 |

## Confirmatory movie-level hypotheses

| hypothesis_id   | method          | comparator    |   horizon |   mean_rmse_delta_comparator_minus_method |   relative_gain_percent |   method_better_movies |   exact_two_sided_sign_flip_p |   holm_adjusted_p |
|:----------------|:----------------|:--------------|----------:|------------------------------------------:|------------------------:|-----------------------:|------------------------------:|------------------:|
| H1              | v166_h1_strict  | v97_no_update |         1 |                                  0.008640 |                0.081008 |                      4 |                      0.843750 |          0.843750 |
| H2              | v166_h6_utility | v97_no_update |         6 |                                  2.311291 |               29.763802 |                      6 |                      0.031250 |          0.062500 |

The six-movie matched transport experiment was already complete in `v157h`; v188 restores it to the primary evidence table instead of using the older baseline-only v166 summary.

## External nested LOMO

| dataset   | objective   | control   |   horizon |   outer_folds |   component_rmse_macro |   component_rmse_ci_low |   component_rmse_ci_high |   r2_macro |   gain_percent_macro |   gain_percent_ci_low |   gain_percent_ci_high |   positive_folds |   sign_test_p_two_sided |
|:----------|:------------|:----------|----------:|--------------:|-----------------------:|------------------------:|-------------------------:|-----------:|---------------------:|----------------------:|-----------------------:|-----------------:|------------------------:|
| HUVEC     | h1_strict   | real      |         1 |            18 |               1.081449 |                1.005971 |                 1.171219 |   0.588353 |             3.227965 |              2.819842 |               3.562588 |               18 |                0.000008 |
| HUVEC     | h1_strict   | real      |         6 |            18 |               1.732523 |                1.675822 |                 1.794461 |   0.960762 |            -7.231396 |             -8.170548 |              -6.243290 |                0 |                0.000008 |
| HUVEC     | h6_guard10  | real      |         1 |            18 |               1.217147 |                1.140882 |                 1.311894 |   0.480631 |            -9.184851 |            -10.583548 |              -7.905225 |                0 |                0.000008 |
| HUVEC     | h6_guard10  | real      |         6 |            18 |               1.439869 |                1.378688 |                 1.515349 |   0.972607 |            11.023770 |              9.548074 |              12.684562 |               18 |                0.000008 |
| MDAMB231  | h1_strict   | real      |         1 |            17 |              23.582245 |               22.737219 |                24.377915 |   0.294786 |             1.831745 |              1.703282 |               1.954147 |               17 |                0.000015 |
| MDAMB231  | h1_strict   | real      |         6 |            17 |              34.357702 |               33.201437 |                35.507588 |  -0.174247 |            -2.719453 |             -2.988800 |              -2.450776 |                0 |                0.000015 |
| MDAMB231  | h6_guard10  | real      |         1 |            17 |              24.911038 |               24.045798 |                25.726668 |   0.212776 |            -3.722138 |             -3.960322 |              -3.476284 |                0 |                0.000015 |
| MDAMB231  | h6_guard10  | real      |         6 |            17 |              31.337439 |               30.209435 |                32.401906 |   0.023738 |             6.328989 |              6.029196 |               6.614322 |               17 |                0.000015 |

## Robust uncertainty response

| operating_point   | condition           |   movies |   scale_factor_mean |   scale_factor_ratio_vs_real_mean |   movies_expanded_vs_real |   mean_step_scale |   calibration_error_gain_mean |   coverage_90_gain_mean |   nll_gain_mean | source_path                                                                                                                       | source_sha256                                                    |
|:------------------|:--------------------|---------:|--------------------:|----------------------------------:|--------------------------:|------------------:|------------------------------:|------------------------:|----------------:|:----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| h1_strict         | missing_0.4         |        6 |            1.031148 |                          1.022843 |                         5 |          2.491323 |                      0.005602 |                0.009274 |        0.009090 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h1_2026-07-29/v157f_uncertainty_response.csv | a3c75333e9b37497edfebd52eaf2ef7ce56adaefc13c0ec9465d5a45163be7fc |
| h1_strict         | real_update_every_1 |        6 |            1.008333 |                          1.000000 |                         0 |          2.435554 |                     -0.002632 |                0.002550 |        0.002361 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h1_2026-07-29/v157f_uncertainty_response.csv | a3c75333e9b37497edfebd52eaf2ef7ce56adaefc13c0ec9465d5a45163be7fc |
| h1_strict         | tracking_noise_2px  |        6 |            1.040719 |                          1.027600 |                         6 |          2.503949 |                      0.004144 |                0.011175 |        0.004922 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h1_2026-07-29/v157f_uncertainty_response.csv | a3c75333e9b37497edfebd52eaf2ef7ce56adaefc13c0ec9465d5a45163be7fc |
| h1_strict         | wrong_cell          |        6 |            1.066667 |                          1.058855 |                         5 |          2.579325 |                      0.028939 |                0.022385 |        0.028745 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h1_2026-07-29/v157f_uncertainty_response.csv | a3c75333e9b37497edfebd52eaf2ef7ce56adaefc13c0ec9465d5a45163be7fc |
| h6_utility        | missing_0.4         |        6 |            1.036932 |                          1.130386 |                         6 |          2.501730 |                      0.010421 |                0.011397 |        0.009852 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h6_2026-07-29/v157f_uncertainty_response.csv | 3927debc3089f331cfe1fce73e7070a16148701e028256184378ae075d12b20e |
| h6_utility        | real_update_every_1 |        6 |            0.916667 |                          1.000000 |                         0 |          2.215668 |                      0.016091 |               -0.021689 |       -0.002903 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h6_2026-07-29/v157f_uncertainty_response.csv | 3927debc3089f331cfe1fce73e7070a16148701e028256184378ae075d12b20e |
| h6_utility        | tracking_noise_2px  |        6 |            1.098055 |                          1.174864 |                         6 |          2.599919 |                      0.003391 |                0.021402 |       -0.003298 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h6_2026-07-29/v157f_uncertainty_response.csv | 3927debc3089f331cfe1fce73e7070a16148701e028256184378ae075d12b20e |
| h6_utility        | wrong_cell          |        6 |            1.225000 |                          1.336836 |                         6 |          2.954633 |                      0.129432 |                0.075041 |        0.096620 | <AIRI_SOURCE_ROOT>/outputs/lachance_foldlocal_semigroup_stress_v188_h6_2026-07-29/v157f_uncertainty_response.csv | 3927debc3089f331cfe1fce73e7070a16148701e028256184378ae075d12b20e |

The frozen base scale is reported separately from a frozen state-aware rule. Only clean update/no-update scales are selected on the validation movie; corruption-level multipliers are never tuned, and declared coordinate noise is propagated through the frozen packet/correction by Monte Carlo. No outer-test target is used for uncertainty calibration.

## Final-fit scope sensitivity

| objective_name   | fit_scope             |   horizon |   movies |   component_rmse |   component_rmse_std |   vector_rmse |       r2 |   gain_vs_no_update_percent |   movies_improved | source_path                                                                                                          | source_sha256                                                    |
|:-----------------|:----------------------|----------:|---------:|-----------------:|---------------------:|--------------:|---------:|----------------------------:|------------------:|:---------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| h1_strict        | train_only            |         1 |        6 |         3.470811 |             0.380402 |      4.908469 | 0.508750 |                    0.193095 |                 3 | <AIRI_SOURCE_ROOT>/outputs/lachance_v166_fit_scope_audit_v189_2026-07-29/v189_fit_scope_summary.csv | b9b515a45b52f1232d30f460204fa20d2b94d2cd49da3630dfc37880aefee20e |
| h1_strict        | train_plus_validation |         1 |        6 |         3.474374 |             0.377774 |      4.913506 | 0.507733 |                    0.081008 |                 4 | <AIRI_SOURCE_ROOT>/outputs/lachance_v166_fit_scope_audit_v189_2026-07-29/v189_fit_scope_summary.csv | b9b515a45b52f1232d30f460204fa20d2b94d2cd49da3630dfc37880aefee20e |
| h6_guard10       | train_only            |         6 |        6 |         5.538759 |             0.737520 |      7.832988 | 0.936788 |                   29.288929 |                 6 | <AIRI_SOURCE_ROOT>/outputs/lachance_v166_fit_scope_audit_v189_2026-07-29/v189_fit_scope_summary.csv | b9b515a45b52f1232d30f460204fa20d2b94d2cd49da3630dfc37880aefee20e |
| h6_guard10       | train_plus_validation |         6 |        6 |         5.500749 |             0.715725 |      7.779233 | 0.937721 |                   29.763802 |                 6 | <AIRI_SOURCE_ROOT>/outputs/lachance_v166_fit_scope_audit_v189_2026-07-29/v189_fit_scope_summary.csv | b9b515a45b52f1232d30f460204fa20d2b94d2cd49da3630dfc37880aefee20e |

The train-only row keeps the validation movie out of the final update fit. Its near-parity shows that the historical train+validation refit does not explain the v166 result.

## Observability decision

Deployable states passing the LaChance gate: `0`.
The public modality program is retained as a negative boundary, not as evidence that no useful prospective state can exist.

## Publication readiness

**Core evidence complete.**

| task_id   | priority   | task                                                   | status                  | progress                             | blocks_publication_ready   |
|:----------|:-----------|:-------------------------------------------------------|:------------------------|:-------------------------------------|:---------------------------|
| A1.2      | P0         | Complete exact KalmanNet outer-movie LOMO              | complete                | 18/18 fold-seed jobs                 | False                      |
| B0        | P0         | Choose license and initialize clean release repository | owner_decision_required | 0/1                                  | True                       |
| B1-B2     | P0         | Build and validate fresh-checkout public package       | pending                 | 0/1                                  | True                       |
| C0-C3     | P0         | Write manuscript, figures, and final claim audit       | pending                 | protocol and evidence skeleton ready | True                       |

Global SOTA wording remains forbidden. The defensible current claim is the strongest completed tested method under the frozen causal-online LaChance protocol.
