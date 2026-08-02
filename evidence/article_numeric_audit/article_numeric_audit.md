# Числовой аудит журнальной рукописи

Проверено количественных утверждений: **287**.

Все значения прочитаны из замороженных таблиц результатов; ручные числовые константы в новых рисунках не допускаются.

## Критические правила интерпретации

- Коллективная добавка на h1 статистически не подтверждена: точное двустороннее p=0,84375.
- Эффект h6 положителен в 6/6 фильмах, но p=0,0625 после поправки Холма не пересекает порог 0,05.
- Выигрыш h6 29,7638% является средним по относительным выигрышам фильмов; отношение агрегированных RMSE оценивает другую величину.
- Последовательный h6 получает промежуточные наблюдения и не сопоставим с оптимумом открытого прогноза из одной исходной точки.
- Высокий R2 на h6 частично отражает большую дисперсию h6-цели и не означает столь же сильную наблюдаемость h1.
- MDA-MB-231 показывает воспроизводимый относительный выигрыш при низком абсолютном R2=0,0237.
- Для MDCK Edge показан разброс между инициализациями, а интервалы HUVEC и MDA получены вложенным бутстрепом фильмов.
- Панель калибровки должна явно разделять замороженные h1- и h6-режимы.
- Проверка замороженной текущей конфигурации на семи фильмах MDCK Bulk 10--16 дала h6-эффект 7/7 и непоправленное p=0,015625. Эти фильмы ранее участвовали в широком разведочном поиске проекта, поэтому результат следует сообщать отдельно от разработческих шести фильмов, но не называть полностью независимым перспективным опытом.
- Потенциальный сектор v198 является эффективным функционалом кинематической инновации. Неизвестная подвижность и провал механического шлюза запрещают трактовать его коэффициенты как физическую энергию или материальные константы.
- Конечное уменьшение функционала подтверждено для потенциальной карты, но наблюдаемый переход не превосходит временную перестановку; это свойство модели, а не доказанная энергия ткани.
- E(2)-эквивариантный потенциальный граф является интерпретируемым суррогатом плотной поправки: на замороженном h6 он дает 4,842 против 4,820 у производственного оператора.
- DeepSea v204 является частичным внешним переносом, а не положительным мультимодальным подтверждением: накопительный v166 улучшает h6 относительно постоянной скорости на 3,66%, но ухудшает h1 относительно собственного априорного прогноза; причинное состояние масок не прошло жесткие контроли.
- Полная v205-кривая h1--h6 содержит 11 недоминируемых точек, но только крайняя h1-точка была исходно подтверждающей. Независимый нижний предел шума локализации для LaChance не установлен.
- LifeAct-MDCK не улучшил условное среднее h1 в межусловной проверке. Положительный результат относится только к условному масштабу ошибки: Student-t4 NLL 3,649 -> 3,614 против 3,644 у лучшего контроля; из-за трех условий и автоматических треков он считается исследовательским.
- C2C12 v209 подтверждает структурный перенос после доменного обучения, а не абсолютное лидерство или перенос весов без настройки: E(2)-оператор улучшает согласованное условное среднее на h6 на 1,22% в 3/3 экспериментах, но постоянная скорость имеет меньшую абсолютную RMSE 4,860 против 5,088. Ручные и автоматические аннотации не объединяются.

## Проверенные основные значения

| Утверждение | Значение | Единица | Источник |
|---|---:|---|---|
| `potential_rmse` | 0.10861 | um/min | `evidence/article_numeric_sources/outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_summary.csv` |
| `potential_gain` | 8.70376 | % | `evidence/article_numeric_sources/outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_summary.csv` |
| `potential_gain_retained` | 98.5774 | % | `evidence/article_numeric_sources/outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_summary.csv` |
| `potential_advective_sign_p` | 0.143139 | p | `evidence/article_numeric_sources/outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_outer_folds.csv` |
| `primary_h1_rmse` | 3.47437 | px | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `primary_h1_r2` | 0.507733 | fraction | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `primary_h6_rmse` | 5.50075 | px | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `primary_h6_r2` | 0.937721 | fraction | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `h6_test_relative_gain_percent` | 29.7638 | % movie-macro | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_paired_movie_statistics.csv` |
| `h6_test_holm_adjusted_p` | 0.0625 | p | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_paired_movie_statistics.csv` |
| `unseen_h6_rmse` | 4.81953 | px | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_configuration_unseen_confirmation.csv` |
| `unseen_h6_r2` | 0.952466 | fraction | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_configuration_unseen_confirmation.csv` |
| `unseen_h6_gain` | 25.8167 | % | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_configuration_unseen_confirmation.csv` |
| `unseen_h6_exact_p` | 0.015625 | p | `evidence/article_numeric_sources/outputs/lachance_streaming_transport_confirmation_v160_full_2026-07-27/v160_confirmation_aggregate.csv` |
| `huvec_component_rmse_macro` | 1.43987 | px | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_external_nested_lomo.csv` |
| `mdamb231_component_rmse_macro` | 31.3374 | px | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v188_2026-07-29/v188_external_nested_lomo.csv` |
| `edge_component_rmse_mean` | 5.26074 | px | `evidence/article_numeric_sources/outputs/lachance_publication_bundle_v166_2026-07-27/v166_dimensionless_transfer.csv` |
| `graph_potential_dev_h6_rmse` | 5.58148 | px | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_bridge_v199_full_2026-07-30/v199_graph_bridge_aggregate.csv` |
| `graph_potential_dev_h6_gain` | 28.6118 | % movie-macro | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_bridge_v199_full_2026-07-30/v199_graph_bridge_aggregate.csv` |
| `graph_potential_unseen_h6_rmse` | 4.84238 | px | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_aggregate.csv` |
| `graph_potential_unseen_h6_r2` | 0.952006 | fraction | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_aggregate.csv` |
| `graph_potential_unseen_h6_gain` | 25.4568 | % movie-macro | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_aggregate.csv` |
| `graph_potential_projection_r2` | 0.951466 | fraction | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_bridge_v199_full_2026-07-30/v199_graph_bridge_projection.csv` |
| `graph_unseen_wrong_cell_advantage` | 2.52938 | px | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_controls.csv` |
| `graph_unseen_stale_time_advantage` | 0.685659 | px | `evidence/article_numeric_sources/outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_controls.csv` |
| `functional_potential_delta` | -0.0404751 | functional units | `evidence/article_numeric_sources/outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_finite_functional_summary.csv` |
| `functional_potential_decrease_fraction` | 1 | fraction | `evidence/article_numeric_sources/outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_finite_functional_summary.csv` |
| `functional_potential_h6_rmse` | 0.158442 | um/min | `evidence/article_numeric_sources/outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_field_rollout_summary.csv` |
| `functional_potential_h6_gain` | 6.30254 | % island-macro | `evidence/article_numeric_sources/outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_field_rollout_summary.csv` |
| `probability_utility_h6_joint_nll_mean` | 6.0065 | NLL | `evidence/article_numeric_sources/outputs/lachance_probabilistic_graph_closure_v201_full_2026-07-30/v201_probabilistic_aggregate.csv` |
| `probability_utility_h6_conformal_radial_coverage90_mean` | 0.896604 | fraction | `evidence/article_numeric_sources/outputs/lachance_probabilistic_graph_closure_v201_full_2026-07-30/v201_probabilistic_aggregate.csv` |
| `lifeact_mean_loo_rmse` | 11.082 | px | `evidence/article_numeric_sources/outputs/lifeact_mdck_mechanochemical_state_gate_v207_center60_multiseed_2026-08-01/v207_multiseed_decision_aggregate.csv` |
| `lifeact_mean_loo_gain` | -0.125009 | % | `evidence/article_numeric_sources/outputs/lifeact_mdck_mechanochemical_state_gate_v207_center60_multiseed_2026-08-01/v207_multiseed_decision_aggregate.csv` |
| `lifeact_uncertainty_student_real` | 3.61387 | NLL | `evidence/article_numeric_sources/outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |
| `lifeact_uncertainty_student_coord` | 3.64928 | NLL | `evidence/article_numeric_sources/outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |
| `lifeact_uncertainty_student_control` | 3.64419 | NLL | `evidence/article_numeric_sources/outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |
| `lifeact_uncertainty_error_spearman` | 0.310576 | rho | `evidence/article_numeric_sources/outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |
| `c2c12_auto_h1_baseline_rmse` | 4.33875 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_main_metrics.csv` |
| `c2c12_auto_h1_real_rmse` | 4.35515 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_main_metrics.csv` |
| `c2c12_auto_h6_baseline_rmse` | 5.14843 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_main_metrics.csv` |
| `c2c12_auto_h6_real_rmse` | 5.08846 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_main_metrics.csv` |
| `c2c12_auto_h6_real_r2` | 0.707292 | fraction | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_main_metrics.csv` |
| `c2c12_auto_h6_paired_gain` | 1.22068 | % | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_experiment_gains.csv` |
| `c2c12_auto_cv_h6_rmse` | 4.86016 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_main_metrics.csv` |
| `c2c12_manual_h6_real_rmse` | 1.3347 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_main_metrics.csv` |
| `c2c12_manual_h6_paired_gain` | 11.3956 | % | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_experiment_gains.csv` |
| `c2c12_auto_no_update_advantage` | 0.0599725 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_cluster_bootstrap.csv` |
| `c2c12_auto_wrong_cell_advantage` | 0.0742839 | px | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_cluster_bootstrap.csv` |
| `c2c12_dominant_scale_multiplier` | 2 | d_nn | `evidence/article_numeric_sources/outputs/c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02/v209_prx_operator_scale_norms.csv` |

Полная машинно-читаемая таблица сохранена в `article_numeric_claims.csv`.
