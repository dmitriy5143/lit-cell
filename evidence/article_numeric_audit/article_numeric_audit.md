# Числовой аудит журнальной рукописи

Проверено количественных утверждений: **275**.

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

## Проверенные основные значения

| Утверждение | Значение | Единица | Источник |
|---|---:|---|---|
| `potential_rmse` | 0.10861 | um/min | `outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_summary.csv` |
| `potential_gain` | 8.70376 | % | `outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_summary.csv` |
| `potential_gain_retained` | 98.5774 | % | `outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_summary.csv` |
| `potential_advective_sign_p` | 0.143139 | p | `outputs/mdck_effective_potential_audit_v198_2026-07-30/v198_potential_outer_folds.csv` |
| `primary_h1_rmse` | 3.47437 | px | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `primary_h1_r2` | 0.507733 | fraction | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `primary_h6_rmse` | 5.50075 | px | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `primary_h6_r2` | 0.937721 | fraction | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_primary_online_benchmark.csv` |
| `h6_test_relative_gain_percent` | 29.7638 | % movie-macro | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_paired_movie_statistics.csv` |
| `h6_test_holm_adjusted_p` | 0.0625 | p | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_paired_movie_statistics.csv` |
| `unseen_h6_rmse` | 4.81953 | px | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_configuration_unseen_confirmation.csv` |
| `unseen_h6_r2` | 0.952466 | fraction | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_configuration_unseen_confirmation.csv` |
| `unseen_h6_gain` | 25.8167 | % | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_configuration_unseen_confirmation.csv` |
| `unseen_h6_exact_p` | 0.015625 | p | `outputs/lachance_streaming_transport_confirmation_v160_full_2026-07-27/v160_confirmation_aggregate.csv` |
| `huvec_component_rmse_macro` | 1.43987 | px | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_external_nested_lomo.csv` |
| `mdamb231_component_rmse_macro` | 31.3374 | px | `outputs/lachance_publication_bundle_v188_2026-07-29/v188_external_nested_lomo.csv` |
| `edge_component_rmse_mean` | 5.26074 | px | `outputs/lachance_publication_bundle_v166_2026-07-27/v166_dimensionless_transfer.csv` |
| `graph_potential_dev_h6_rmse` | 5.58148 | px | `outputs/lachance_equivariant_graph_bridge_v199_full_2026-07-30/v199_graph_bridge_aggregate.csv` |
| `graph_potential_dev_h6_gain` | 28.6118 | % movie-macro | `outputs/lachance_equivariant_graph_bridge_v199_full_2026-07-30/v199_graph_bridge_aggregate.csv` |
| `graph_potential_unseen_h6_rmse` | 4.84238 | px | `outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_aggregate.csv` |
| `graph_potential_unseen_h6_r2` | 0.952006 | fraction | `outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_aggregate.csv` |
| `graph_potential_unseen_h6_gain` | 25.4568 | % movie-macro | `outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_aggregate.csv` |
| `graph_potential_projection_r2` | 0.951466 | fraction | `outputs/lachance_equivariant_graph_bridge_v199_full_2026-07-30/v199_graph_bridge_projection.csv` |
| `graph_unseen_wrong_cell_advantage` | 2.52938 | px | `outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_controls.csv` |
| `graph_unseen_stale_time_advantage` | 0.685659 | px | `outputs/lachance_equivariant_graph_unseen_v202_full_2026-07-30/v202_unseen_graph_controls.csv` |
| `functional_potential_delta` | -0.0404751 | functional units | `outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_finite_functional_summary.csv` |
| `functional_potential_decrease_fraction` | 1 | fraction | `outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_finite_functional_summary.csv` |
| `functional_potential_h6_rmse` | 0.158442 | um/min | `outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_field_rollout_summary.csv` |
| `functional_potential_h6_gain` | 6.30254 | % island-macro | `outputs/mdck_effective_functional_dynamics_v200_full_2026-07-30/v200_field_rollout_summary.csv` |
| `probability_utility_h6_joint_nll_mean` | 6.0065 | NLL | `outputs/lachance_probabilistic_graph_closure_v201_full_2026-07-30/v201_probabilistic_aggregate.csv` |
| `probability_utility_h6_conformal_radial_coverage90_mean` | 0.896604 | fraction | `outputs/lachance_probabilistic_graph_closure_v201_full_2026-07-30/v201_probabilistic_aggregate.csv` |
| `lifeact_mean_loo_rmse` | 11.082 | px | `outputs/lifeact_mdck_mechanochemical_state_gate_v207_center60_multiseed_2026-08-01/v207_multiseed_decision_aggregate.csv` |
| `lifeact_mean_loo_gain` | -0.125009 | % | `outputs/lifeact_mdck_mechanochemical_state_gate_v207_center60_multiseed_2026-08-01/v207_multiseed_decision_aggregate.csv` |
| `lifeact_uncertainty_student_real` | 3.61387 | NLL | `outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |
| `lifeact_uncertainty_student_coord` | 3.64928 | NLL | `outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |
| `lifeact_uncertainty_student_control` | 3.64419 | NLL | `outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |
| `lifeact_uncertainty_error_spearman` | 0.310576 | rho | `outputs/lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01/v208_uncertainty_decision.csv` |

Полная машинно-читаемая таблица сохранена в `article_numeric_claims.csv`.
