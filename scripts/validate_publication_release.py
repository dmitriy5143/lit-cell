#!/usr/bin/env python3
"""Validate the portable publication release and registered manuscript claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
MANUSCRIPT = ROOT / "manuscript" / "main_ru.tex"
REPORT = EVIDENCE / "publication_release_validation.json"


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def close(actual: float, expected: float, atol: float = 5e-6) -> bool:
    return math.isclose(float(actual), expected, abs_tol=atol, rel_tol=0.0)


def one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def validate_manifest(checks: list[str]) -> None:
    path = EVIDENCE / "release_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("algorithm") == "sha256", "release manifest uses SHA-256", checks)
    files = payload.get("files", {})
    require(len(files) >= 150, "release manifest covers at least 150 artifacts", checks)
    for relative, metadata in files.items():
        artifact = ROOT / relative
        require(artifact.is_file(), f"manifest artifact exists: {relative}", checks)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        require(digest == metadata["sha256"], f"manifest hash matches: {relative}", checks)


def validate_primary(checks: list[str]) -> None:
    benchmark = pd.read_csv(EVIDENCE / "v188" / "v188_primary_online_benchmark.csv")
    require(benchmark["movies"].eq(6).all(), "primary benchmark uses six outer movies", checks)
    expected = {
        ("v166_h1_strict", 1): (3.474374, 0.507733),
        ("v166_h1_strict", 6): (6.784638, 0.905043),
        ("v166_h6_utility", 1): (3.807965, 0.407988),
        ("v166_h6_utility", 6): (5.500749, 0.937721),
        ("v97_no_update", 6): (7.812040, 0.874604),
        ("hgbdt_v52", 6): (8.404088, 0.855278),
        ("gru_track", 6): (8.412099, 0.855164),
        ("kalmannet", 6): (8.794405, 0.840182),
    }
    for key, (rmse, r2) in expected.items():
        row = one(benchmark, method=key[0], horizon=key[1])
        require(close(row.component_rmse, rmse), f"frozen RMSE matches {key}", checks)
        require(close(row.r2, r2), f"frozen R2 matches {key}", checks)

    paired = pd.read_csv(EVIDENCE / "v188" / "v188_paired_movie_statistics.csv")
    primary = paired[paired["confirmatory"].eq(True)].sort_values("hypothesis_id")
    require(primary["hypothesis_id"].tolist() == ["H1", "H2"], "confirmatory family is exactly H1/H2", checks)
    h1, h2 = primary.iloc[0], primary.iloc[1]
    require(close(h1.holm_adjusted_p, 0.84375), "H1 Holm-adjusted p is 0.84375", checks)
    require(close(h2.exact_two_sided_sign_flip_p, 0.03125), "H2 raw exact p is 0.03125", checks)
    require(close(h2.holm_adjusted_p, 0.0625), "H2 Holm-adjusted p is 0.0625", checks)


def validate_confirmation_and_external(checks: list[str]) -> None:
    confirmation = pd.read_csv(EVIDENCE / "v188" / "v188_configuration_unseen_confirmation.csv")
    row = one(confirmation, objective_name="h6_guard10", control="real", horizon=6)
    require(int(row.movies) == 7, "frozen confirmation covers seven movies", checks)
    require(close(row.component_rmse, 4.819533), "confirmation h6 RMSE is 4.819533", checks)
    require(close(row.r2, 0.952466), "confirmation h6 R2 is 0.952466", checks)
    require(close(row.gain_vs_v97_percent, 25.816659), "confirmation h6 gain is 25.816659%", checks)

    external = pd.read_csv(EVIDENCE / "v188" / "v188_external_nested_lomo.csv")
    expected = {
        ("HUVEC", "h6_guard10", 6): (18, 1.439869, 0.972607, 11.023770),
        ("MDAMB231", "h6_guard10", 6): (17, 31.337439, 0.023738, 6.328989),
    }
    for key, values in expected.items():
        row = one(external, dataset=key[0], objective=key[1], control="real", horizon=key[2])
        require(int(row.outer_folds) == values[0], f"external fold count matches {key[0]}", checks)
        require(close(row.component_rmse_macro, values[1]), f"external RMSE matches {key[0]}", checks)
        require(close(row.r2_macro, values[2]), f"external R2 matches {key[0]}", checks)
        require(close(row.gain_percent_macro, values[3]), f"external gain matches {key[0]}", checks)


def validate_late_evidence(checks: list[str]) -> None:
    deepsea = pd.read_csv(EVIDENCE / "deepsea_v204" / "deepsea_v204_key_results.csv")
    row = one(deepsea, module="complete_system", variant="v166_external_h1_strict", horizon=1)
    require(close(row.movie_macro_rmse, 0.155282), "DeepSea strict h1 RMSE matches", checks)
    row = one(deepsea, module="complete_system", variant="v166_external_h6_utility", horizon=6)
    require(close(row.movie_macro_rmse, 0.200450), "DeepSea utility h6 RMSE matches", checks)

    h1 = pd.read_csv(EVIDENCE / "h1_v205" / "v205_h1_h6_aggregate.csv")
    row = one(h1, objective_name="lambda_00", variant="lambda_00_real", horizon=1)
    require(close(row.component_rmse_mean, 3.474374), "late h1 evidence matches primary", checks)

    lifeact = pd.read_csv(EVIDENCE / "lifeact_v206_v208" / "v208_uncertainty_decision.csv")
    row = one(lifeact, protocol="leave_one_sequence_out")
    require(close(row.coord_student_t4_nll, 3.649278), "LifeAct coordinate Student-t4 NLL matches", checks)
    require(close(row.real_student_t4_nll, 3.613868), "LifeAct real-state Student-t4 NLL matches", checks)
    require(close(row.best_control_student_t4_nll, 3.644185), "LifeAct best-control Student-t4 NLL matches", checks)


def validate_manuscript(checks: list[str]) -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    required_tokens = [
        "3,474", "5,501", "4,820", "0{,}938", "0{,}952",
        "0{,}0625", "3,649", "3,614", "3,644", r"20\,000",
    ]
    for token in required_tokens:
        require(token in text, f"manuscript contains registered token {token}", checks)
    forbidden = [r"v166", r"v97", r"HGBDT v52", r"p_\{97\}", "производствен"]
    for token in forbidden:
        require(re.search(token, text, flags=re.IGNORECASE) is None, f"manuscript omits internal term {token}", checks)
    require("Holm" in text and "H1" in text and "H2" in text, "manuscript states multiplicity family", checks)
    require("глобальное превосходство" in text, "manuscript limits global-SOTA claim", checks)


def validate_features_and_figures(checks: list[str]) -> None:
    dictionary = pd.read_csv(EVIDENCE / "raw_context_v2_source_dictionary.csv")
    require(len(dictionary) == 1093, "source feature dictionary has 1,093 columns", checks)
    require(dictionary["column"].is_unique, "source feature dictionary columns are unique", checks)
    require(not dictionary["family"].eq("other").any(), "all source columns have a declared family", checks)

    ledger = pd.read_csv(EVIDENCE / "architecture_search_ledger.csv")
    require(len(ledger) >= 15, "architecture search ledger covers at least 15 branches", checks)
    require(ledger["branch"].is_unique, "architecture search branches are unique", checks)

    for index in range(1, 9):
        pdfs = list((ROOT / "manuscript" / "figures").glob(f"fig{index}_*.pdf"))
        require(len(pdfs) == 1 and pdfs[0].stat().st_size > 10_000, f"Figure {index} PDF exists", checks)
        pngs = list((ROOT / "manuscript" / "figures").glob(f"fig{index}_*.png"))
        require(len(pngs) == 1, f"Figure {index} PNG preview exists", checks)
        with Image.open(pngs[0]) as image:
            require(image.width >= 1200 and image.height >= 650, f"Figure {index} raster is publication-sized", checks)


def validate_code_layout(checks: list[str]) -> None:
    runners = sorted((ROOT / "experiments" / "publication").glob("*.py"))
    require(len(runners) >= 90, "publication runner dependency closure is present", checks)
    for runner in runners:
        text = runner.read_text(encoding="utf-8")
        root_assignments = re.findall(
            r"^ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]",
            text,
            flags=re.MULTILINE,
        )
        for depth in root_assignments:
            require(depth == "2", f"portable repository root in {runner.name}", checks)
    require((ROOT / "src" / "airi_forecasting" / "streaming_forecaster.py").is_file(), "reusable forecasting package is present", checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()
    checks: list[str] = []
    if not args.skip_manifest:
        validate_manifest(checks)
    validate_primary(checks)
    validate_confirmation_and_external(checks)
    validate_late_evidence(checks)
    validate_manuscript(checks)
    validate_features_and_figures(checks)
    validate_code_layout(checks)
    payload = {"status": "PASS", "checks": len(checks), "details": checks}
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: {len(checks)} checks")


if __name__ == "__main__":
    main()
