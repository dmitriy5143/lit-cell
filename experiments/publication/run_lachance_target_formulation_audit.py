#!/usr/bin/env python3
"""Fast target/formulation audit for LaChance forecasting.

This runner asks a pre-architectural question: are we stuck because the model is
weak, or because the target formulation hides the predictable signal?

It evaluates many deployable formulations with cheap probes:

- direct endpoint displacement;
- residual to current velocity / acceleration / tissue-flow baselines;
- direction + magnitude heads;
- stepwise h1-first sequence targets;
- feature blocks and shuffled controls.

No target/future quantity is used as an inference feature.  Target-aware rows are
not implemented here deliberately; those belong to the next teacher stage.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_fast_feature_triage as triage  # noqa: E402
import run_lachance_h1_sequence_raw_context_decoder as seq  # noqa: E402

DEFAULT_FEATURES = (
    ROOT
    / "outputs"
    / "lachance_raw_context_v2_grid_bulk_full60k_2026-06-19"
    / "raw_context_v2_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_target_formulation_audit_2026-06-22"
EPS = 1e-8


@dataclass
class EndpointForm:
    name: str
    horizon: int
    base_train: np.ndarray
    base_val: np.ndarray
    base_test: np.ndarray
    target_train: np.ndarray
    target_val: np.ndarray
    target_test: np.ndarray
    reconstruct: str = "vector"


@dataclass
class StepForm:
    name: str
    base_train: np.ndarray
    base_val: np.ndarray
    base_test: np.ndarray
    target_train: np.ndarray
    target_val: np.ndarray
    target_test: np.ndarray


def finite_json(value: Any) -> Any:
    return seq.finite_json(value)


def parse_ints(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text or "").split(",") if p.strip()]


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def safe_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1e6, 1e6).astype(np.float32, copy=False)


def vector_rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(safe_array(pred) - safe_array(y)), axis=-1))))


def vector_r2(y: np.ndarray, pred: np.ndarray) -> float:
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(pred, dtype=np.float64)
    sse = float(np.sum(np.square(y64 - p64)))
    yc = y64 - y64.mean(axis=0, keepdims=True)
    sst = float(np.sum(np.square(yc)))
    return float(1.0 - sse / sst) if sst > EPS else float("nan")


def cosine(y: np.ndarray, pred: np.ndarray) -> float:
    y = safe_array(y)
    pred = safe_array(pred)
    den = np.maximum(np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1), EPS)
    return float(np.mean(np.sum(y * pred, axis=1) / den))


def magnitude_ratio(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred, axis=1)) / max(float(np.mean(np.linalg.norm(y, axis=1))), EPS))


def gain_pct(base: float, value: float) -> float:
    return float((base - value) / max(abs(base), EPS) * 100.0)


def standardize(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if StandardScaler is None:
        raise RuntimeError("sklearn StandardScaler is unavailable")
    scaler = StandardScaler()
    train_z = scaler.fit_transform(safe_array(train_x))
    val_z = scaler.transform(safe_array(val_x))
    test_z = scaler.transform(safe_array(test_x))
    return safe_array(train_z), safe_array(val_z), safe_array(test_z)


def fit_predict(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    hgbdt_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_z, val_z, test_z = standardize(train_x, val_x, test_x)
    train_y = safe_array(train_y)
    val_y = safe_array(val_y)
    if model_name == "ridge":
        if Ridge is None:
            raise RuntimeError("sklearn Ridge is unavailable")
        best: tuple[float, float, Any] | None = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            model = Ridge(alpha=float(alpha), solver="auto")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(train_z, train_y)
                val_pred = safe_array(model.predict(val_z))
            rmse = vector_rmse(val_y.reshape(len(val_y), -1, 1), val_pred.reshape(len(val_pred), -1, 1))
            if best is None or rmse < best[0]:
                best = (rmse, float(alpha), model)
        assert best is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pred = safe_array(best[2].predict(test_z))
        return pred, {"alpha": best[1], "val_target_rmse": best[0]}
    if model_name == "hgbdt":
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("sklearn HistGradientBoostingRegressor is unavailable")
        preds = []
        val_preds = []
        for dim in range(train_y.shape[1]):
            model = HistGradientBoostingRegressor(
                max_iter=int(hgbdt_iter),
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.04,
                random_state=int(seed) + 13 * dim,
            )
            model.fit(train_z, train_y[:, dim])
            preds.append(model.predict(test_z))
            val_preds.append(model.predict(val_z))
        pred = safe_array(np.column_stack(preds))
        val_pred = safe_array(np.column_stack(val_preds))
        return pred, {"val_target_rmse": vector_rmse(val_y.reshape(len(val_y), -1, 1), val_pred.reshape(len(val_pred), -1, 1))}
    raise ValueError(f"unknown model={model_name}")


def feature_blocks(df: pd.DataFrame, names: list[str]) -> dict[str, list[str]]:
    specs = triage.make_feature_specs(df)
    out: dict[str, list[str]] = {"trajectory_only": []}
    for name in names:
        if name == "trajectory_only":
            out[name] = []
        elif name in specs:
            out[name] = specs[name]
    return out


def block_matrix(df: pd.DataFrame, cols: list[str], *, mode: str, seed: int) -> np.ndarray:
    traj = triage.trajectory_matrix(df)
    block = triage.clean_matrix(df, cols)
    if mode == "real":
        pass
    elif mode == "row_shuffled":
        block = triage.shuffled_by_row(block, seed)
    elif mode == "time_shuffled":
        block = seq.ifp.time_shuffled_image(df, cols, seed)
    else:
        raise ValueError(mode)
    return triage.concat_blocks(traj, block)


def endpoint(df: pd.DataFrame, h: int) -> np.ndarray:
    return df[[f"target_h{h}_dx", f"target_h{h}_dy"]].to_numpy(np.float32)


def steps(df: pd.DataFrame, max_h: int) -> np.ndarray:
    return np.stack(
        [df[[f"step{h}_dx", f"step{h}_dy"]].to_numpy(np.float32) for h in range(1, max_h + 1)],
        axis=1,
    )


def base_step(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "cv":
        return df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    if kind == "accel":
        v = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
        cols = [c for c in ["ax_px_s2", "ay_px_s2"] if c in df.columns]
        if len(cols) == 2:
            a = df[["ax_px_s2", "ay_px_s2"]].fillna(0.0).to_numpy(np.float32)
        else:
            a = np.zeros_like(v)
        return v + 0.5 * a
    match = re.match(r"tf_(.+)", kind)
    if match:
        token = match.group(1)
        u = f"tf_{token}_u"
        v = f"tf_{token}_v"
        if u in df.columns and v in df.columns:
            return df[[u, v]].fillna(0.0).to_numpy(np.float32)
    return np.zeros((len(df), 2), dtype=np.float32)


def find_flow_step_kinds(df: pd.DataFrame, limit: int) -> list[str]:
    kinds: list[str] = []
    for col in df.columns:
        if not col.startswith("tf_") or not col.endswith("_u"):
            continue
        base = col[3:-2]
        if f"tf_{base}_v" in df.columns and (
            "cur_center" in base
            or "cur_u_mean" in col
            or "cur_u_median" in col
            or "cur_center" in col
        ):
            kinds.append(f"tf_{base}")
    # Prefer center, then mean/median; keep deterministic order.
    def score(k: str) -> tuple[int, str]:
        return (0 if "center" in k else 1 if "mean" in k else 2, k)

    return sorted(set(kinds), key=score)[: int(limit)]


def make_endpoint_forms(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, horizons: list[int], flow_kinds: list[str]) -> list[EndpointForm]:
    forms: list[EndpointForm] = []
    for h in horizons:
        ytr, yva, yte = endpoint(train, h), endpoint(val, h), endpoint(test, h)
        ztr, zva, zte = [np.zeros_like(x) for x in (ytr, yva, yte)]
        forms.append(EndpointForm(f"h{h}_direct_endpoint", h, ztr, zva, zte, ytr, yva, yte))
        for kind in ["cv", "accel", *flow_kinds]:
            btr = float(h) * base_step(train, kind)
            bva = float(h) * base_step(val, kind)
            bte = float(h) * base_step(test, kind)
            forms.append(EndpointForm(f"h{h}_residual_to_{kind}", h, btr, bva, bte, ytr - btr, yva - bva, yte - bte))
        # Alternative head: direction + log magnitude of endpoint.
        mag_tr = np.linalg.norm(ytr, axis=1, keepdims=True)
        mag_va = np.linalg.norm(yva, axis=1, keepdims=True)
        mag_te = np.linalg.norm(yte, axis=1, keepdims=True)
        target_tr = np.concatenate([ytr / np.maximum(mag_tr, EPS), np.log1p(mag_tr)], axis=1)
        target_va = np.concatenate([yva / np.maximum(mag_va, EPS), np.log1p(mag_va)], axis=1)
        target_te = np.concatenate([yte / np.maximum(mag_te, EPS), np.log1p(mag_te)], axis=1)
        forms.append(EndpointForm(f"h{h}_direction_logmag", h, ztr, zva, zte, target_tr, target_va, target_te, reconstruct="direction_logmag"))
    return forms


def make_step_forms(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, max_h: int, flow_kinds: list[str]) -> list[StepForm]:
    forms: list[StepForm] = []
    str_, sva, ste = steps(train, max_h), steps(val, max_h), steps(test, max_h)
    zero_tr = np.zeros_like(str_)
    zero_va = np.zeros_like(sva)
    zero_te = np.zeros_like(ste)
    forms.append(StepForm("step_direct", zero_tr, zero_va, zero_te, str_, sva, ste))
    for kind in ["cv", "accel", *flow_kinds[:4]]:
        btr = np.repeat(base_step(train, kind)[:, None, :], max_h, axis=1)
        bva = np.repeat(base_step(val, kind)[:, None, :], max_h, axis=1)
        bte = np.repeat(base_step(test, kind)[:, None, :], max_h, axis=1)
        forms.append(StepForm(f"step_residual_to_{kind}", btr, bva, bte, str_ - btr, sva - bva, ste - bte))
    return forms


def reconstruct_endpoint(form: EndpointForm, pred_target: np.ndarray) -> np.ndarray:
    if form.reconstruct == "vector":
        return form.base_test + pred_target
    if form.reconstruct == "direction_logmag":
        direction = pred_target[:, :2]
        direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), EPS)
        mag = np.expm1(np.clip(pred_target[:, 2:3], -4.0, 6.0))
        return direction * mag
    raise ValueError(form.reconstruct)


def evaluate_endpoint_rows(
    *,
    dataset: str,
    seed: int,
    model: str,
    block: str,
    control: str,
    form: EndpointForm,
    pred_target: np.ndarray,
    info: dict[str, Any],
) -> dict[str, Any]:
    y = endpoint_target_original(form)
    pred = reconstruct_endpoint(form, pred_target)
    base = form.base_test if form.reconstruct == "vector" else np.zeros_like(pred)
    return {
        "dataset": dataset,
        "seed": seed,
        "scope": "endpoint",
        "horizon": int(form.horizon),
        "target_form": form.name,
        "model": model,
        "feature_block": block,
        "control": control,
        "rmse_px": vector_rmse(y, pred),
        "base_rmse_px": vector_rmse(y, base),
        "gain_vs_base_pct": gain_pct(vector_rmse(y, base), vector_rmse(y, pred)),
        "r2": vector_r2(y, pred),
        "cosine": cosine(y, pred),
        "magnitude_ratio": magnitude_ratio(y, pred),
        "target_space_rmse": vector_rmse(form.target_test.reshape(len(form.target_test), -1, 1), pred_target.reshape(len(pred_target), -1, 1)),
        **info,
    }


def endpoint_target_original(form: EndpointForm) -> np.ndarray:
    if form.reconstruct == "vector":
        return form.base_test + form.target_test
    # direction/logmag target encodes the raw endpoint directly.
    direction = form.target_test[:, :2]
    mag = np.expm1(form.target_test[:, 2:3])
    return direction * mag


def evaluate_step_rows(
    *,
    dataset: str,
    seed: int,
    model: str,
    block: str,
    control: str,
    form: StepForm,
    pred_target_flat: np.ndarray,
    horizons: list[int],
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    max_h = form.target_test.shape[1]
    pred_res = pred_target_flat.reshape(len(form.target_test), max_h, 2)
    pred_steps = form.base_test + pred_res
    y_steps = form.base_test + form.target_test
    base_steps = form.base_test
    rows: list[dict[str, Any]] = []
    for h in horizons:
        y = y_steps[:, :h].sum(axis=1)
        pred = pred_steps[:, :h].sum(axis=1)
        base = base_steps[:, :h].sum(axis=1)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "scope": "stepwise",
                "horizon": int(h),
                "target_form": form.name,
                "model": model,
                "feature_block": block,
                "control": control,
                "rmse_px": vector_rmse(y, pred),
                "base_rmse_px": vector_rmse(y, base),
                "gain_vs_base_pct": gain_pct(vector_rmse(y, base), vector_rmse(y, pred)),
                "r2": vector_r2(y, pred),
                "cosine": cosine(y, pred),
                "magnitude_ratio": magnitude_ratio(y, pred),
                "target_space_rmse": vector_rmse(form.target_test.reshape(len(form.target_test), -1, 1), pred_target_flat.reshape(len(pred_target_flat), -1, 1)),
                **info,
            }
        )
    return rows


def write_report(out_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    best = summary[summary["control"].eq("real")].sort_values(["horizon", "rmse_px"]).groupby(["scope", "horizon"]).head(12)
    controls = summary[summary["control"].ne("real")].sort_values(["horizon", "rmse_px"]).groupby(["scope", "horizon"]).head(8)
    lines = [
        "# LaChance Target/Formulation Audit",
        "",
        "## Payload",
        "",
        "```json",
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top Real Formulations",
        "",
        best.to_markdown(index=False) if len(best) else "_No rows._",
        "",
        "## Best Controls",
        "",
        controls.to_markdown(index=False) if len(controls) else "_No controls._",
        "",
        "## Reading Guide",
        "",
        "- A formulation is promising only if its real block beats its base and shuffled controls.",
        "- Strong gains for residual-to-flow imply target should be defined relative to tissue motion.",
        "- Strong stepwise gains imply h1-first decomposition should precede endpoint regression.",
        "- If direction/logmag beats vector RMSE, use a magnitude-aware decoder.",
    ]
    (out_dir / "target_formulation_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> pd.DataFrame:
    features = pd.read_csv(args.features)
    horizons = parse_ints(args.horizons)
    max_h = max(horizons)
    full = seq.build_sequence_table(
        features=features,
        table_root=args.table_root,
        dataset=args.dataset,
        max_horizon=max_h,
    )
    split = seq.make_split(
        full,
        parse_ints(args.train_sequences),
        parse_ints(args.val_sequences),
        parse_ints(args.test_sequences),
        int(args.seed),
    )
    train = seq.sample_rows(split.train, int(args.max_train_rows), int(args.seed) + 11).reset_index(drop=True)
    val = seq.sample_rows(split.val, int(args.max_val_rows), int(args.seed) + 13).reset_index(drop=True)
    test = seq.sample_rows(split.test, int(args.max_test_rows), int(args.seed) + 17).reset_index(drop=True)
    blocks = feature_blocks(full, parse_strs(args.blocks))
    flow_kinds = find_flow_step_kinds(full, int(args.flow_limit))
    endpoint_forms = make_endpoint_forms(train, val, test, horizons, flow_kinds)
    step_forms = make_step_forms(train, val, test, max_h, flow_kinds)
    rows: list[dict[str, Any]] = []
    models = parse_strs(args.models)
    controls = ["real"]
    if args.include_controls:
        controls += ["row_shuffled", "time_shuffled"]

    for block_name, cols in blocks.items():
        for control in controls:
            if control != "real" and block_name == "trajectory_only":
                continue
            xtr = block_matrix(train, cols, mode=control, seed=int(args.seed) + 101)
            xva = block_matrix(val, cols, mode=control, seed=int(args.seed) + 103)
            xte = block_matrix(test, cols, mode=control, seed=int(args.seed) + 107)
            for model in models:
                for form in endpoint_forms:
                    pred, info = fit_predict(
                        model,
                        xtr,
                        form.target_train,
                        xva,
                        form.target_val,
                        xte,
                        seed=int(args.seed) + form.horizon,
                        hgbdt_iter=int(args.hgbdt_iter),
                    )
                    rows.append(
                        evaluate_endpoint_rows(
                            dataset=args.dataset,
                            seed=int(args.seed),
                            model=model,
                            block=block_name,
                            control=control,
                            form=form,
                            pred_target=pred,
                            info={"feature_dim": int(xtr.shape[1]), **info},
                        )
                    )
                for form in step_forms:
                    pred, info = fit_predict(
                        model,
                        xtr,
                        form.target_train.reshape(len(form.target_train), -1),
                        xva,
                        form.target_val.reshape(len(form.target_val), -1),
                        xte,
                        seed=int(args.seed) + 997,
                        hgbdt_iter=int(args.hgbdt_iter),
                    )
                    rows.extend(
                        evaluate_step_rows(
                            dataset=args.dataset,
                            seed=int(args.seed),
                            model=model,
                            block=block_name,
                            control=control,
                            form=form,
                            pred_target_flat=pred,
                            horizons=horizons,
                            info={"feature_dim": int(xtr.shape[1]), **info},
                        )
                    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--table-root", type=Path, default=seq.ifp.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--models", default="ridge")
    parser.add_argument(
        "--blocks",
        default="trajectory_only,ms_shape_tf_all,ms_shape_tf_alignment_rc_core,ms_all_tf_all_rc,rc_all,obs_context_core",
    )
    parser.add_argument("--max-train-rows", type=int, default=20000)
    parser.add_argument("--max-val-rows", type=int, default=8000)
    parser.add_argument("--max-test-rows", type=int, default=8000)
    parser.add_argument("--flow-limit", type=int, default=8)
    parser.add_argument("--hgbdt-iter", type=int, default=80)
    parser.add_argument("--include-controls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = run(args)
    summary.to_csv(args.out_dir / "target_formulation_summary.csv", index=False)
    write_report(args.out_dir, summary, args)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
