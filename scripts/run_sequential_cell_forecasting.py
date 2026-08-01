#!/usr/bin/env python3
"""Dispatch descriptive publication workflows to their exact provenance runners."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "experiments" / "publication"

WORKFLOWS = {
    "outer-lomo-benchmark": "run_lachance_online_lomo_benchmark_v102.py",
    "online-neural-screen": "run_lachance_online_architecture_benchmark_v99.py",
    "online-core": "run_lachance_causal_innovation_state_space_v97.py",
    "fold-local-transport": "run_lachance_foldlocal_semigroup_confirmation_v157e.py",
    "transport-pareto": "run_lachance_foldlocal_semigroup_pareto_v157h.py",
    "frozen-confirmation": "run_lachance_streaming_transport_confirmation_v160.py",
    "external-lomo": "run_lachance_external_movie_lomo_publication_v165.py",
    "learned-comparators": "run_lachance_confirmation_learned_comparators_v193.py",
    "kalmannet": "run_lachance_kalmannet_outer_lomo_v188.py",
    "sparse-transport": "run_lachance_sparse_pareto_transport_v193.py",
    "field-law": "run_mdck_equivariant_field_law_v197.py",
    "effective-potential": "run_mdck_effective_potential_audit_v198.py",
    "graph-bridge": "run_lachance_equivariant_graph_bridge_v199.py",
    "field-dynamics": "run_mdck_effective_functional_dynamics_v200.py",
    "probabilistic-field": "run_lachance_probabilistic_graph_closure_v201.py",
    "unseen-field": "run_lachance_equivariant_graph_unseen_v202.py",
    "deepsea-transfer": "run_deepsea_multimodal_validation_v204.py",
    "h1-audit": "run_lachance_h1_evidence_bundle_v205.py",
    "lifeact-segmentation": "run_lifeact_mdck_segmentation_identity_gate_v206.py",
    "lifeact-state": "run_lifeact_mdck_mechanochemical_state_gate_v207.py",
    "lifeact-uncertainty": "evaluate_lifeact_mdck_state_uncertainty_gate_v208.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", nargs="?", choices=sorted(WORKFLOWS))
    parser.add_argument(
        "--list",
        action="store_true",
        help="List scientific workflow names and exact provenance entry points.",
    )
    args, forwarded = parser.parse_known_args()

    if args.list:
        for name, filename in WORKFLOWS.items():
            print(f"{name:24s} {filename}")
        return
    if args.workflow is None:
        parser.error("provide a workflow or use --list")

    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    runner = RUNNERS / WORKFLOWS[args.workflow]
    if not runner.is_file():
        raise FileNotFoundError(f"Publication runner is missing: {runner}")
    subprocess.run([sys.executable, str(runner), *forwarded], check=True)


if __name__ == "__main__":
    main()
