#!/usr/bin/env python3
"""Manual model promotion CLI.

Usage:
    uv run python scripts/promote_model.py --run-id abc123 --test-auc 0.85
    uv run python scripts/promote_model.py --list
    uv run python scripts/promote_model.py --version 3 --set-alias champion
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual model promotion tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="MLflow run ID to promote")
    group.add_argument("--list", action="store_true", help="List all registered model versions")

    parser.add_argument("--test-auc", type=float, default=None, help="Override test AUC for promotion gate")
    parser.add_argument("--skip-gate", action="store_true", help="Force promote without AUC gate")
    parser.add_argument("--version", help="Specific version to tag/alias")
    parser.add_argument("--set-alias", help="Set an alias on a specific version")

    args = parser.parse_args()

    from loan_risk.config import get_settings
    from loan_risk.logging_setup import configure_logging
    from loan_risk.registry.client import MLflowRegistryClient

    configure_logging(json_output=False)
    cfg = get_settings()
    registry = MLflowRegistryClient()

    if args.list:
        versions = registry.list_versions()
        if not versions:
            print("No registered model versions found.")
        else:
            print(f"Registered versions for '{cfg.mlflow.registered_model_name}':")
            for v in versions:
                print(f"  v{v['version']}  run_id={v['run_id'][:8]}  status={v['status']}  tags={v['tags']}")
        return 0

    if args.run_id:
        if args.skip_gate:
            import mlflow
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            version = mlflow.register_model(
                model_uri=f"runs:/{args.run_id}/model",
                name=cfg.mlflow.registered_model_name,
            )
            registry._client.set_registered_model_alias(
                name=cfg.mlflow.registered_model_name,
                alias="champion",
                version=version.version,
            )
            print(f"Force promoted run {args.run_id} to version {version.version} (champion)")
        elif args.test_auc is not None:
            from loan_risk.exceptions import ModelPromotionError
            try:
                mv = registry.promote_if_passes_gate(
                    run_id=args.run_id,
                    test_auc=args.test_auc,
                )
                if mv:
                    print(f"Promoted to version {mv.version} (champion). AUC: {args.test_auc:.4f}")
            except ModelPromotionError as exc:
                print(f"Promotion rejected: {exc}")
                return 1
        else:
            print("Provide either --test-auc or --skip-gate with --run-id")
            return 1

    if args.version and args.set_alias:
        registry._client.set_registered_model_alias(
            name=cfg.mlflow.registered_model_name,
            alias=args.set_alias,
            version=args.version,
        )
        print(f"Set alias '{args.set_alias}' on version {args.version}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
