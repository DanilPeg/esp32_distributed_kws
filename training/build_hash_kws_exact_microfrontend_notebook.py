from __future__ import annotations

import json
from pathlib import Path

from build_hash_kws_experiments_notebook import build_notebook, build_runtime_payloads


def main() -> None:
    root = Path(__file__).resolve().parent
    notebook_path = root / "hash_kws_exact_microfrontend_esp32.ipynb"
    notebook_path.write_text(
        json.dumps(
            build_notebook(
                build_runtime_payloads(root.parents[1]),
                notebook_title="Hash KWS Exact Microfrontend Lab",
                notebook_description=(
                    "This notebook mirrors the hash KWS branch, but trains directly on the "
                    "same exact TensorFlow microfrontend semantics that the current ESP32 firmware uses."
                ),
                notebook_goals=[
                    "keep the analytic-hash student branch deploy-oriented from the start;",
                    "match the current firmware microfrontend instead of the log-mel fallback;",
                    "materialize a reusable exact-feature cache so the GPU stops waiting on per-sample CPU frontend work;",
                    "preserve direct export into the custom ESP32 hash runtime without changing the model format;",
                    "measure how much accuracy survives after frontend alignment.",
                ],
                pip_comment=(
                    '# !pip -q install -r '
                    '"code/training/requirements-kws-hash-exact-frontend.txt"'
                ),
                selected_recipe="hash_deeper_fair_3block_big_pointwise_exact_microfrontend",
                next_iteration_ideas=[
                    "Use `hash_deeper_fair_ce_exact_microfrontend_tuned` as the new deploy-aligned CE reference.",
                    "Start with `hash_deeper_fair_3block_big_pointwise_exact_microfrontend` as the default bigger-codebook, lower-MAC probe.",
                    "Then compare against `hash_deeper_fair_balanced_budget_exact_microfrontend` to test a milder fixed-budget reallocation without changing depth.",
                    "Then re-check `hash_deeper_fair_residual_exact_microfrontend` and `hash_deeper_fair_residual_specaug_exact_microfrontend` with the longer 60-epoch schedule.",
                    "Keep `hash_deeper_fair_pointwise_budget_exact_microfrontend` as the more aggressive pointwise-heavy budget probe.",
                    "Keep `hash_deeper_fair_signed_cached_exact_microfrontend` as the no-teacher signed-hash probe.",
                    "Treat the resulting bundle as the first candidate for real ESP32 accuracy checks.",
                ],
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(notebook_path)


if __name__ == "__main__":
    main()
