"""Independent digitized threshold transfer with the release's paired-origin labels."""
from pathlib import Path
import io
import json
import sys

import pandas as pd
from scipy.io import loadmat

from caplife.esr_validation import digitize_figure, paired_examples
from caplife.neural_benchmark import evaluate_transfer
from caplife.public_benchmark import load_eos2

from caplife.constrained_validation import robust_origin_features, CANDIDATES, score_predictions, predict_candidate


def origins(cap, esr, margin):
    reference = paired_examples(cap, esr, 12, margin)
    data = robust_origin_features(cap, "probe", 48, 12, False)
    data["device_key"] = data["device"].str.replace("probe / ", "", regex=False)
    data = reference.merge(
        data[["device_key", "age_h", "relative_headroom", "projected_margin", "sustained", "tempered"]],
        left_on=["device", "origin_h"], right_on=["device_key", "age_h"], how="left", validate="one_to_one",
    )
    assert data["sustained"].notna().all()
    return data


def main(data_dir, output):
    import torch
    torch.set_num_threads(1)
    eos2, mat = load_eos2((data_dir / "eos2.zip").read_bytes())
    eos2esr = pd.DataFrame(1 + loadmat(io.BytesIO(mat), simplify_cells=True)["ESR"] / 100,
                           index=eos2.index, columns=eos2.columns)
    jian = digitize_figure((data_dir / "figure4a.png").read_bytes(), True)
    jianesr = digitize_figure((data_dir / "figure4b.png").read_bytes(), False)
    data = {}
    for margin in (.05, .10, .15):
        devices = {"Jian": origins(jian, jianesr, margin), "NASA EOS-2": origins(eos2, eos2esr, 0.0)}
        tests = {}
        for target, frame in devices.items():
            other = next(v for label, v in devices.items() if label != target)
            test, train = frame.dropna(subset="outcome"), other.dropna(subset="outcome")
            raw, _ = evaluate_transfer(train, test)
            tests[target] = {
                "resolved": len(test),
                "unresolved": len(frame) - len(test),
                "positive_origins": int(test.outcome.sum()),
                "models": {
                    **{method: score_predictions(test, predict_candidate(train, test, method), 0.5)
                       for method in CANDIDATES},
                    "trend": score_predictions(test, test["trend"].to_numpy(), 0.5),
                    "neural": score_predictions(test, raw["neural"].to_numpy(), 0.5),
                },
            }
        data[str(margin)] = tests
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
