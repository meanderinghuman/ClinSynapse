import json

import numpy as np

from evals.gate import gate


def write_run(path, faithfulness):
    path.mkdir(parents=True)
    with (path / "results.jsonl").open("w") as f:
        for i, v in enumerate(faithfulness):
            f.write(json.dumps({"item": {"id": str(i)}, "metrics": {"faithfulness": v}}) + "\n")


def test_gate_fails_on_large_significant_drop(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.binomial(1, 0.9, 200).astype(float)
    worse = np.where(rng.random(200) < 0.25, 0.0, base)
    write_run(tmp_path / "a", base.tolist())
    write_run(tmp_path / "b", worse.tolist())
    _, failures = gate(tmp_path / "a", tmp_path / "b", {"faithfulness": 0.05})
    assert [f.metric for f in failures] == ["faithfulness"]


def test_gate_passes_on_tiny_drop(tmp_path):
    base = [1.0] * 200
    slightly = [0.0] * 4 + [1.0] * 196  # -0.02: significant-ish but under threshold
    write_run(tmp_path / "a", base)
    write_run(tmp_path / "b", slightly)
    _, failures = gate(tmp_path / "a", tmp_path / "b", {"faithfulness": 0.05})
    assert failures == []


def test_gate_passes_on_identical_runs(tmp_path):
    write_run(tmp_path / "a", [1.0, 0.0] * 50)
    write_run(tmp_path / "b", [1.0, 0.0] * 50)
    _, failures = gate(tmp_path / "a", tmp_path / "b", {"faithfulness": 0.0})
    assert failures == []
