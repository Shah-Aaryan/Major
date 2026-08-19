"""
Unit tests for the 15-optimizer benchmark harness.
"""

import pytest
import os
import pandas as pd
from analysis.optimizer_benchmark_harness import OptimizerBenchmarkHarness


def test_benchmark_harness_runs(tmp_path):
    output_dir = str(tmp_path / "bench_test")
    harness = OptimizerBenchmarkHarness(output_dir=output_dir)
    
    df_res = harness.run_benchmark(n_iterations=5, n_seeds=2, objective_name="SphereTest")
    
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == 15
    assert 'rank' in df_res.columns
    assert os.path.exists(os.path.join(output_dir, "optimizer_benchmark_spheretest.csv"))
    assert os.path.exists(os.path.join(output_dir, "optimizer_benchmark_spheretest.json"))
    assert os.path.exists(os.path.join(output_dir, "optimizer_benchmark_spheretest.md"))
