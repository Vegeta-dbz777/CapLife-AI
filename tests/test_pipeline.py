import numpy as np
import pandas as pd
import pytest
from caplife.simulator import simulate_capacitors
from caplife.data import DataValidationError, validate_measurements, windowize
from caplife.predict import load_engine, assess_capacitor


def test_generator_and_window():
    df = simulate_capacitors(n_caps=5, seed=7)
    clean = validate_measurements(df)
    X, y, cap_ids = windowize(clean)
    assert X.shape[1:] == (6, 10)
    assert len(X) == len(y) == len(cap_ids)
    assert np.isfinite(X).all()
    assert (y >= 0).all()


def test_duplicate_rejected():
    df = simulate_capacitors(n_caps=1)
    duplicated = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_measurements(duplicated)


def test_missing_feature_rejected():
    df = simulate_capacitors(n_caps=1).drop(columns=["ESR_ratio"])
    with pytest.raises(DataValidationError, match="Missing required"):
        validate_measurements(df)


def test_saved_model_round_trip():
    df = simulate_capacitors(n_caps=1, seed=77)
    net, metadata = load_engine("artifacts")
    assessment = assess_capacitor(df, net, metadata)
    assert np.isfinite(assessment["rul_hours"])
    assert assessment["rul_hours"] >= 0
    assert assessment["data_provenance"]
