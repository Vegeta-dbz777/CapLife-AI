"""Documented dataset contract for all parts of the prototype."""
FEATURES = [
    "Hours", "Ambient_C", "Core_C", "Voltage_ratio", "Ripple_A",
    "Ripple_Hz", "C_ratio", "ESR_ratio", "Leakage_ratio", "Case_Strain_Index"
]
SEQ_LEN = 6
RUL_SCALE_HOURS = 1000.0
# Illustrative teaching thresholds only, NOT universal AEC standards.
EOL_THRESHOLDS = {
    "C_ratio_min": 0.80,
    "ESR_ratio_max": 2.0,
    "Leakage_ratio_max": 5.0,
    "Case_Strain_Index_max": 1.0,
}
