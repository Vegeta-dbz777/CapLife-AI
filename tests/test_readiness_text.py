import pandas as pd

from caplife.readiness import audit_measurement_readiness, grouped_actions
from caplife.measurement_review import read_experimental_csv
from caplife.real_data import assess_real_measurements


def _audit():
    rows = ["Cap_ID,Hours,Capacitance_uF,ESR_ohm,Leakage_uA,H2_Generated_mL_STP"]
    for cap in ("A", "B", "C"):
        for k in range(8):
            esr = "" if (cap == "B" and k in (3, 5)) else f"{0.1 + 0.01 * k}"
            rows.append(f"{cap},{500 * k},{1000 - 10 * k},{esr},{10 + k},{0.05 * k}")
    rows += ["D,1000,990,0.11,11,0.05", "D,2000,980,0.12,12,0.1", "D,3000,970,0.13,13,0.15"]
    report = assess_real_measurements(read_experimental_csv("\n".join(rows).encode()))
    return audit_measurement_readiness(report)


def test_capacitor_findings_name_their_own_numbers():
    audit = _audit()
    summary = audit["summary"].set_index("Cap_ID")
    assert summary.loc["A", "Data_readiness"] == "Ready" and summary.loc["A", "Limits"] == "—"
    assert summary.loc["B", "Data_readiness"] == "Limited" and "ESR: 2 of 8 missing" in summary.loc["B", "Limits"]
    assert "No 0 h reading (first at 1,000 h)" in summary.loc["D", "Limits"] and "only 3 readings" in summary.loc["D", "Limits"]
    missing = audit["actions"].query("Cap_ID == 'B' and Category == 'Missing readings'")["Recommendation"].iloc[0]
    assert "2 of 8 readings (1,500, 2,500 h)" in missing


def test_each_recommendation_is_listed_once_with_its_capacitors():
    audit = _audit()
    grouped = grouped_actions(audit["actions"], audit["summary"]["Cap_ID"])
    assert not grouped.duplicated(["Category", "Recommendation"]).any()
    protocol = grouped.loc[grouped["Category"] == "Measurement protocol", "Applies_to"]
    assert list(protocol) == ["All 4 capacitors"]
    assert grouped.loc[grouped["Category"] == "Missing readings", "Applies_to"].tolist() == ["B"]
    # Capacitor-specific findings come before file-wide documentation advice.
    assert grouped["Priority"].tolist().index("Documentation") > grouped["Category"].tolist().index("Missing readings")
    assert grouped_actions(audit["actions"].iloc[0:0], ["A"]).empty


def test_long_capacitor_lists_are_shortened():
    actions = pd.DataFrame({"Cap_ID": [f"C{i}" for i in range(8)], "Category": "Sampling gaps", "Priority": "Review",
                            "Recommendation": "Same text", "Scope": "Capacitor"})
    grouped = grouped_actions(actions, [f"C{i}" for i in range(10)])
    assert grouped["Applies_to"].tolist() == ["C0, C1, C2 + 5 more"]
    assert list(grouped.columns) == ["Applies_to", "Recommendation", "Priority", "Category"]
