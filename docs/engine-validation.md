# CapLife AI engine validation

The available experiments support measured health assessment and research
comparisons of threshold forecasts. The neural model remains experimental.
Across the study transfers, a three-point capacitance trend had lower forecast
error than either neural feature set. Physical remaining useful life requires
observed failure endpoints and independent validation.

## Does ESR improve a 48-hour warning?

Both neural variants use the same three historical observations, test origins,
folds, and outcomes. The capacitance-only inputs are current loss and two
recent loss rates. The augmented variant adds two recent ESR log-change rates
per hour to keep the feature independent of the ESR measurement units. A
prior EOS-2 run that also included the absolute normalized ESR level gave a
slightly higher Brier score than capacitance alone. The feature representation
matters even within that study.

Table values are Brier scores, with lower values indicating better predictions.
Each capacitor contributes equally. Resolved origins are repeated forecast
windows from the same devices.

| Test | Loss threshold | Capacitors | Resolved origins | Trend | Neural capacitance | Neural capacitance + ESR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EOS-2, leave one device out | 20% | 6 | 39 | 0.151 | 0.024 | 0.020 |
| EOS-2, leave one device out | 12% | 6 | 36 | 0.167 | 0.007 | 0.005 |
| Jian figures, leave one device out | 12% | 8 | 80 | 0.131 | 0.069 | 0.067 |
| Train on EOS-2, test on Jian figures | 12% | 8 | 80 | 0.131 | 0.176 | 0.392 |
| Train on Jian figures, test on EOS-2 | 12% | 6 | 36 | 0.167 | 0.346 | 0.292 |

- **Alerts.** In the 20% EOS-2 test, both variants warned on 14 of 15 crossing
  windows and issued zero false alerts at a 50% cutoff. In the EOS-2 to Jian
  transfer, adding ESR raised false alerts from 13 to 44 and detected 8 of 9
  crossing windows, up from 7.
- **Feature coverage.** In the EOS-2 to Jian transfer, 60 of 80 resolved
  origins fell outside training ranges on at least one augmented input.
  On the same 20 in-range origins, ESR increased the Brier score from 0.257
  to 0.307. All 36 resolved origins in the reverse transfer had at least
  one input outside the training ranges.
- **Probability calibration.** In the EOS-2 to Jian transfer, the 40 windows
  in the ESR model's 75–100% score bin averaged 92% predicted risk and 27%
  observed crossings after weighting by device. In the reverse transfer,
  the 29 windows in its 0–25% bin averaged 2% predicted risk and 38%
  observed crossings. These bins are descriptive. There are only six to
  eight test devices per transfer.

## Measurement and outcome limits

- NASA EOS-2 supplied original ESR increase and capacitance-loss measurements.
  The Jian data came from digitizing published capacitance and ESR figures
  at 20-hour intervals from 560 to 960 hours. Occluded readings were left
  missing. The common 12% threshold comes from Jian's experiment and serves
  as an exploratory transfer check. [NASA EOS-2](https://data.nasa.gov/dataset/capacitor-electrical-stress-2) · [Jian et al., Figure 4 and §4.2](https://doi.org/10.3390/electronics14224452)
- Jian crossing times are derived from the plotted threshold. Two crossing
  windows inferred from Figure 4(a) differ from the ages implied by the
  remaining-life examples in §4.2. A ±0.10 percentage-point margin accounts
  for reading the plot. The transfer conclusions held with ±0.05 and ±0.15.
  Unresolved crossing windows are counted separately and excluded from
  scoring. [Jian et al., Figure 4 and §4.2](https://doi.org/10.3390/electronics14224452)
- The original 20% cross-study test used raw NASA EOS-2 and PoFINN capacitance
  histories. The trend outscored the capacitance-only neural model in both
  directions. The PoFINN export used here has no paired ESR observations.
  The additional NASA ES10 check uses below-threshold histories to measure
  false alerts. [20% benchmark results](../artifacts/neural_benchmark.json)

## Readiness of the real-data workflow

| Check | Result |
| --- | --- |
| Summarize measured degradation and bracket observed threshold crossings | Implemented in the real-data workflow and covered by regression tests |
| Improve on the trend across independent measurement studies | Trend had lower error in both paired ESR transfers and both original 20% transfers |
| Show reliable threshold probabilities across studies | Observed frequencies differ materially from the neural scores |
| Estimate physical RUL with calibrated uncertainty | Needs independent, comparable raw histories with documented failures or censoring, and coverage of the intended operating conditions |

## Reproduce

From the repository root with the project's Python dependencies installed.
The command retrieves the public sources and writes their SHA-256 hashes,
per-device changes in Brier score, alert counts, calibration bins, and
margin sensitivity to the output JSON. Local copies can be supplied through
`--eos2`, `--figure-a`, and `--figure-b`.

```bash
python -m caplife.esr_validation --output artifacts/esr_validation.json
python -m pytest -q
```

[Machine-readable results](../artifacts/esr_validation.json)
