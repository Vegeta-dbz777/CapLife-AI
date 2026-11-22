# CapLife AI engine validation

The available experiments support measured health assessment and research
comparisons of threshold forecasts. The neural model remains experimental.
Physical remaining useful life requires observed failure endpoints and
independent validation.

## Extended validation on original inspection records

We compared six methods with whole experiments held out. The data cover 38
devices across NASA EOS-2, PoFINN, and a NASA multivoltage experiment with
three cohorts. The three voltage cohorts share an inspection schedule and
were held out together. The primary task asks whether 20% capacitance loss
will be reached within 48 hours. The 1,503 resolved windows come from
repeated forecasts on 36 eligible devices. Two ES14 capacitors reached
the threshold before three distinct inspection ages were available.
Nine windows remain unresolved.
[NASA EOS-2](https://data.nasa.gov/dataset/capacitor-electrical-stress-2) · [PoFINN data](https://github.com/Anurag-Dutta/PoFINN/tree/6fe0fa7dd3df31b2dd62bd148182cb85f19494c7/DATA) · [NASA multivoltage archive](https://data.phmsociety.org/nasa/#12-capacitor-electrical-stress)

| Held-out experiment | Devices with forecasts | Resolved windows | Crossing windows | Persistence | Linear trend | Median-slope trend | Calibrated trend | Neural | Projection-based neural |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NASA EOS-2 | 6 | 39 | 15 | 0.369 | **0.151** | 0.175 | 0.253 | 0.304 | 0.260 |
| PoFINN | 8 | 89 | 16 | 0.191 | 0.186 | 0.309 | 0.158 | **0.158** | 0.187 |
| NASA ES10/12/14 | 22 | 1,375 | 8 | 0.058 | 0.068 | 0.072 | **0.054** | 0.153 | 0.154 |

Values are Brier scores with equal device weighting and lower scores
indicating better agreement with outcomes. On PoFINN the full-precision
scores are 0.158015 for the neural model and 0.158277 for the calibrated
trend. At a descriptive 50% alert cutoff, the PoFINN neural model missed
all 16 crossing windows. On the NASA multivoltage holdout, the same neural
design detected four of eight crossing windows and issued 190 false alerts.
The projection-based neural model detected six of eight with 354 false
alerts. In its 50–75% score bin for this holdout, 326 windows averaged
61% predicted risk and 2% observed crossings after weighting by device.

The six methods include a low-risk persistence baseline, two direct trend
extrapolations, a logistic model of projected loss, and two small neural
models. The projection-based model sees the two trend projections relative
to the threshold and the remaining distance to it. Each fit uses only
resolved training outcomes. Test inputs use only inspections through the
forecast origin. Additional checks vary the horizon (24 and 72 hours),
loss threshold (18% and 22%), repeat confirmation, capacitance measurement
method, and whether to weight the training experiments equally. Results
are sensitive to these choices. No neural model consistently improves on
the simpler methods. [All scenarios, alerts and calibration bins](../artifacts/release_validation.json)

### NASA impedance measurements

The extractor matched 8,835 impedance scans to NASA's date and reference-age
table across the three voltage cohorts. It used series capacitance nearest
100 Hz within 12%, median scans within each inspection, and median
inspections at repeated ages. The re-extracted ES10 capacitance-loss
histories reproduced the prior session audit exactly. The ES14 values
reproduced the reference-age sample.

| Cohort | Date-matched scans | Devices reaching 20% with capacitance near 100 Hz | Devices reaching 20% with low-frequency series RC fit |
| --- | ---: | ---: | ---: |
| ES10 | 2,981 | 0 of 8 | 1 of 8 |
| ES12 | 2,921 | 0 of 8 | 8 of 8 |
| ES14 | 2,933 | 6 of 8 | 8 of 8 |

The alternative capacitance comes from fitting the capacitive reactance
of a series RC model to the measured 20–2,000 Hz portion of each spectrum.
It tests method sensitivity. A measurement-dependent crossing can reverse
at a later inspection. The app now shows both the first measured crossing
window and the time of the second consecutive above-threshold reading when
available. It also shows the first subsequent reading below the threshold.
These are observed statuses for the submitted measurement protocol.

### Reproduce the extended validation

Extract `ES10.mat`, `ES12.mat` and `ES14.mat` from the
[public NASA archive](https://phm-datasets.s3.amazonaws.com/NASA/12.+Capacitor+Electrical+Stress.zip).
The extraction tool requires `h5py` in addition to the app dependencies.
The benchmark downloads the other sources or accepts local copies via
`--eos2` and `--pofinn`.

```bash
for cohort in ES10 ES12 ES14
do
  python -m caplife.nasa_eis "${cohort}.mat" "$cohort" "${cohort}.csv"
done
python -m caplife.release_validation \
  --es10-sessions ES10.csv --es12-sessions ES12.csv --es14-sessions ES14.csv
```

The results artifact records the input hashes and device-level first
crossing ages. The benchmarks evaluate an electrical degradation threshold.
The public evidence in this release does not establish physical lifetime
or calibrated uncertainty for arbitrary field uploads.

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
| Improve on simple methods across independent measurement studies | No neural model won consistently across the held-out public experiments |
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
