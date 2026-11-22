# CapLife AI remaining-life research

Two endpoint definitions were evaluated. The electrical benchmark asks when
capacitance loss first reaches 20% or when a second above-threshold inspection
confirms it. The Aalborg study reports open- or short-circuit failure and
separately identifies vent opening. For another product or application, an
independently documented functional or endurance limit can define a distinct
remaining-life task. [Aalborg failure study](https://doi.org/10.1109/TPEL.2023.3263125) · [TDK useful-life conditions](https://www.tdk-electronics.tdk.com/download/530704/11a9011cd3cac4ccfbc490fbd4ce88b4/pdf-generaltechnicalinformation.pdf)

## 48-hour threshold warnings

The fixed score combines three capacitance projections. A monotone neural
candidate learns a score from these projections and the distance to the
threshold. Structures and constants were set before these runs. Earlier
results on the same experiments informed the design, so the comparison
remains exploratory. [Scoring and sensitivity results](../artifacts/constrained_validation.json)

| Held-out experiment | Original trend Brier | Fixed physical score Brier | Original neural Brier | Monotone neural Brier | Monotone neural false alerts | Monotone neural missed windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NASA EOS-2 | 0.151 | 0.061 | 0.304 | 0.202 | 0 | 11 of 15 |
| PoFINN | 0.186 | 0.158 | 0.158 | 0.144 | 2 | 15 of 16 |
| NASA ES10/12/14 | 0.068 | 0.058 | 0.153 | 0.099 | 174 | 2 of 8 |

Lower Brier scores are better. Alert counts use a fixed cutoff of 0.5.
The fixed score beat the original trend in 20 of 21 test and sensitivity
comparisons. On the multivoltage holdout it warned correctly in three of eight
crossing windows and raised 50 false alerts. Selecting stricter cutoffs on
training experiments alone eliminated true warnings in two of the three
holdouts for the monotone neural model. A separate 12% loss test using
digitized Jian figures supported the fixed score and exposed 63 false alerts
from the monotone neural model when tested on the Jian sample at the central
digitization margin. [Digitized 12% sensitivity](../artifacts/constrained_jian_probe.json)

## Remaining time and uncertainty

An interval-censored log-normal model trains on the time between inspections
that bracket a crossing. For units that finish below the threshold, it trains
on survival past the last available inspection. The neural model estimates
a conditional remaining-time distribution and is compared with intercept-only
and log-linear models. The study holdouts preserve the original experiment
groups. The repeated-reading sensitivity uses the recorded confirmation age.

The 20% loss forecasts still transfer poorly. Keeping full follow-up
produced very wide prediction bands in two short-duration holdouts. A
separate short-horizon test limits training follow-up to 72 hours and reports
only crossing probabilities within that limit. The results distinguish
indeterminate horizons and report conservative Brier bounds across them.
[Time-to-threshold results](../artifacts/remaining_time_validation.json)

## Physical failures in the Aalborg study

The study reports 11 device outcomes under three accelerated stress
conditions. The analysis uses 49 paired pre-failure capacitance and ESR
markers digitized from Figures 7 and 8. The third condition has a reported
failure with insufficient measurements for a forecast. The other ten
devices contribute one forecast per device, made at the third paired
reading. Training holds out all devices from the tested condition.
Whole-hour reported terminal ages are evaluated with a ±0.5-hour rounding
interval. [Source and figures](https://vbn.aau.dk/ws/portalfiles/portal/639038831/letter_3.26.pdf)

| Held-out condition | Devices scored | Intercept-only mean distance to failure window | Log-linear mean distance | Neural mean distance | Neural nominal 80% band overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant stress | 6 | 254 h | 190 h | 192 h | 0 of 6 |
| Variable stress | 4 | 286 h | 255 h | 301 h | 0 of 4 |

The intervals use approximate readings from publication markers under one
lab's test setup. They failed on both held-out conditions. The analysis
keeps vent opening separate from catastrophic failure and preserves the
reported brackets for records without an exact age. The force-sensor
resistance plotted in the paper is a deformation proxy. It supplies neither
case expansion in micrometers nor direct internal pressure. The public
histories in this test contain no complete device records with capacitance,
ESR, electrical leakage current, case expansion, cumulative H₂ and absolute
pressure linked to adjudicated outcomes.
[Physical failure results](../artifacts/physical_failure_validation.json)

## Evidence needed to validate the six-indicator model

- Collect the six direct measurements on the same devices over time, with
  sensor calibration, product series, lot, stress schedule, and test conditions.
- Record the intended functional endpoint and the per-device failure time
  or inspection bracket. Record the last confirmed functional age and
  follow-up reason for units that finish without failure.
- Freeze model fitting and alert thresholds before evaluating an independent
  campaign. Measure missed events, false alerts and lead time by device.
  Check prediction-band coverage and width against the declared endpoint.

## Train and evaluate from measured histories

The web app's **Train RUL model** page runs this evaluation on uploaded files,
applies release checks to the neural model on the test group, and forecasts
capacitors without an outcome row only when they pass. See the README for the
checks. The command below produces the evaluation report without forecasts.

Put all measurements in a CSV using the CapLife measurement column names.
Include the device ID, age in hours, capacitance, ESR, electrical leakage
current, case expansion, cumulative H₂, and absolute internal pressure.
Capacitor IDs must identify a single physical device across the entire
dataset. Record comparable sensor conditions, including ESR frequency and
temperature where available.

```csv
Cap_ID,Hours,Capacitance_uF,ESR_ohm,Leakage_uA,Case_Expansion_um,H2_Generated_mL_STP,Pressure_Abs_kPa
```

Use a separate outcome CSV with this header.

```csv
Cap_ID,Experiment_ID,Endpoint,Last_alive_h,Failure_by_h
```

Use one row per capacitor and one endpoint definition across the dataset.
For an exact event, put its age in both time columns. For an event observed
between inspections, record the last functional age and the age by which
the event was observed. Leave `Failure_by_h` blank when follow-up ended
while the device was still functional.

```bash
python -m caplife.six_indicator_validation --measurements measurements.csv \
  --outcomes outcomes.csv --calibration-experiment CALIBRATION \
  --test-experiment FINAL_HOLDOUT --output six_indicator_evaluation.json
```

The remaining experiments train the models. Each device contributes one
forecast, at its third complete reading starting from a complete zero-hour
baseline. The report includes excluded devices and compares the neural model
with intercept-only and log-linear baselines. It scores prediction bands on
the final experiment, including conservative calibration against the
documented outcome bounds. If the calibration sample cannot support a
finite band, its calibrated result is null. Generalizing coverage to any
new operating conditions still requires suitable independent evidence.

## Reproduce

Use the public EOS-2 archive, the original PoFINN CSV, and the NASA ES10,
ES12 and ES14 session exports described in the
[engine validation report](engine-validation.md). Save the original EOS-2
archive as `eos2.zip` and the generated session tables as `ES10.csv`,
`ES12.csv` and `ES14.csv` in the chosen data directory. Save the
[Aalborg manuscript](https://vbn.aau.dk/ws/portalfiles/portal/639038831/letter_3.26.pdf)
as `yao.pdf`. The figure extractor uses Poppler's `pdfimages` command.
For the 12% transfer check, save the [capacitance figure](https://mdpi-res.com/electronics/electronics-14-04452/article_deploy/html/images/electronics-14-04452-g004a.png)
and [ESR figure](https://mdpi-res.com/electronics/electronics-14-04452/article_deploy/html/images/electronics-14-04452-g004b.png)
as `figure4a.png` and `figure4b.png` in the same data directory.

```bash
python -m caplife.constrained_validation --data-dir data --pofinn data_combined.csv \
  --output artifacts/constrained_validation.json
python -m caplife.constrained_jian_probe data artifacts/constrained_jian_probe.json
python -m caplife.remaining_time_validation --eos2 data/eos2.zip --pofinn data_combined.csv \
  --es10-sessions data/ES10.csv --es12-sessions data/ES12.csv --es14-sessions data/ES14.csv
python -m caplife.yao_figures yao.pdf data
python -m caplife.physical_failure_validation --markers data/yao_digitized_electrical.csv \
  --endpoints data/yao_physical_endpoints.csv
```
