# CapLife AI — End-to-end AEC RUL **research demonstrator**

A browser app that accepts capacitor measurement histories, checks input quality,
and displays measured health indicators and downloadable assessments. The public
benchmarks evaluate threshold forecasts on real experiments. A separate synthetic
demonstration illustrates a PyTorch GRU and a remaining-life estimate.

> **Research release.** Uploads return measurement assessments and observed
> threshold status. Real-device RUL and its uncertainty still need validation.

Open [CapLife AI](https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app/) to
use the app in a browser. Select **Experimental measurements** to upload one
or more capacitors in a CSV. Use **Check an observed capacitance-loss threshold**
to set a loss threshold, inspect the first recorded crossing window, and export
device status. The general CSV uses an hour-zero baseline when present. The
reference-age CSV uses its first recorded age as the baseline.

## 1 — Try the web app locally (Windows / macOS / Linux)

1. Install Python 3.11 or 3.12 (Python 3.13 may work depending on installed packages).
2. Extract this ZIP, then open a terminal inside the `CapLife_AI_Starter` folder.
3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Launch the interface:

   ```bash
   streamlit run app.py
   ```

6. Your browser should open the application, usually at `http://localhost:8501`.
   With no upload, it shows a pre-generated synthetic example automatically.

The bundled `artifacts/gru_demo_state.pt` and `artifacts/metadata.json` contain
trained model weights and preprocessing statistics. You **do not need to train**
the example model to open the website.

### Public-data forecast benchmark

Choose **Public-data forecast benchmark** in the sidebar to inspect the results
and download the device-level forecasts. The inputs are the six devices in
[NASA EOS-2](https://data.nasa.gov/dataset/capacitor-electrical-stress-2) and the
eight devices in the
[original PoFINN measurement table](https://github.com/Anurag-Dutta/PoFINN/blob/6fe0fa7dd3df31b2dd62bd148182cb85f19494c7/DATA/data_combined.csv)
(MIT, copyright 2025 The Authors).

The endpoint is the first recorded reading at or above 20% capacitance loss
relative to the initial reading. A crossing is bracketed by the previous and
current inspections. One EOS-2 device finishes below the threshold and is
right-censored at its final inspection. The benchmark uses a shared forecast
origin of 94 aging hours. Each fit uses one source and is tested on every
device in the other source, with test inputs limited to measurements through
94 hours. The benchmark uses original inspections and retains each study's
recorded aging clock.

**How results change with forecast age** compares six pairs of nearby original
inspections. Each study keeps its actual age in a pair. The forecast cohort
at each age consists of devices still below the threshold, based on their
measurements through that inspection. The same devices can contribute at
multiple ages. The sensitivity download includes the number of observed
crossings and censored devices, point errors, and prediction-band metrics.

**When should a forecast be withheld?** compares showing all calibrated-trend
forecasts with two screening policies across the six inspection pairs. The
first requires increasing loss across the last three measurements and a band
whose upper edge fits within the elapsed time between those measurements.
The second also requires current loss to lie inside the training study's loss
range at the paired inspection. Both decisions use data available at the origin.
The section shows the number of forecasts retained, the error on their observed
crossings, and the number of bands that overlap or fully contain an observed
window. The benchmark reuses devices across ages, and the two existing studies
provide an exploratory screening check. Uploads receive measured-health results
while forecast performance and uncertainty await independent validation.

- **Age only** fits a log-normal distribution to the training study's remaining
  times at 94 hours. Its median is the same for every test device.
- **Recent trend** fits a straight line to the last three available losses and
  uses its slope to extrapolate from the latest observation to the threshold.
- **Recent trend with study calibration** fits a single multiplicative correction
  and a log-normal spread around the trend forecast. Both log-normal models use
  the likelihood of the observed windows and right-censored follow-up, with a
  log-scale bound of 0.15 to 3 to limit extreme fits in these small groups.

Point error is the distance to the observed window. For the nominal 80% bands,
overlap means the true crossing *could* be inside the band and full containment
means the entire observed crossing window fits. The band width matters as well
as the count. These are experimental threshold forecasts. The two protocols
support neither validated physical-failure forecasts nor validated uncertainty on
new devices. Temperature, voltage and ripple effects require comparable
operating-condition records across groups.

With the project's Python dependencies installed, regenerate the results from
the public source URLs or pass downloaded files via `--eos2` and `--pofinn`.
The output records the SHA-256 of the source measurements for reproducibility.

```bash
python -m caplife.public_benchmark
```

### Public-data neural experiment

The neural experiment estimates 48-hour crossing probability for a 20% loss
threshold. It uses current capacitance loss and the loss rates over the previous
two inspection intervals. The evaluation holds out the entire other study in
both directions. Fits and Brier scores give equal weight to each capacitor.
A horizon is unresolved if it falls inside an inspection bracket or beyond the
available follow-up. The published artifact includes all origins, their status,
held-out predictions, flags for inputs outside training ranges, and inspection traces
normalized to initial capacitance.

The four-unit neural network and logistic model use identical features and L2
regularization. The fixed baselines are training event frequency and a three-point
linear trend. The neural result averages three fixed initializations, whose spread
illustrates fit sensitivity. The simple trend scored better in both transfers.
NASA ES10 is an additional false-alert check using the audited session export,
with repeated reference ages represented by median capacitance. Its recorded
trajectories stay below the study threshold.

```bash
python -m caplife.neural_benchmark
python -m caplife.neural_benchmark --es10-sessions /path/to/nasa_es10_sessions_with_reference_age.csv
```

## 2 — Input CSV contract (important)

See `sample_data/upload_template.csv`. Every row is one timestamp/measurement.
Every capacitor must have at least **six unique measurement hours**.

Required columns:

| Column | Description |
| --- | --- |
| `Cap_ID` | Unique capacitor identity, repeated for its measurements |
| `Hours` | Elapsed operating hours (not a date string) |
| `Ambient_C` | Operating ambient temperature, degrees C |
| `Core_C` | Estimated/measured core temperature, degrees C |
| `Voltage_ratio` | Applied voltage / rated voltage |
| `Ripple_A` | RMS ripple current, amperes |
| `Ripple_Hz` | Ripple frequency, Hz |
| `C_ratio` | Capacitance divided by *initial* capacitance |
| `ESR_ratio` | ESR divided by *initial* ESR, measured at consistent frequency/temperature |
| `Leakage_ratio` | Leakage current divided by an initial reference, under controlled test conditions |
| `Case_Strain_Index` | Hypothetical nonnegative strain indicator, not hydrogen volume |

The app rejects missing columns, non-numeric measurements, invalid values,
duplicate hours per capacitor, and histories with fewer than six points. It will
warn if a prediction uses inputs outside the numerical range of the simulated
training data. **Matching column names does not imply an uploaded real dataset
is within the validated application domain.** It is not validated at all yet.

Changing measurement frequency, temperature, voltage, or method may change
observed C/ESR/leakage without corresponding aging. A future model must explicitly
account for measurement protocols or normalize readings appropriately.

## 3 — Retrain the demo model (optional)

From the project folder:

```bash
python -m caplife.train --root . --caps 180 --epochs 25
```

This replaces the *demo* artifacts and example dataset. Training, validation and
final test are split **by capacitor ID**, not randomly by measurement row.
Scaler statistics are fit on the training capacitors alone. Early stopping uses
validation capacitors, leaving test capacitors out of model selection.

Saved artifacts:
- `artifacts/gru_demo_state.pt`: neural network state dictionary.
- `artifacts/metadata.json`: feature order, scaler, range checks, test metrics,
  data provenance, and schema.
- `sample_data/demo_capacitors.csv`: generated synthetic example histories.
- `sample_data/upload_template.csv`: six-row CSV input example.

## 4 — How it works

`app.py` -> `caplife/data.py` -> `caplife/predict.py` -> `caplife/model.py`.

- The **frontend** is the Streamlit browser app.
- The **preprocessor** validates columns, orders measurements by time, and
  standardizes data using **training-only** statistics.
- The **neural network** is a small gated recurrent unit (GRU), taking six
  successive measurements, each with ten explanatory inputs.
- The **RUL output** is a continuous number of remaining hours.
- An **illustrative EOL rule** overrides the prediction with zero at its toy
  threshold (C <= 0.8 initial, ESR >= 2x, leakage >= 5x, strain >= 1.0).
- The **results view** plots individual health indicators and allows CSV export.

This is a time-series neural-network model, not an LSTM. Later work can compare
GRU, LSTM, and physics-aware hybrid methods on the same validated real dataset.

## 5 — Share the website through Streamlit Community Cloud

1. Create a GitHub repository containing the project folder's **contents**,
   including `app.py`, `requirements.txt`, `caplife/`, `artifacts/`, and
   `sample_data/`. Avoid uploading confidential experiment data.
2. Open https://share.streamlit.io and link the repository.
3. Select entry point `app.py`, choose compatible Python version, and deploy.
4. Share the resulting `*.streamlit.app` address with others.

This ZIP does not deploy itself. Uploaded data are processed on the remote host;
review data ownership, storage, provider policies, and security before asking
external researchers to upload sensitive measurements. Streamlit Community
Cloud is suitable for a demonstrator, not automatically a secure production
scientific data service.

## 6 — What must be done before scientific use?

1. Obtain **real** accelerated life test (ALT) data with calibration records,
   device type/batch metadata, measured environmental/operating history, and
   actual end-of-life events or explicitly censored trajectories.
2. Define failure modes and application-specific EOL criteria, consistent with
   manufacturer specifications and intended operation.
3. Support different instrument protocols; ensure C/ESR/leakage comparisons
   are made at comparable conditions or corrected using validated models.
4. Use grouped and external validation by capacitor, batch, technology and
   stress regime. Evaluate generalization under changing/future conditions.
5. Handle censored lifetimes explicitly. The public trend benchmark uses
   interval and right censoring. The fixed-horizon experiment scores outcomes
   only when the inspections resolve them.
6. Quantify and calibrate uncertainty; detect missing inputs, distribution
   shifts, and unsupported conditions and **abstain** when necessary.
7. Conduct scientific review before removing the demonstrator warning.

### Model limitations today

- The GRU demonstration uses synthetic data. Real-data neural experiments use
  capacitance only and need independent validation on comparable cohorts.
- Six-timestep GRU estimates RUL; it does not separately forecast future C/ESR.
- Experimental ranges have limited transfer coverage. A validated prediction
  interval and health score require independent evidence.
- No true support for arbitrary missing-column dataset formats.
- Changing future stress is not modeled through user-supplied scenarios yet.
- No database, authentication, production privacy controls, or server-side job queue.
- The simulated 'strain index' is **not** a calibrated measurement of internal H2.

## 7 — File map

```text
CapLife_AI_Starter/
  app.py                         Streamlit interface
  caplife/config.py              Canonical feature schema
  caplife/simulator.py           Entirely invented teaching simulator
  caplife/data.py                CSV validation, six-step windows, standardization
  caplife/model.py               PyTorch GRU neural network
  caplife/train.py               Training and held-out testing
  caplife/predict.py             Reusable loaded prediction API
  artifacts/gru_demo_state.pt    Pretrained demo weights
  artifacts/metadata.json       Preprocessing, provenance, metrics
  sample_data/demo_capacitors.csv
  sample_data/upload_template.csv
  tests/test_pipeline.py
  requirements.txt
```
