# CapLife AI

A browser app for aluminum electrolytic capacitors with three workspaces:

- **Monitor and Predict:** upload measurement histories (capacitance, ESR, leakage current, case expansion,
  generated hydrogen, internal pressure; any of them) with optional operating conditions. See each
  parameter's health against your own end-of-life limits, a neural network's remaining useful life with an
  80% range, the predicted trend of every parameter, the effect of different operating conditions, and
  download a PDF or Excel report.
- **Train The AI:** train a neural network on your own tested capacitors, check it on capacitors it never
  saw, and forecast the rest only if it passes.
- **My Data:** the datasets you saved (sign-in required).

> **Research release.** The Monitor and Predict networks learned from simulated capacitors; they demonstrate
> the method, and their accuracy on real capacitors is untested. Not a maintenance or safety tool.

Open [CapLife AI](https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app/) in a browser.

## Monitor and Predict

Choose **Monitor and Predict** in the sidebar and upload a CSV with `Cap_ID`, `Hours`
and any of `Capacitance_uF`, `ESR_ohm`, `Leakage_uA`, `Case_Expansion_um`,
`H2_Generated_mL_STP` and `Pressure_Abs_kPa`, or load the simulated example
(optionally with only capacitance, ESR and leakage). Operating conditions are
optional and recommended: `Ambient_C`, `Core_C`, `Voltage_V` (applied),
`Rated_V`, `Ripple_A`, `Rated_Ripple_A` and `Ripple_Hz` (the operating
frequency), recorded at each reading. The page scores each measured parameter's
health against its limit, and a GRU neural network predicts remaining useful
life with a calibrated 80% range at every reading, using only the readings
available at that time.

The network reads, with every reading, core and ambient temperature, applied
over rated voltage, ripple over rated ripple current and operating frequency,
each with a "recorded" flag. The latest conditions are taken to continue, so
the page can show remaining life under planned conditions ("Try different
operating conditions from now on"), next to the standard aging rule (2× per
10 °C of core temperature, cube of the voltage ratio) applied to the current
prediction. The page also judges the conditions: voltage above or near rating,
ripple above or near rating, high temperature, and conditions outside the
training range.

The network was trained with parameters and conditions hidden at random, and
each parameter carries a "measured" flag, so one model predicts from any
combination. The 80% range is calibrated separately for each of the 63
parameter combinations, on capacitors scored both with and without conditions,
so it widens when fewer parameters are available, and the page reports the test
accuracy for the combination in use.

The network was trained on 2,400 capacitors from a physics-inspired simulator
(`caplife/six_sim.py`): electrolyte loss, which speeds up 2× for every 10 °C of
core temperature and with the cube of applied over rated voltage, lowers
capacitance and raises ESR; ripple current heats the core (less at higher
operating frequencies); half of the capacitors change operating conditions
part-way through life; leakage current generates hydrogen by Faraday's law, and
retained hydrogen raises internal pressure and bulges the case. End of life is
the first parameter to reach its limit. Users set the six limits for each
capacitor (a series' own tolerance); the defaults are capacitance loss of 20%,
ESR at 2× initial, leakage at 5× initial, case expansion of 52 µm, 6 mL of
generated H₂ and internal pressure at 250 kPa. Each training capacitor had
limits drawn at random from the ranges the app allows (30% kept the defaults),
and every limit is an input to the network, so it predicts the time to the
limits a user sets. On 400 further simulated capacitors with the default limits:

| Parameters | Typical error, with conditions | Without conditions | True RUL inside 80% range | Trend line |
|---|---|---|---|---|
| All six | 1,340 h (15%) | 1,411 h (16%) | 83% | 3,348 h |
| Capacitance, ESR, leakage | 2,269 h (22%) | 2,403 h (23%) | 83% | 6,677 h |
| Capacitance, ESR | 2,442 h (23%) | 2,516 h (24%) | 83% | 7,270 h |
| Capacitance only | 3,048 h (29%) | 3,102 h (30%) | 78% | 9,874 h |

Conditions help most early in a history (first three predictions per capacitor:
19% vs 23% typical error) and under steady conditions (11% vs 13%). For
capacitors whose conditions changed later in life, readings before the change
cannot foresee it, so accuracy is the same either way (18%). Under a planned
change, entered on the page, the network gets the
direction right but understates large changes (about two thirds of the effect,
typical error 32% on simulated capacitors); the standard aging rule had 20% on
the same capacitors, which follow that rule by construction.

On 400 more with random limits (half of them inspected again in the last 3% of
their life), the typical error with all six is 833 h and the 80% range held the
true value 77–84% of the time across all 63 combinations. Within 5% of a limit
the range is calibrated separately (Mondrian conformal prediction): it held the
true value 82% of the time, with a typical error of 260 h, and the page says
when a parameter is that close. A looser limit never shortens the predicted
remaining life.

A second network forecasts each parameter's path from the latest reading to the
predicted end of life, at 10% steps of the remaining life, as a median that never
improves and a calibrated 80% range. On the test capacitors its median is
typically within 2–3% of the limit halfway there and 4–7% at end of life, and
the range held the true path 80–82% of the time. The trend charts start at the
first reading: measured readings, then the forecast.

Aging damage is treated as lasting: health and end of life use the worst level a
parameter has held (three consecutive readings), and the network sees earlier
readings cleaned with their neighbours, so one faulty reading affects only its own
prediction. Status: Good above 70% health, Watch from 40%, Critical below.
These figures hold for the simulator only: the coefficients are illustrative,
and accuracy on real capacitors is unknown until the model is tested on
measured failures. Retrain with `python -m caplife.six_rul --output artifacts/six_rul`.

**Reports.** Under Downloads, *Report (PDF)* and *Report (Excel)* cover the fleet (health, status, remaining
life and its 80% range), the selected capacitor (limits, operating conditions as recorded and as used,
parameter health, history and forecast charts, remaining life over time) and a data check (missing readings,
no 0 h reading, sampling gaps, changing ESR test settings). The Excel workbook adds every table in full. Both
are built when the button is clicked (`caplife/monitor_report.py`).

## Train The AI

Choose **Train The AI** in the sidebar. Upload a measurement history (any
of the six indicators; the model uses the ones every capacitor with an outcome
has measured at least three times) and an outcome file with one row per tested capacitor (`Cap_ID`,
`Experiment_ID`, `Endpoint`, `Last_alive_h`, `Failure_by_h`; leave
`Failure_by_h` blank for capacitors still working when testing stopped).
Capacitors with readings but no outcome row are the ones to forecast.

The model learns from every reading from a capacitor's third onward (changes
since the first reading, recent rates and elapsed time), with each capacitor
counting equally; calibration and testing use one reading per capacitor, chosen
from its ID and readings, never its outcome. Forecasts are made at each
capacitor's latest reading, with a chart of how the forecast developed.

The page splits capacitors into training, calibration and test groups, by
experiment when there are at least three, otherwise by capacitor ID, never by
outcome. A small neural network learns from the training group, the
calibration group sets the width of an 80% range, and the test group scores it
alongside a log-linear model and an average-lifetime guess. Forecasts appear
only when the neural network passes every check on the test group:

- at least 3 observed test failures;
- its 80% range contains at least 70% of them;
- its average miss is no worse than predicting the average lifetime;
- at most 20% of still-working test capacitors get a range ending before their last check.

Forecasts use the tested model and range unchanged, and are withheld for
capacitors whose early readings lie more than 10% of the training range beyond
the training capacitors. A synthetic example shows the workflow; its results
say nothing about real capacitors. The logic lives in `caplife/rul_workflow.py`.

## My Data: accounts and saved datasets (optional)

When sign-in and a database are configured in Streamlit secrets, visitors can
sign in with Google, save uploads to their account, reopen them on *Monitor and
Predict* and *Train The AI*, and manage or delete them on **My Data**.
Each user sees only their own datasets. Without these settings the app is a
public demo and stores nothing. Setup takes about 20 minutes:
[docs/accounts-and-storage.md](docs/accounts-and-storage.md).

## Measurement files

*Monitor and Predict* and *Train The AI* read CSV and
Excel (.xlsx) files: comma, semicolon or tab separated, with decimal points or
the decimal commas European Excel writes, in UTF-8 or Windows encoding. Common
column-name variations (`cap_id`, `Time (h)`, `Capacitance (uF)`, `ESR`,
`Pressure`...) are recognised, `n/a` and `-` count as missing, and rows without
a capacitor ID or time, with negative time, or repeating a reading are left out
with a note rather than rejecting the file. Time is read in hours unless the
column name says otherwise (a `Days` or `Weeks` column converts automatically),
and pressure is taken as absolute unless a capacitor's first reading is below
80 kPa, which is read as gauge pressure and converted. The page lists every
adjustment; no reading is invented.

## Run the app locally (Windows / macOS / Linux)

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
   Open **Monitor and Predict** and choose the simulated example capacitors.

The trained models are bundled in `artifacts/six_rul/`; you do not need to train
anything to open the app.

## Share the app through Streamlit Community Cloud

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

## Before scientific use

1. Obtain **real** accelerated life test (ALT) data with calibration records,
   device type/batch metadata, measured environmental/operating history, and
   actual end-of-life events or explicitly censored trajectories.
2. Define failure modes and application-specific EOL criteria, consistent with
   manufacturer specifications and intended operation.
3. Support different instrument protocols; ensure C/ESR/leakage comparisons
   are made at comparable conditions or corrected using validated models.
4. Use grouped and external validation by capacitor, batch, technology and
   stress regime. Evaluate generalization under changing/future conditions.
5. Handle censored lifetimes explicitly (Train The AI accepts right-censored
   capacitors that were still working when testing stopped).
6. Quantify and calibrate uncertainty; detect missing inputs, distribution
   shifts, and unsupported conditions and **abstain** when necessary.
7. Conduct scientific review before removing the demonstrator warning.

## Run the tests

Install the app and test dependencies, then run the suite from the repository root.
`requirements.txt` stays limited to what Streamlit Community Cloud needs.

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub runs the same tests on every pull request and every change to `main`
(`.github/workflows/tests.yml`), including the storage tests against PostgreSQL.

## File map

```text
app.py                         Streamlit interface: Home and the three workspaces
caplife/home_page.py           Home page
caplife/six_rul_page.py        Monitor and Predict page
caplife/six_rul.py             Six-parameter GRU models (remaining life and parameter trends)
caplife/six_sim.py             Physics-inspired six-parameter capacitor simulator (training data)
caplife/operating_conditions.py  Operating conditions: judgements, planned changes, estimated history
caplife/monitor_report.py      PDF and Excel reports for Monitor and Predict
caplife/chart_hover.py         Chart hover helpers
caplife/rul_page.py            Train The AI page
caplife/rul_workflow.py        Train, test and forecast remaining life on uploaded outcomes
caplife/accounts.py            Optional sign-in, save controls and My Data page
caplife/storage.py             Private per-user dataset storage (SQL)
caplife/file_reader.py         CSV / Excel reading with header aliases and unit handling
artifacts/six_rul/             Trained models and their test metrics
tests/                         Automated tests
requirements.txt               Deployed app dependencies
requirements-dev.txt           Adds test dependencies
```

Research code from earlier versions (public-data benchmarks, the first GRU demonstration) remains in
`caplife/` and `artifacts/` for reference; the app no longer shows it.
