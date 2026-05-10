# Healthcare Patient Clustering & Medical Risk Grouping

Unsupervised clustering of clinical data (UCI CKD) with a Streamlit decision-support dashboard.
AIE323 — Data Mining, Alamein University.

## Setup (Windows / PowerShell)

```powershell
cd D:\healthcare
py -3.11 -m venv .venv     # or `python -m venv .venv` if `python` is on PATH
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt   # full dev env (notebooks + EDA)
# or `pip install -r requirements.txt` for the slim dashboard-only env
```

## Run

- Notebooks: `jupyter lab notebooks/`
- Dashboard: `streamlit run app/streamlit_app.py`

See `docs/specs/` for the design spec and `docs/plans/` for milestone plans.
