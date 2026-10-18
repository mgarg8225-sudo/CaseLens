# CaseLens
Privacy-first analytics to trace where violence-against-women cases stall across police, hospitals, forensic labs, prosecutors and courts.

## Features
- Probabilistic record linkage (Splink) without a common ID
- Case-journey funnel, delays and drop-offs with uncertainty ranges
- Differential privacy, small-group suppression, role-based access, audit logs
- Streamlit dashboard

## Run
pip install -r requirements.txt
python -m streamlit run app/dashboard.py

## Note
All data is synthetic. No real survivor data is used. Aggregates only.
