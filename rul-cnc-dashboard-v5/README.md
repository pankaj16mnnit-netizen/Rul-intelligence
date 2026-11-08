# CNC RUL Intelligence Dashboard

Two-page Streamlit dashboard for the M.Tech AI/ML project on remaining useful life (RUL) of a CNC machining centre.

- **Page 1, Model comparison:** five models on 20 held-out machines. The leader depends on the metric: LSTM leads R², RMSE and MAE; CNN-LSTM leads the asymmetric score, which penalises late predictions.
- **Page 2, CNN-LSTM prediction:** upload sensor data (CSV or Excel) and get RUL for each machine, with status, gauge and trajectory.

## Structure
    app.py                    dashboard (both pages)
    src/pipeline.py           preprocessing, identical to training
    src/inference.py          model loading and prediction
    src/metrics.py            metrics, ranking and status logic
    src/ui.py                 theme and components
    models/                   copied from the Colab export
    results/                  model_comparison.csv, test_predictions.csv
    data/                     sample uploads
    notebook/                 RUL_CNC_v4_export.ipynb
    docs/                     final report
    DEPLOY.md                 deployment steps

## Run
    pip install -r requirements.txt
    streamlit run app.py
