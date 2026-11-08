# Deployment guide

## 1. Run locally first
    python -m venv .venv
    .venv\Scripts\activate            (macOS/Linux: source .venv/bin/activate)
    pip install -r requirements.txt
    streamlit run app.py

## 2. Push to GitHub
    git init
    git add .
    git commit -m "CNC RUL intelligence dashboard"
    git branch -M main
    git remote add origin https://github.com/<you>/rul-cnc-dashboard.git
    git push -u origin main

## 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "Create app" → "Deploy a public app from GitHub".
3. Repository: <you>/rul-cnc-dashboard · Branch: main · Main file path: app.py
4. Advanced settings: choose Python 3.11 or 3.12 (TensorFlow needs a supported Python version).
   Keep the pins in requirements.txt: scikit-learn==1.6.1 and tensorflow-cpu==2.21.0 (the versions used in Colab).
5. Click Deploy and wait for the build. The first build can take several minutes.
6. Update later by pushing to GitHub; the app redeploys automatically.

## 4. Check after deployment
- Page 1 shows the leaderboard and the leader for each metric (LSTM leads R², RMSE and MAE; CNN-LSTM leads the asymmetric score).
- Page 2 loads the CNN-LSTM model and predicts from the bundled Excel sample.

## Troubleshooting
- Build fails on tensorflow-cpu: change the pin in requirements.txt to the nearest version available for your Python, then push again.
- "CNN-LSTM weights could not be loaded": models/ must contain preprocessing.json and cnn_lstm_seed*.keras.
- Model loads but predictions look wrong: the TensorFlow version must match the one used in Colab.
