from flask import Flask, jsonify
from kaggle_driver import download_data
from helper import get_results
import joblib


app = Flask(__name__)


@app.route('/')
def card_predictions():
    results = get_results(model, cols, scaler)

    return jsonify(results)



if __name__ == '__main__':
    download_data()

    model = joblib.load('../Model and Scaler/final_svc.pkl')
    scaler = joblib.load('../Model and Scaler/final_scaler.pkl')
    cols = joblib.load('../Model and Scaler/cols.pkl')

    app.run()
