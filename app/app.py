from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd
import plotly as pl
import plotly.express as px
import matplotlib.pyplot as plt
from joblib import load


app = Flask(__name__)


def PricePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 7)
    loaded_model = load('model.joblib')
    result = loaded_model.predict(to_predict)
    return result[0]


def plot_features(columns, importances):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances}).sort_values("feature_importances", ascending=False).reset_index(drop=True))
    fig, ax = plt.subplots()
    ax.barh(df["features"], df["feature_importances"])

def BuildVisuals():
    df = pd.read_csv('toyota-cars-temp.csv')
    df_cur = df.drop(columns=['price'])
    for label, content in df_cur.items():
        if pd.api.types.is_string_dtype(content):
            df_cur[label] = content.astype("category").cat.as_ordered()
            df_cur[label] = pd.Categorical(content).codes
    fig = pl.hist_frame(df.price)
    fig.write_image("static/datadistribution.svg", width=500)
    correlation = df_cur.corr()
    fig_cor = px.imshow(correlation)
    fig_cor.write_image("static/correlation_hist.svg", width=500)
    model = load('model.joblib')
    importances = model.feature_importances_
    features_df = (pd.DataFrame({"features": df_cur.columns, "feature_importances": importances}).sort_values("feature_importances", ascending=False).reset_index(drop=True))
    fig_features = px.bar(features_df, x=features_df.features, y=features_df.feature_importances)
    fig_features.write_image("static/feature_importances.svg", width=500)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = round(PricePredictor(to_predict_list), 2)
        return render_template("result.html", prediction=result)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/forgeeks")
def geeks():
    BuildVisuals()
    return render_template("forgeeks.html", href="static/datadistribution.svg", href2="static/correlation_hist.svg", href3="static/feature_importances.svg")