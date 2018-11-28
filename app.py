#!/usr/bin/env python
from flask import Flask, flash, redirect, render_template, request, url_for
from utils import GlobalModel, LocalModel, CompinedModel

app = Flask(__name__)

globalmodel = GlobalModel('England_Scotland')
globalmodel.load_model("best_estimator_global_mean_England_Scotland.pkl")
localmodel = LocalModel('England')
localmodel.load_model("best_estimator_local_one-hot_England_2018.pkl")
modelcompined = CompinedModel('England_compined', globalmodel, localmodel)
modelcompined.load_model("best_estimator_compined_England_compined.pkl")


@app.route('/')
def index():
    return render_template('index.html', data=[{'name': x} for x in localmodel.columns])


@app.route("/results", methods=['GET', 'POST'])
def results():
    home_team = request.form.get('team1').split("_")[1]
    away_team = request.form.get('team2').split("_")[1]
    referee = request.form.get('referee').split("_")[1]
    prediction = modelcompined.predict('2018', home_team, away_team, referee)[0]
    min_prediction = prediction - modelcompined.confidense
    max_prediction = prediction + modelcompined.confidense
    text = "min: %s \n\n mid: %s \n max: %s \n" % (min_prediction, prediction, max_prediction)
    return text

if __name__=='__main__':
    app.run(debug=True)