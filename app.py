#!/usr/bin/env python
from flask import Flask, flash, redirect, render_template, request, url_for
from utils import YellowModel

app = Flask(__name__)

modelobj = YellowModel('England')
modelobj.load_model('best_estimator_one-hot')


@app.route('/')
def index():
    return render_template(
        'index.html',
        data=[{'name': x} for x in modelobj.columns])


@app.route("/test", methods=['GET', 'POST'])
def test():
    home_team = request.form.get('team1')
    away_team = request.form.get('team2')
    referee = request.form.get('referee')
    prediction = modelobj.predict(home_team, away_team, referee)[0]
    min_prediction = prediction - modelobj.confidense
    max_prediction = prediction + modelobj.confidense
    text = "min: %s \n\n mid: %s \n max: %s \n" % (min_prediction, prediction, max_prediction)
    return text

if __name__=='__main__':
    app.run(debug=True)