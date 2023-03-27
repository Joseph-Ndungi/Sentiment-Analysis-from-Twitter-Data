from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import json
import pandas as pd
from forms import *


app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

@app.route('/' , methods=['GET', 'POST'])
def index():
    # form = DateForm()
    # if form.validate_on_submit():
    #     print(form.startDate.data)
    #     print(form.endDate.data)
    #     print(form.query.data)
    #     return redirect(url_for('index'))
    return render_template('dashboard.html')



@app.route('/sentiments', methods=['GET', 'POST'])
def sentiments():

    # form = DateForm()
    # if form.validate_on_submit():
    #     print(form.startDate.data)
    #     print(form.endDate.data)
    #     print(form.query.data)
    #     return redirect(url_for('index'))
    return render_template('query.html')

if __name__ == '__main__':
    app.run(debug=False)