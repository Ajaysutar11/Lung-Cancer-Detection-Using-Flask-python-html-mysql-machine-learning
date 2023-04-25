import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('HDP.html')

@app.route('/predict',methods=['POST'])
def predict():

    print(request.form)
    rad = float(request.form['radius_mean'])
    text = float(request.form['texture_mean'])
    peri = float(request.form['perimeter_mean'])
    area = float(request.form['area_mean'])
    smooth = float(request.form['smoothness_mean'])
    compact = float(request.form['compactness_mean'])
    concavity= float(request.form['concavity_mean'])
    concave = float(request.form['concave points_mean'])
    symmetry = float(request.form['symmetry_mean'])
    fractal = float(request.form['fractal_dimension_mean'])
    rad_se = float(request.form['radius_se'])
    text_se = float(request.form['texture_se'])
    peri_se = float(request.form['perimeter_se'])
    area_se = float(request.form['area_se'])
    smooth_se = float(request.form['smoothness_se'])
    compact_se = float(request.form['compactness_se'])
    concave_se = float(request.form['concavity_se'])
    con_points = float(request.form['concave points_se'])
    sym_se = float(request.form['symmetry_se'])
    fractal_se = float(request.form['fractal_dimension_se'])
    rad_worst = float(request.form['radius_worst'])
    text_worst = float(request.form['texture_worst'])
    peri_worst = float(request.form['perimeter_worst'])
    area_worst = float(request.form['area_worst'])
    smooth_worst = float(request.form['smoothness_worst'])
    compact_worst = float(request.form['compactness_worst'])
    concave_worst = float(request.form['concavity_worst'])
    con_points_worst = float(request.form['concave points_worst'])
    sym_worst = float(request.form['symmetry_worst'])
    fractal_worst = float(request.form['fractal_dimension_worst'])

    mymodel = pickle.load(open('logreg.pkl', "rb"))
    query = [[rad,text,peri,area,smooth,compact,concavity,concave,symmetry,fractal,rad_se,text_se,peri_se,
    area_se,smooth_se,compact_se,concave_se,con_points,sym_se,fractal_se,rad_worst,text_worst,peri_worst,
    area_worst,smooth_worst,compact_worst,concave_worst,con_points_worst,sym_worst,fractal_worst]]

    data = pd.DataFrame(query, columns=['radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se',
    'symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'])

    print(data)
    output = np.round(mymodel.predict(data),2)
    if output == 0:
        a = 'THE PATIENT IS NOT LIKELY TO HAVE A HEART FAILURE'
    else:
        a = 'THE PATIENT IS LIKELY TO HAVE A HEART FAILURE'
    return "<h1> {} </h1>".format(a)
        
if __name__=='__main__':
    app.run(debug=True)