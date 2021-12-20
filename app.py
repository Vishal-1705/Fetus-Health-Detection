from flask import Flask, render_template, request, redirect, url_for
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model = joblib.load('catb_model.sav')

scaler = MinMaxScaler()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        FHR = float(request.form['fhr'])
        Accelerations = float(request.form['accelerations'])
        SevereDecelerations = float(request.form['severedecelerations'])
        FetalMovement = float(request.form['fetalmovement'])
        UterineContractions = float(request.form['uterinecontractions'])
        LightDecelerations = float(request.form['lightdecelerations'])
        ProlonguedDecelerations = float(request.form['prolongueddecelerations'])
        AbShortTermVar = float(request.form['abnormalshorttermvariability'])
        MeanAbShortTermVar = float(request.form['meanshorttermvariability'])
        AbLongTermVar = float(request.form['abnormallongtermvariability'])
        MeanAbLongTermVar = float(request.form['meanlongtermvariability'])
        
        FHR = (FHR-106)/(160-106)
        Accelerations = (Accelerations-0.000)/(0.019-0.000)
        SevereDecelerations = (SevereDecelerations-0.000)/(0.001-0.000)
        FetalMovement = (FetalMovement-0.000)/(0.481-0.000)
        UterineContractions = (UterineContractions-0.000)/(0.015-0.000)
        LightDecelerations = (LightDecelerations-0.000)/(0.015-0.000)
        ProlonguedDecelerations = (ProlonguedDecelerations-0.000)/(0.005-0.000)
        AbShortTermVar = (AbShortTermVar-12)/(87-12)
        MeanAbShortTermVar = (MeanAbShortTermVar-0.2)/(7.0-0.2)
        AbLongTermVar = (AbLongTermVar-0.0)/(91.0-0.0)
        MeanAbLongTermVar = (MeanAbLongTermVar-0.0)/(50.7-0.0)

        prediction = model.predict([[FHR, Accelerations, FetalMovement, UterineContractions, LightDecelerations, SevereDecelerations, ProlonguedDecelerations, AbShortTermVar, MeanAbShortTermVar, AbLongTermVar, MeanAbLongTermVar]])
        output = prediction[0]

        if output=="Normal":
            return render_template('index.html', prediction_text="Normal")
        elif output=="Suspectible":
            return render_template('index.html', prediction_text="Suspectible")
        else:
            return render_template('index.html', prediction_text="Pathological")


if __name__ == "__main__":
    app.run(debug=True)