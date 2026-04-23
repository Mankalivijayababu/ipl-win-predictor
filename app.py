from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load trained model
with open("model/ipl_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():

    batting_team = request.form.get("batting_team", "Team")
    bowling_team = request.form.get("bowling_team", "Opponent")

    runs_left = float(request.form["runs_left"])
    balls_left = float(request.form["balls_left"])
    wickets = float(request.form["wickets"])
    target_runs = float(request.form["target_runs"])
    cur_run_rate = float(request.form["cur_run_rate"])
    req_run_rate = float(request.form["req_run_rate"])

    prediction = model.predict_proba([[runs_left,
                                      balls_left,
                                      wickets,
                                      target_runs,
                                      cur_run_rate,
                                      req_run_rate]])

    win_prob = round(prediction[0][1] * 100, 2)

    return render_template(
        "index.html",
        prediction=f"{batting_team} Winning Probability: {win_prob}%",
        prediction_prob=win_prob,

        batting_team=batting_team,
        bowling_team=bowling_team,
        runs_left=runs_left,
        balls_left=balls_left,
        wickets=wickets,
        target_runs=target_runs,
        cur_run_rate=cur_run_rate,
        req_run_rate=req_run_rate
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)