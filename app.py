from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# load the vectorizer
vectorizer, clf = joblib.load("models/best_spam_model.joblib")


@app.route("/")
def home():
    # render the html template
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # check if the form is submitted
    message = request.form["message"]

    # feed the input text to the vectorizer and clf model
    message_vec = vectorizer.transform([message])
    prediction = clf.predict(message_vec)[0]
    prob = clf.predict_proba(message_vec)[0]

    result = {
        "message": message,
        "predicted_label": "Spam" if prediction == 1 else "Not Spam",
        "spam_probability": prob[1],
        "ham_probability": prob[0],
    }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
