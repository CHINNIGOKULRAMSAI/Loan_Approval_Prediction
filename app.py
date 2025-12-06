import os
import smtplib
from email.message import EmailMessage

from flask import (
    Flask, render_template, request, flash,
    redirect, url_for, session, jsonify
)
from models import db, User

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["SECRET_KEY"] = "SECRET_KEY"

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = 'chinnigokul43@gmail.com'
app.config["MAIL_PASSWORD"] = 'ayuxdlrhgbmehvpw'

db.init_app(app)

with app.app_context():
    db.create_all()


def send_email(subject, recipient, body):
    username = app.config.get("MAIL_USERNAME")
    password = app.config.get("MAIL_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = recipient
    msg.set_content(body)

    try:
        with smtplib.SMTP(app.config["MAIL_SERVER"], app.config["MAIL_PORT"]) as server:
            if app.config.get("MAIL_USE_TLS"):
                server.starttls()
            server.login(username, password)
            server.send_message(msg)
    except Exception as e:
        app.logger.error(f"Error sending email: {e}")


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            flash("Username already exists")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists")
            return redirect(url_for("register"))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        send_email(
            subject="Welcome to LoanApp!",
            recipient=user.email,
            body=f"Hello {user.username},\n\nThank you for registering with LoanApp!",
        )

        flash("Registration successful! Please log in.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session["user_id"] = user.id
            flash("Login successful!")
            return redirect(url_for("predict_datapoint"))
        else:
            flash("Invalid username or password")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if "user_id" not in session:
        flash("Please log in to access this page.")
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template(
            "home.html",
            results=None,
            confidence=None,
            input_summary=None,
            explanation=None,
        )
    
    data = CustomData(
        loan_id=request.form.get("loan_id"),
        no_of_dependents=request.form.get("no_of_dependents"),
        education=request.form.get("education"),
        self_employed=request.form.get("self_employed"),
        income_annum=request.form.get("income_annum"),
        loan_amount=request.form.get("loan_amount"),
        loan_term=request.form.get("loan_term"),
        cibil_score=request.form.get("cibil_score"),
        residential_assets_value=request.form.get("residential_assets_value"),
        commercial_assets_value=request.form.get("commercial_assets_value"),
        luxury_assets_value=request.form.get("luxury_assets_value"),
        bank_asset_value=request.form.get("bank_asset_value"),
    )

    pred_df = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    user_id = session.get("user_id")
    user = User.query.get(user_id) if user_id else None

    # Build result details for display
    prediction = int(results[0])
    result_text = (
        "Approved" if prediction == 1 else "Rejected"
    )

    # Email the user about their result
    if user:
        send_email(
            subject="Your Loan Prediction Result",
            recipient=user.email,
            body=f"Hello {user.username},\n\nYour loan application is likely to be {result_text}.",
        )

    # Persist minimal context to session and redirect to result page
    session["last_result"] = {
        "status": result_text,
        "raw": prediction,
        "loan_id": request.form.get("loan_id"),
        "income_annum": request.form.get("income_annum"),
        "loan_amount": request.form.get("loan_amount"),
        "loan_term": request.form.get("loan_term"),
        "cibil_score": request.form.get("cibil_score"),
    }

    return redirect(url_for("result"))


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully.")
    return redirect(url_for("login"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        data_obj = CustomData(
            loan_id=data["loan_id"],
            no_of_dependents=data["no_of_dependents"],
            education=data["education"],
            self_employed=data["self_employed"],
            income_annum=data["income_annum"],
            loan_amount=data["loan_amount"],
            loan_term=data["loan_term"],
            cibil_score=data["cibil_score"],
            residential_assets_value=data["residential_assets_value"],
            commercial_assets_value=data["commercial_assets_value"],
            luxury_assets_value=data["luxury_assets_value"],
            bank_asset_value=data["bank_asset_value"],
        )
    except KeyError as e:
        return jsonify({"error": f"Missing input: {str(e)}"}), 400

    pred_df = data_obj.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    results, explanation = predict_pipeline.predict(pred_df)

    return jsonify(
        {
            "prediction": int(results[0]),
            "explanation": explanation,
        }
    )


@app.route("/result", methods=["GET"])
def result():
    if "user_id" not in session:
        flash("Please log in to access this page.")
        return redirect(url_for("login"))

    data = session.get("last_result")
    if not data:
        flash("No recent prediction found. Please submit the form.")
        return redirect(url_for("predict_datapoint"))

    return render_template("result.html", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
