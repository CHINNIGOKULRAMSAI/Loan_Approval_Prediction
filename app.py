from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from models import db,User
import os
import sys
import pandas as pd
import numpy as np

import smtplib
from email.message import EmailMessage

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SECRET_KEY'] = 'your_secret_key_here'
db.init_app(app)

# email configuration for gmail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # fixed typo
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'chinnigokul43@gmail.com'
app.config['MAIL_PASSWORD'] = 'ayuxdlrhgbmehvpw'

@app.route("/ping")
def ping():
    return "pong"

@app.route('/')
def index():
    return render_template('index.html')

def send_email(subject, recipient, body):

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = app.config['MAIL_USERNAME']
    msg['To'] = recipient
    msg.set_content(body)

    with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
        if app.config.get('MAIL_USE_TLS'):
            server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)

@app.route('/register', methods= ['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash("Username already exists")
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash("Email already exists")
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        send_email(
            subject='Welcome to LoanApp!',
            recipient=user.email,
            body=f'Hello {user.username},\n\nThank you for registering with LoanApp!'
        )
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            flash('Login successful!')
            return redirect(url_for('predict_datapoint'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if 'user_id' not in session:
        flash('Please log in to access this page.')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('home.html', confidence=None, input_summary=None)
    else:
        data = CustomData(
            loan_id = request.form.get('loan_id'),
            no_of_dependents = request.form.get('no_of_dependents'),
            education = request.form.get('education'),
            self_employed = request.form.get('self_employed'),
            income_annum = request.form.get('income_annum'),
            loan_amount = request.form.get('loan_amount'),
            loan_term = request.form.get('loan_term'),
            cibil_score = request.form.get('cibil_score'),
            residential_assets_value = request.form.get('residential_assets_value'),
            commercial_assets_value = request.form.get('commercial_assets_value'),
            luxury_assets_value = request.form.get('luxury_assets_value'),
            bank_asset_value = request.form.get('bank_asset_value'))
        
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results, explanation = predict_pipeline.predict(pred_df)

        user_id = session.get('user_id')
        user = User.query.get(user_id) if user_id else None
        if user:
            result_text = (
                "Congratulations! Your loan is likely to be Approved."
                if results[0] == 1 else
                "Sorry, your loan application is likely to be Rejected."
            )
            send_email(
                subject='Your Loan Prediction Result',
                recipient=user.email,
                body=f'Hello {user.username},\n\n{result_text}'
            )

        return render_template('home.html', results=results[0], confidence=None, input_summary=None, explanation=explanation)

@app.route('/logout',methods = ["GET","POST"])
def logout():
    session.pop('user_id',None)
    flash('Logged out successfully.')
    return redirect(url_for('login'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        loan_id = data['loan_id']
        no_of_dependents = data['no_of_dependents']
        education = data['education']
        self_employed = data['self_employed']
        income_annum = data['income_annum']
        loan_amount = data['loan_amount']
        loan_term = data['loan_term']
        cibil_score = data['cibil_score']
        residential_assets_value = data['residential_assets_value']
        commercial_assets_value = data['commercial_assets_value']
        luxury_assets_value = data['luxury_assets_value']
        bank_asset_value = data['bank_asset_value']
    except KeyError as e:
        return jsonify({'error': f'Missing input: {str(e)}'}), 400

    data_obj = CustomData(
        loan_id=loan_id,
        no_of_dependents=no_of_dependents,
        education=education,
        self_employed=self_employed,
        income_annum=income_annum,
        loan_amount=loan_amount,
        loan_term=loan_term,
        cibil_score=cibil_score,
        residential_assets_value=residential_assets_value,
        commercial_assets_value=commercial_assets_value,
        luxury_assets_value=luxury_assets_value,
        bank_asset_value=bank_asset_value
    )
    pred_df = data_obj.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    results, explanation = predict_pipeline.predict(pred_df)

    return jsonify({
        'prediction': int(results[0]),  # 1 or 0
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
