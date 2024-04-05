from flask import Flask, render_template,request,redirect,url_for,jsonify,flash,send_file
from flask_cors import CORS 
import sqlite3
import requests
import torch
from torch.nn.functional import softmax
import shap
import matplotlib.pyplot as plt
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline,AutoModel
from shap.plots._force_matplotlib import draw_additive_plot
app = Flask(__name__,template_folder='templates', static_folder='static')
app.secret_key = 'password'
model_dir = "temp"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
CORS(app)
def init_db():
    conn=sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error=None
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if not username:
            error='Username is required'
        elif not password:
            error='Password is required'

        # Insert user details into the database
        if error is None:
            try:
                conn = sqlite3.connect('users.db')
                cursor = conn.cursor()
                cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
                conn.commit()
                conn.close()
            except cursor.IntegrityError:
                error=f"User {username} is already registered."
        else:
            return redirect(url_for('login'))
        return redirect(url_for('login'))
    flash(error)
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if user exists in the database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email=? AND password=?', (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid credentials. Please try again.'

    return render_template('login.html')


@app.route('/')
def home():
    return render_template('/landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/chatgptdetection',methods=['GET','POST'])
def chatgptdetection():
    return render_template('result.html')

@app.route('/deepfakedetection')
def deepfakedetection():
    return render_template('deepfake-detection.html')

@app.route('/audiospoofdetection')
def audiospoofdetection():
    return render_template('audio-spoof-detection.html')

if __name__ == '__main__':
    app.run(debug=True)
