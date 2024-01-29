from flask import Flask, render_template,request,redirect,url_for,jsonify  
from flask_cors import CORS 
import sqlite3
import requests
import torch
from torch.nn.functional import softmax
# from flask_sqlalchemy import SQLAlchemy
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
app = Flask(__name__,template_folder='templates', static_folder='static')
model_dir = "complete"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
CORS(app)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
# db = SQLAlchemy(app)
def init_db():
    conn = sqlite3.connect('users.db')
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
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Insert user details into the database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))

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
    return render_template('chatgpt-detection.html')

@app.route('/processed_text', methods=['POST'])
def process_text():
    text = request.form.get('text')
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    max_prob, predicted_class_id = torch.max(probs, dim=1)
    print("Probabilities:", f"{round(max_prob.item() * 100, 2)}%")
    print("Predicted Class:", model.config.id2label[predicted_class_id.item()])
    processed_result = text
    return render_template('result.html', result=processed_result,probs=round(max_prob.item() * 100, 2),pclass=model.config.id2label[predicted_class_id.item()])

@app.route('/deepfakedetection')

def deepfakedetection():
    return render_template('deepfake-detection.html')

@app.route('/audiospoofdetection')
def audiospoofdetection():
    return render_template('audio-spoof-detection.html')

if __name__ == '__main__':
    app.run(debug=True)
