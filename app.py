import os
import pickle
import io
import json
import requests
import datetime
import math 
import random
import numpy as np
import csv
import hashlib
import re
import ssl
from dotenv import load_dotenv
import google.generativeai as genai
import markdown
from fpdf import FPDF 
from flask_login import current_user 
import qrcode  
from flask_limiter import Limiter 
from flask_limiter.util import  get_remote_address
from flask import make_response
from flask_mail import Mail, Message
from flask import session 
from flask import Flask, render_template, redirect, url_for, flash, request, make_response, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.linear_model import LinearRegression

#=====================================
# 1. CONFIGURATION & EXTENSIONS
#=====================================

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key-123')

db_url = os.getenv('DATABASE_URL')

if db_url:
    
    if "?" in db_url:
        db_url = db_url.split("?")[0]
        
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+pg8000://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+pg8000://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'sqlite:///' + os.path.join(basedir, 'blood_ai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ==============================
# MAIL CONFIG (Updated for Production)
# ==============================
app.config["MAIL_SERVER"] = "api.brevo.com" 
app.config["MAIL_PORT"] = 443
app.config["MAIL_USE_TLS"] = False
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")

# GEMINI API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ Warning: GEMINI_API_KEY is still missing!")

mail = Mail(app)

from threading import Thread

def send_async_email(app, msg):
    with app.app_context():
        try:
            url = "https://api.brevo.com/v3/smtp/email"
            api_key = os.environ.get("MAIL_PASSWORD") 
            
            payload = {
                "sender": {"email": os.environ.get("MAIL_DEFAULT_SENDER")},
                "to": [{"email": msg.recipients[0]}],
                "subject": msg.subject,
                "textContent": msg.body
            }
            
            headers = {
                "accept": "application/json",
                "api-key": api_key,
                "content-type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code in [200, 201, 202]:
                print("✅ Email sent successfully via Brevo API!")
            else:
                print(f"❌ Brevo Error: {response.text}")
        except Exception as e:
            print(f"❌ API Email Error: {e}")

context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

db = SQLAlchemy(app, engine_options={
    "connect_args": {"ssl_context": context},
    "pool_pre_ping": True,   
    "pool_recycle": 280    
})

login_manager = LoginManager(app)
login_manager.login_view = 'login'

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

#=====================================
# 2. ML ENGINE (PREDICTION MODEL)
# ====================================
ml_model = None 

def load_or_train_model():
    global ml_model
    model_file = 'blood_ai_model.pkl'
    
    if os.path.exists(model_file):
        try:
            with open(model_file, 'rb') as f:
                ml_model = pickle.load(f)
            print("✅ AI Model Loaded from Disk (Super Fast!)")
        except Exception as e:
            print(f"❌ Error loading model: {e}. Retraining...")
            force_train(model_file)
    else:       
        print("⚠️ Model file not found. Training new model...")
        force_train(model_file)

def force_train(filename):
    global ml_model
    ml_model = LinearRegression()
    
    X = np.array([
        [1.0,  365, 100, 25], [50.0, 30,  50,  60], 
        [5.0,  120, 90,  30], [20.0, 60,  70,  45], 
        [2.0,  90,  80,  22], [10.0, 30,  60,  50] 
    ])
    y = np.array([99, 20, 80, 50, 90, 40]) 
    
    try:
        ml_model.fit(X, y)
    
        with open(filename, 'wb') as f:
            pickle.dump(ml_model, f)
        print(f"💾 AI Model Trained & Saved to '{filename}'")
    except Exception as e:
        print(f"ML Error: {e}")

def get_ai_priority(distance, last_donation_date, health_score, dob):
  
    if not last_donation_date:
        days_wait = 365 
    else:
        if isinstance(last_donation_date, str):
            last_donation_date = datetime.datetime.strptime(last_donation_date, '%Y-%m-%d').date()
        delta = datetime.date.today() - last_donation_date
        days_wait = delta.days

    age = 30 
    if dob:
        if isinstance(dob, str):
            dob = datetime.datetime.strptime(dob, '%Y-%m-%d').date()
        today = datetime.date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    
    try:
        if ml_model is None:
            load_or_train_model() 
            
        features = np.array([[float(distance), float(days_wait), float(health_score), float(age)]])
        score = ml_model.predict(features)[0]
        final_score = min(max(int(score), 0), 100)
        
        if final_score >= 75: priority_label = "High"
        elif final_score >= 50: priority_label = "Medium"
        else: priority_label = "Low"
            
        return final_score, priority_label, age
    except:
        return 50, "Medium", age
 
#=====================================
# 3. HELPER FUNCTIONS
#=====================================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(R * c, 2)                     
                             
#=====================================
# 4. DATABASE MODELS
#=====================================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    donor_id = db.Column(db.String(10), unique=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False) 
    phone = db.Column(db.String(15))
    blood_group = db.Column(db.String(5))
    dob = db.Column(db.Date)
    health_score = db.Column(db.Integer, default=100)
    lat = db.Column(db.Float) 
    lon = db.Column(db.Float) 
    last_donation = db.Column(db.Date)
    is_available = db.Column(db.Boolean, default=True)
    report_count = db.Column(db.Integer, default=0) 
    is_blocked = db.Column(db.Boolean, default=False) 
    donation_count = db.Column(db.Integer, default=0)
    @property
    def badge_info(self):
        if self.donation_count >= 10:
            return {"name": "Gold Legend", "icon": "🥇", "color": "#FFD700", "class": "warning"}
        elif self.donation_count >= 5:
            return {"name": "Silver Hero", "icon": "🥈", "color": "#C0C0C0", "class": "secondary"}
        elif self.donation_count >= 1:
            return {"name": "Bronze Saver", "icon": "🥉", "color": "#CD7F32", "class": "danger"}
        else:
            return {"name": "New Donor", "icon": "🌱", "color": "#28a745", "class": "success"}


class BloodStock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    blood_group = db.Column(db.String(5), unique=True)
    units = db.Column(db.Integer, default=0)

class RequestNotification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    req_id = db.Column(db.Integer, db.ForeignKey('blood_request.id'))
    donor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    status = db.Column(db.String(20), default='Pending') 
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    request_info = db.relationship('BloodRequest', backref='notifications')
    donor_info = db.relationship('User', backref='received_requests')

class BloodRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    fulfilled_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)     
    patient_name = db.Column(db.String(100))
    blood_group = db.Column(db.String(5))
    units = db.Column(db.Integer)
    urgency = db.Column(db.Integer) 
    contact_number = db.Column(db.String(15))
    hospital_address = db.Column(db.String(200)) 
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)
    status = db.Column(db.String(20), default='Active') 
    bag_serial_no = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class BloodCamp(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    organizer = db.Column(db.String(100))
    location = db.Column(db.String(200), nullable=False)
    date = db.Column(db.String(20), nullable=False) 
    time = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class CampRegistration(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    camp_id = db.Column(db.Integer, db.ForeignKey('blood_camp.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    registered_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class BloodChain(db.Model):
    id = db.Column(db.Integer,  primary_key=True)
    req_id = db.Column(db.Integer, db.ForeignKey('blood_request.id'))
    step = db.Column(db.String(50)) 
    details = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    prev_hash = db.Column(db.String(64)) 
    current_hash = db.Column(db.String(64)) 

    def calculate_hash(self):        
        block_string = f"{self.req_id}{self.step}{self.details}{self.timestamp}{self.prev_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

def add_blockchain_record(req_id, step, details):
       
    last_block = BloodChain.query.filter_by(req_id=req_id).order_by(BloodChain.id.desc()).first()
    
    prev_hash = "0000000000000000" 
    if last_block:
        prev_hash = last_block.current_hash
    
    new_block = BloodChain(
        req_id=req_id,
        step=step,
        details=details,
        prev_hash=prev_hash
    )
    
    new_block.current_hash = new_block.calculate_hash()
    
    db.session.add(new_block)
    db.session.commit()

#=====================================
# 5. AUTHENTICATION ROUTES
#=====================================

@app.route('/register/donor', methods=['GET', 'POST'])
def register_donor():
    if request.method == 'POST':
        hashed_pw = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        
        lat = 0.0
        lon = 0.0
        if request.form.get('lat'):
            lat = float(request.form['lat'])
            lon = float(request.form['lon'])

        health_score = 100
        
        if request.form.get('q_hiv_hepa') == 'yes': health_score = 0
        elif request.form.get('q_cancer') == 'yes': health_score = 0
        elif request.form.get('q_heart') == 'yes': health_score = 0
        elif request.form.get('q_diabetes') == 'yes': health_score = 0        
        else:            
            if request.form.get('q_weight') == 'yes': health_score -= 20
            if request.form.get('q_alcohol') == 'yes': health_score -= 10
            if request.form.get('q_surgery') == 'yes': health_score -= 30
            if request.form.get('q_tattoo') == 'yes': health_score -= 20
            if request.form.get('q_meds') == 'yes': health_score -= 15
            if request.form.get('q_preg') == 'yes': health_score -= 30
            
        if health_score < 0: health_score = 0
        
        session['temp_user'] = {
            'name': request.form['name'],
            'email': request.form['email'],
            'password': hashed_pw,
            'role': 'donor',  
            'phone': request.form['phone'],
            'blood_group': request.form.get('blood_group'),
            'dob': request.form.get('dob'),
            'health_score': health_score,
            'lat': lat, 'lon': lon
        } 

        otp = random.randint(100000, 999999)
        session['otp'] = otp
        
        try:
            msg = Message('BloodAI Verification Code', recipients=[request.form['email']])
            msg.subject = "Verify your BloodAI Account"
            msg.body = f"""
            Hello,

            Welcome to BloodAI! 🩸 

            To complete your registration, please use the following verification code:
            OTP: {otp}

            If you did not request this, please ignore this email.

            Stay Safe,
            Team BloodAI
            """
            Thread(target=send_async_email, args=(app, msg)).start()
            flash('OTP sent to your email! Please verify.', 'info')
        except Exception:
            print(f"------------\nOTP (Email Failed): {otp}\n------------")
            flash(f'Email failed (Simulated). OTP is {otp}', 'warning')

        return redirect(url_for('verify_otp'))
            
    return render_template('auth/register_donor.html')

@app.route('/register/requester', methods=['GET', 'POST'])
def register_requester():
    if request.method == 'POST':
        hashed_pw = generate_password_hash(request.form['password'], method='pbkdf2:sha256')

        session['temp_user'] = {
            'name': request.form['name'],
            'email': request.form['email'],
            'password': hashed_pw,
            'role': 'requester', 
            'phone': request.form['phone'],
            'blood_group': None,
            'dob': None,
            'health_score': 100, 
            'lat': 0.0, 'lon': 0.0
        }

        otp = random.randint(100000, 999999)
        session['otp'] = otp
        
        try:
            msg = Message('BloodAI Verification Code', recipients=[request.form['email']])
            msg.subject = "Verify your BloodAI Account"
            msg.body = f"""
            Hello,

            Welcome to BloodAI! 🩸 

            To complete your registration, please use the following verification code:
            OTP: {otp}

            If you did not request this, please ignore this email.

            Stay Safe,
            Team BloodAI
            """
            Thread(target=send_async_email, args=(app, msg)).start()
            flash('OTP sent to your email! Please verify.', 'info')
        except Exception:
            print(f"------------\nOTP (Email Failed): {otp}\n------------")
            flash(f'Email failed (Simulated). OTP is {otp}', 'warning')

        return redirect(url_for('verify_otp'))
            
    return render_template('auth/register_requester.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def verify_otp(): 
    if 'temp_user' not in session:
        flash('Session expired. Register again.', 'danger')
        return redirect(url_for('register'))

    if request.method == 'POST':
        user_otp = request.form['otp']
        generated_otp = session.get('otp')

        if str(user_otp) == str(generated_otp):
            data = session['temp_user']

            dob_date = None
            if data['dob']:
                dob_date = datetime.datetime.strptime(data['dob'], '%Y-%m-%d').date()

            unique_id = str(random.randint(1000, 9999))
            while User.query.filter_by(donor_id=unique_id).first():
                unique_id = str(random.randint(1000, 9999))

            new_user = User(
                donor_id=unique_id,
                name=data['name'],
                email=data['email'],
                password=data['password'],
                role=data['role'],
                phone=data['phone'],
                blood_group=data['blood_group'],
                dob=dob_date,
                health_score=data['health_score'],
                lat=data['lat'], lon=data['lon'],
                last_donation=None
            )

            try: 
                db.session.add(new_user)
                db.session.commit()

                session.pop('temp_user', None)
                session.pop('otp', None)
                
                flash(f'Verification Successful! Your ID is #{unique_id}', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash(f'Database Error: {e}', 'danger')
                return redirect(url_for('register'))
        else:
            flash('Invalid OTP. Try again.', 'danger')

    return render_template('auth/verify.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        
        if user and user.is_blocked:
            flash('🚫 Your account has been BLOCKED due to multiple reports or low health score. Contact Admin.', 'danger')
            return redirect(url_for('login'))
 
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            if user.role == 'admin': return redirect(url_for('admin_dashboard'))
            if user.role == 'donor': return redirect(url_for('donor_dashboard'))
            return redirect(url_for('requester_dashboard'))
        else:
            flash('Login Unsuccessful. Check email and password.', 'danger')
    return render_template('auth/login.html')
    
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))    
    
#====================================
# 6. GENERAL ROUTES (HOME & CHAT)
#==================================== 
@app.route('/')
def home():
    return render_template('home.html')

def get_database_context():
    """
    Intha function database la irunthu live data va eduthu
    AI ku 'Context String' ah tharum.
    """
    try:
        groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
        donor_stats = []
        for g in groups:
            count = User.query.filter_by(role='donor', blood_group=g, is_available=True, is_blocked=False).count()
            if count > 0:
                donor_stats.append(f"{g}: {count} donors")
        
        donor_text = ", ".join(donor_stats) if donor_stats else "No active donors currently."

        stocks = BloodStock.query.all()
        stock_text = ", ".join([f"{s.blood_group}: {s.units} units" for s in stocks])

        camps = BloodCamp.query.order_by(BloodCamp.date.asc()).limit(3).all()
        camp_text = ""
        if camps:
            for c in camps:
                camp_text += f"- {c.title} at {c.location} on {c.date}\n"
        else:
            camp_text = "No upcoming camps scheduled."

        return f"""
        LIVE DATABASE STATUS:
        - Active Donors Nearby: {donor_text}
        - Blood Bank Inventory: {stock_text}
        - Upcoming Camps: 
        {camp_text}
        """
    except Exception as e:
        return "Database currently unavailable."

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '')

    if 'chat_history' not in session:
        session['chat_history'] = []

    user_context = "User is a Guest."
    if current_user.is_authenticated:
        user_context = f"""
        User Name: {current_user.name}
        Role: {current_user.role}
        Blood Group: {current_user.blood_group}
        Donor ID: {current_user.donor_id}
        Health Score: {current_user.health_score}
        Last Donation: {current_user.last_donation}
        Phone: {current_user.phone}
        """

    db_context = get_database_context()

    system_instruction = f"""
    You are 'BloodAI Assistant', a helpful AI for a Blood Donation App.
    
    RULES:
    1. ONLY answer questions related to blood donation, this app, health eligibility, and the provided live data.
    2. If asked about politics, coding, movies, or general topics, politely refuse and say you only discuss BloodAI.
    3. Use the LIVE DATA below to answer questions like "Any O+ donors?" or "Upcoming camps?".
    4. Speak in the SAME LANGUAGE the user asks (Tamil, Hindi, English, etc.).
    5. Keep answers short, friendly, and use emojis.
    6. Format output in HTML (use <b> for bold, <br> for new lines).

    {user_context}

    {db_context}

    APP INFO:
    - Admin: admin@bloodai.com
    - Emergency: Call 108
    - Features: AI Match, Live Tracking, Blockchain Certificates.
    """

    try:

        model = genai.GenerativeModel('gemini-flash-latest')

        current_history = session['chat_history']
        
        chat_session = model.start_chat(history=current_history)

        response = chat_session.send_message(f"System Context: {system_instruction}\n\nUser Question: {user_msg}")

        session['chat_history'].append({'role': 'user', 'parts': [user_msg]})
        session['chat_history'].append({'role': 'model', 'parts': [response.text]})

        if len(session['chat_history']) > 20:
            session['chat_history'] = session['chat_history'][-20:]
            
        session.modified = True  
        
        ai_reply = markdown.markdown(response.text)
        ai_reply = ai_reply.replace('<p>', '').replace('</p>', '')

        return {'response': ai_reply}

    except Exception as e:
        print(f"Gemini Error: {e}")

        session.pop('chat_history', None)
        return {'response': "⚠️ My memory was refreshed due to a network glitch. Please ask again!"}

#=====================================
# 7. REQUESTER ROUTES
#=====================================
@app.route('/requester/dashboard')
@login_required
def requester_dashboard():
    my_requests = BloodRequest.query.filter_by(requester_id=current_user.id).order_by(BloodRequest.created_at.desc()).all()
    return render_template('requester/dashboard.html', requests=my_requests)

@app.route('/request/new', methods=['GET', 'POST']) 
@login_required
def request_blood():
    if request.method == 'POST':
        req_lat = float(request.form['lat']) 
        req_lon = float(request.form['lon'])        
        new_req = BloodRequest(
            requester_id=current_user.id,
            patient_name=request.form['patient_name'],
            blood_group=request.form['blood_group'],
            units=int(request.form['units']),
            urgency=int(request.form['urgency']),
            contact_number=request.form['contact_number'],
            hospital_address=request.form['hospital_address'],
            lat=req_lat, lon=req_lon
        )
        db.session.add(new_req)
        db.session.commit()
        
        add_blockchain_record(
            req_id=new_req.id,
            step="REQUEST_INITIATED",
            details=f"Patient: {new_req.patient_name} | Hospital: {new_req.hospital_address} | Created By: {current_user.name}"
        )
        
        return redirect(url_for('find_donors', req_id=new_req.id))
    return render_template('requester/request_form.html')

@app.route('/request/matches/<int:req_id>')
@login_required
def find_donors(req_id):
    req = BloodRequest.query.get_or_404(req_id) 

    donors = User.query.filter_by(role='donor', blood_group=req.blood_group, is_available=True, is_blocked=False).all()
    
    scored_donors = []
    for donor in donors:       
        dist = calculate_distance(req.lat, req.lon, donor.lat, donor.lon)
        
        ai_score, label, age = get_ai_priority(dist, donor.last_donation, donor.health_score, donor.dob)
        
        scored_donors.append({
            'id': donor.id,
            'name': donor.name,
            'phone': donor.phone,
            'email': donor.email,
            'distance': dist,
            'health_score': donor.health_score,
            'age': age,
            'ai_score': ai_score,
            'priority_label': label, 
            'lat': donor.lat,
            'lon': donor.lon
        })
    
    scored_donors.sort(key=lambda x: x['ai_score'], reverse=True)
    
    return render_template('requester/matches.html', request=req, donors=scored_donors)  
 
@app.route('/send_request/<int:donor_id>/<int:req_id>')
@login_required
def send_app_request(donor_id, req_id):
    existing = RequestNotification.query.filter_by(donor_id=donor_id, req_id=req_id).first()
    
    if not existing:     
        new_notif = RequestNotification(donor_id=donor_id, req_id=req_id)
        db.session.add(new_notif)
        db.session.commit()
        
        donor = User.query.get(donor_id)
        req = BloodRequest.query.get(req_id)
        
        email_status = "Email Sent"
        try:
            msg = Message(f"URGENT: Blood Needed for {req.patient_name}!", recipients=[donor.email])
            msg.body = f"""
            URGENT BLOOD REQUEST - BLOODAI
            
            Hello {donor.name},
            
            You have received a direct request for blood donation.
            
            Patient: {req.patient_name}
            Blood Group: {req.blood_group}
            Hospital: {req.hospital_address}
            Contact: {req.contact_number}
            
            Please open the BloodAI App Dashboard to ACCEPT this request.
            """
            Thread(target=send_async_email, args=(app, msg)).start()
        except Exception as e:
            print(f"Email Error: {e}")
            email_status = "Email Failed (Check Internet)"

        flash(f'✅ Request sent to Dashboard & {email_status} to {donor.name}!', 'success')
        
    else:
        flash('⚠️ You have already requested this donor.', 'warning')
        
    return redirect(url_for('find_donors', req_id=req_id))
  
@app.route('/report_donor/<int:donor_id>/<int:req_id>')
@login_required
def report_donor(donor_id, req_id):
    donor = User.query.get_or_404(donor_id)
    
    donor.report_count += 1
  
    if donor.report_count >= 3:
        donor.is_blocked = True
        donor.is_available = False
        flash(f'Report submitted. System has AUTO-BLOCKED {donor.name} due to multiple reports.', 'dark')
    elif donor.health_score < 30:
        donor.is_blocked = True
        donor.is_available = False
        flash(f'Report submitted. Donor blocked due to critical health score.', 'dark')
    else:
        flash(f'Report submitted against {donor.name}. Warning issued.', 'warning')
        
    db.session.commit()
    return redirect(url_for('find_donors', req_id=req_id))

@app.route('/request/close/<int:req_id>')
@login_required
def close_request(req_id):  
    req = BloodRequest.query.get_or_404(req_id)
  
    if req.requester_id == current_user.id:
        req.status = 'Fulfilled'
        
        add_blockchain_record(
            req_id=req.id,
            step="BLOOD_TRANSFUSED",
            details=f"Blood units received at {req.hospital_address}. Case Closed. Verified by {current_user.name}."
        )
        
        db.session.commit()
        flash('Request closed successfully! Thank you.', 'success')
    return redirect(url_for('home'))

@app.route('/track/<int:donor_id>/<int:req_id>')
@login_required
def track_donor(donor_id, req_id):
    donor = User.query.get_or_404(donor_id)
    req = BloodRequest.query.get_or_404(req_id)
    return render_template('requester/track.html', donor=donor, request=req)    
    
#=====================================
# 8. DONOR ROUTES
#=====================================
@app.route('/donor/dashboard')
@login_required 
def donor_dashboard():
    days_left = 0
    if current_user.last_donation:
        delta = (datetime.date.today() - current_user.last_donation).days
        if delta < 90:
            days_left = 90 - delta
        else:
            if not current_user.is_available:
                current_user.is_available = True
                db.session.commit()

    direct_requests = RequestNotification.query.filter(
        RequestNotification.donor_id == current_user.id,
        RequestNotification.status.in_(['Pending', 'Accepted']) 
    ).order_by(RequestNotification.created_at.desc()).all()

    direct_req_ids = [n.req_id for n in direct_requests]
    
    general_requests = []
    if current_user.is_available:
        all_matches = BloodRequest.query.filter_by(status='Active', blood_group=current_user.blood_group).all()
        general_requests = [r for r in all_matches if r.id not in direct_req_ids]

    history = BloodRequest.query.filter_by(fulfilled_by_id=current_user.id).all()

    return render_template('donor/dashboard.html', 
                           direct_requests=direct_requests, 
                           general_requests=general_requests, 
                           history=history, 
                           days_left=days_left)

@app.route('/donor/accept/<int:req_id>')
@login_required 
def accept_request(req_id):
 
    notif = RequestNotification.query.filter_by(req_id=req_id, donor_id=current_user.id).first()

    if not notif:
        notif = RequestNotification(req_id=req_id, donor_id=current_user.id)
        db.session.add(notif)

    notif.status = 'Accepted'
    
    add_blockchain_record(
        req_id=req_id,
        step="DONOR_ACCEPTED",
        details=f"Donor {current_user.name} (ID: {current_user.donor_id}) accepted the request. Transit started."
    )
    
    db.session.commit()
    
    flash('You accepted the request! The Requester has been notified.', 'info')
    return redirect(url_for('donor_dashboard'))

@app.route('/donor/donate/<int:req_id>', methods=['POST'])
@login_required
def donate_blood(req_id):
    req = BloodRequest.query.get_or_404(req_id)
    
    serial_no = request.form.get('bag_serial')

    if not serial_no or serial_no.strip() == "":
        flash('⚠️ Error: Blood Bag Serial Number cannot be empty!', 'danger')
        return redirect(url_for('donor_dashboard'))

    existing_bag = BloodRequest.query.filter_by(bag_serial_no=serial_no).first()
    
    if existing_bag:
        if existing_bag.id != req.id:
            flash(f'🚫 FAKE ALERT: The Bag ID "{serial_no}" has ALREADY been used for another patient! Please check the bag ID.', 'danger')
            return redirect(url_for('donor_dashboard'))

    req.status = 'Donated'  
    req.fulfilled_by_id = current_user.id
    req.bag_serial_no = serial_no  

    notif = RequestNotification.query.filter_by(req_id=req_id, donor_id=current_user.id).first()
    if notif: notif.status = 'Fulfilled'

    current_user.last_donation = datetime.date.today()
    current_user.is_available = False  
    current_user.donation_count += 1
    
    add_blockchain_record(
        req_id=req.id,
        step="BLOOD_COLLECTED",
        details=f"Donor: {current_user.name} | Bag: {serial_no} | Status: In Transit"
    )
    
    db.session.commit()

    flash(f'Hero! ❤️ Donation Recorded with Bag ID: {serial_no}. +1 Point added.', 'success')
    flash('Blockchain Ledger Updated! 🔗', 'success')
    return redirect(url_for('donor_dashboard'))

@app.route('/donor/toggle', methods=['POST'])
@login_required
def toggle_availability():
 
    if current_user.last_donation:
        delta = (datetime.date.today() - current_user.last_donation).days
        if delta < 90:
            days_left = 90 - delta
            flash(f'Health Safety: You cannot donate yet. Please wait {days_left} more days.', 'warning')
            return redirect(url_for('donor_dashboard'))

    current_user.is_available = not current_user.is_available
    db.session.commit()
    return redirect(url_for('donor_dashboard'))

@app.route('/donor/certificate/<int:req_id>')
@login_required
def download_certificate(req_id):
    req = BloodRequest.query.get_or_404(req_id)

    if req.fulfilled_by_id != current_user.id:
        flash("You are not authorized to download this certificate.", "danger")
        return redirect(url_for('donor_dashboard'))    

    pdf = FPDF('L', 'mm', 'A4')

    pdf.set_auto_page_break(False) 
    
    pdf.add_page()

    pdf.set_line_width(2)
    pdf.set_draw_color(220, 53, 69) 
    pdf.rect(10, 10, 277, 190)

    pdf.set_y(35) 
    pdf.set_font('Arial', 'B', 36)
    pdf.set_text_color(220, 53, 69)
    pdf.cell(0, 10, 'CERTIFICATE OF LIFE SAVER', 0, 1, 'C')

    pdf.set_y(55)
    pdf.set_font('Arial', '', 14)
    pdf.set_text_color(50, 50, 50) 
    pdf.cell(0, 10, 'This certificate is proudly presented to', 0, 1, 'C')

    pdf.set_y(75)
    pdf.set_font('Arial', 'B', 32)
    pdf.set_text_color(0, 0, 0) 
    pdf.cell(0, 15, current_user.name.upper(), 0, 1, 'C')

    pdf.set_draw_color(100, 100, 100)
    pdf.set_line_width(0.5)
    name_width = pdf.get_string_width(current_user.name.upper()) + 30
    start_x = (297 - name_width) / 2
    pdf.line(start_x, 92, start_x + name_width, 92)

    pdf.set_y(105)
    pdf.set_font('Arial', '', 13)
    pdf.multi_cell(0, 8, f"For your selfless act of voluntarily donating 1 unit(s) of {req.blood_group} blood.\nYour generosity has given the gift of life to a patient in need.", 0, 'C')

    pdf.set_y(135)
    pdf.set_fill_color(240, 240, 240) 
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 12, f"Donated on: {req.created_at.strftime('%d-%B-%Y')}   |   Location: {req.hospital_address}", 0, 1, 'C', fill=True)

    pdf.set_y(165)
    
    pdf.set_x(40)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 10, f"Donor ID: #{current_user.donor_id}", 0, 0, 'L')

    pdf.set_x(190)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(60, 10, 'BloodAI Authority', 'T', 0, 'C') 

    pdf.set_y(192)
    pdf.set_font('Arial', '', 8)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(0, 5, 'This is a computer-generated document and needs no physical signature.', 0, 1, 'C')

    try:        
        pdf_data = pdf.output(dest='S')

        if isinstance(pdf_data, str):
            pdf_data = pdf_data.encode('latin-1')

        pdf_output = io.BytesIO(pdf_data)
        pdf_output.seek(0)
        safe_name = f"Certificate_{current_user.id}.pdf"
        
        return send_file(
            pdf_output,
            as_attachment=True,
            download_name=safe_name,
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return "Error generating PDF. Please try again later.", 500


#=====================================
# 9. ADMIN ROUTES
#=====================================
@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin': return redirect(url_for('home'))
     
    total_donors = User.query.filter_by(role='donor').count()
    active_reqs = BloodRequest.query.filter(BloodRequest.status.in_(['Active', 'Donated'])).count()
    fulfilled_reqs = BloodRequest.query.filter_by(status='Fulfilled').count()
    total_users = User.query.count()

    stats = {
        'total_users': total_users,
        'donors': total_donors,
        'active_requests': active_reqs,
        'fulfilled_requests': fulfilled_reqs
    }

    blood_groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
    
    donor_data = {bg: 0 for bg in blood_groups}
    req_data = {bg: 0 for bg in blood_groups}

    donors = User.query.filter_by(role='donor').all()
    for d in donors:
        if d.blood_group in donor_data:
            donor_data[d.blood_group] += 1

    requests = BloodRequest.query.all()
    for r in requests:
        if r.blood_group in req_data:
            req_data[r.blood_group] += 1

    labels = blood_groups
    donor_counts = [donor_data[bg] for bg in blood_groups]
    req_counts = [req_data[bg] for bg in blood_groups]

    return render_template('admin/dashboard.html', 
                           stats=stats, 
                           labels=labels, 
                           donor_counts=donor_counts, 
                           req_counts=req_counts)   

@app.route('/admin/inventory', methods=['GET', 'POST'])
@login_required
def admin_inventory():
    if current_user.role != 'admin': return redirect(url_for('home'))
    
    if request.method == 'POST':
        b_group = request.form['blood_group']
        action = request.form['action'] 
        amount = int(request.form['units'])
        
        stock = BloodStock.query.filter_by(blood_group=b_group).first()
        if stock:
            if action == 'add':
                stock.units += amount
            elif action == 'remove':
                stock.units = max(0, stock.units - amount)
            db.session.commit()
            flash(f'Stock updated for {b_group}', 'success')
        else:
            flash('Error: Blood group not found', 'danger')
            
    stocks = BloodStock.query.all()
    return render_template('admin/inventory.html', stocks=stocks)

@app.route('/admin/export/<data_type>')
@login_required
def export_data(data_type):
    if current_user.role != 'admin': return redirect(url_for('home'))
    
    si = io.StringIO()  
    cw = csv.writer(si)
    filename = f"{data_type}_report.csv"

    if data_type == 'users':
        cw.writerow(['ID', 'Name', 'Email', 'Role', 'Phone', 'Blood Group'])
        users = User.query.all()
        for u in users:
            cw.writerow([u.id, u.name, u.email, u.role, u.phone, u.blood_group or '-'])

    elif data_type == 'donors':
        cw.writerow(['ID', 'Name', 'Blood Group', 'Phone', 'Health Score', 'Last Donation', 'Location (Lat/Lon)'])
        donors = User.query.filter_by(role='donor').all()
        for d in donors:
            cw.writerow([
                d.id, d.name, d.blood_group, d.phone, d.health_score, 
                str(d.last_donation) if d.last_donation else 'Never',
                f"{d.lat}, {d.lon}"
            ])

    elif data_type == 'active':
        cw.writerow(['Request ID', 'Patient Name', 'Blood Group', 'Units', 'Urgency', 'Hospital', 'Contact', 'Requested By', 'Date'])
        reqs = BloodRequest.query.filter_by(status='Active').all()
        for r in reqs:
            requester = User.query.get(r.requester_id)
            req_name = requester.name if requester else "Unknown"
            cw.writerow([
                r.id, r.patient_name, r.blood_group, r.units, r.urgency, 
                r.hospital_address, r.contact_number, req_name, r.created_at
            ])
            
    elif data_type.startswith('attendees_'):

        camp_id = int(data_type.split('_')[1])
        camp = BloodCamp.query.get(camp_id)
        filename = f"Camp_{camp_id}_Attendees.csv"
        
        cw.writerow(['Name', 'Blood Group', 'Phone', 'Email', 'Registered Date'])
        regs = CampRegistration.query.filter_by(camp_id=camp_id).all()
        for r in regs:
            user = User.query.get(r.user_id)
            if user:
                cw.writerow([user.name, user.blood_group, user.phone, user.email, r.registered_at])       

    elif data_type == 'history':
        cw.writerow(['Request ID', 'Patient Name', 'Blood Group', 'Hospital', 'Requested By', 'FULFILLED BY (Donor)', 'Donor Phone', 'Date'])
        reqs = BloodRequest.query.filter_by(status='Fulfilled').all()
        for r in reqs:
            requester = User.query.get(r.requester_id)
            donor = User.query.get(r.fulfilled_by_id) if r.fulfilled_by_id else None
            
            req_name = requester.name if requester else "Unknown"
            donor_name = donor.name if donor else "Unknown Donor"
            donor_phone = donor.phone if donor else "-"
            
            cw.writerow([
                r.id, r.patient_name, r.blood_group, r.hospital_address, 
                req_name, donor_name, donor_phone, r.created_at
            ])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-type"] = "text/csv"
    return output

#====================================
# 10. CAMP ROUTES
#==================================== 
@app.route('/admin/camp/new', methods=['GET', 'POST'])
@login_required
def create_camp():
    if current_user.role != 'admin': return redirect(url_for('home'))
    
    if request.method == 'POST':
        new_camp = BloodCamp(
            title=request.form['title'],
            organizer=request.form['organizer'],
            location=request.form['location'],
            date=request.form['date'],
            time=request.form['time'],
            description=request.form['description']
        )
        db.session.add(new_camp)
        db.session.commit()
        flash('Blood Donation Camp Created Successfully! 🎉', 'success')
        return redirect(url_for('admin_dashboard'))
        
    return render_template('admin/add_camp.html')

@app.route('/camps')
def view_camps():
    camps = BloodCamp.query.order_by(BloodCamp.created_at.desc()).all()

    my_registrations = []
    if current_user.is_authenticated:
        regs = CampRegistration.query.filter_by(user_id=current_user.id).all()
        my_registrations = [r.camp_id for r in regs]
        
    return render_template('camps.html', camps=camps, my_camps=my_registrations)

@app.route('/camp/join/<int:camp_id>')
@login_required
def join_camp(camp_id):
    exists = CampRegistration.query.filter_by(user_id=current_user.id, camp_id=camp_id).first()
    
    if exists:
        flash('You are already registered for this camp.', 'info')
    else:
        new_reg = CampRegistration(user_id=current_user.id, camp_id=camp_id)
        db.session.add(new_reg)
        db.session.commit()
        flash('Registration Successful! See you at the camp. 🩸', 'success')
        
    return redirect(url_for('view_camps'))

@app.route('/admin/view/camps')
@login_required
def view_admin_camps():
    if current_user.role != 'admin': return redirect(url_for('home'))
    
    camps = BloodCamp.query.order_by(BloodCamp.created_at.desc()).all()
    
    display_data = []
    for c in camps:
      
        count = CampRegistration.query.filter_by(camp_id=c.id).count()
        display_data.append({
            'id': c.id,
            'title': c.title,
            'date': c.date,
            'time': c.time,
            'location': c.location,
            'organizer': c.organizer,
            'attendee_count': count
        })
    
    return render_template('admin/details.html', title="Manage Camps", data=display_data, type="camps_list")

@app.route('/admin/camp/<int:camp_id>/attendees')
@login_required
def view_camp_attendees(camp_id):
    if current_user.role != 'admin': return redirect(url_for('home'))
    
    camp = BloodCamp.query.get_or_404(camp_id)
    regs = CampRegistration.query.filter_by(camp_id=camp_id).all()
    
    display_data = []
    for r in regs:
        user = User.query.get(r.user_id)
        if user:
            display_data.append({
                'name': user.name,
                'blood_group': user.blood_group,
                'phone': user.phone,
                'email': user.email,
                'registered_at': r.registered_at
            })
            
    return render_template('admin/details.html', title=f"Attendees: {camp.title}", data=display_data, type="attendees", camp_id=camp_id)
    
@app.route('/admin/export_donors')
@login_required
def export_donors():
    if current_user.role != 'admin': return redirect(url_for('home'))
    
    donors = User.query.filter_by(role='donor').all()
    
    output = []
    output.append(['Name', 'Email', 'Phone', 'Blood Group', 'Health Score', 'Last Donation'])
    
    for d in donors:
        output.append([
            d.name, 
            d.email, 
            d.phone, 
            d.blood_group, 
            d.health_score, 
            str(d.last_donation) if d.last_donation else 'Never'
        ])
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerows(output)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=donors_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/admin/view/users')
@login_required
def view_users():
    if current_user.role != 'admin': return redirect(url_for('home'))
    users = User.query.all()
    return render_template('admin/details.html', title="All Users", data=users, type="users")

@app.route('/admin/view/donors')
@login_required
def view_donors():
    if current_user.role != 'admin': return redirect(url_for('home'))
    donors = User.query.filter_by(role='donor').all()
    return render_template('admin/details.html', title="Registered Donors", data=donors, type="donors")

@app.route('/admin/view/active_requests')
@login_required
def view_active_requests():
    if current_user.role != 'admin': return redirect(url_for('home'))
   
    reqs = BloodRequest.query.filter_by(status='Active').order_by(BloodRequest.created_at.desc()).all()

    display_data = []
    for r in reqs:
        requester = User.query.get(r.requester_id)
        display_data.append({
            'id': r.id,
            'patient_name': r.patient_name,
            'blood_group': r.blood_group,
            'requester_name': requester.name if requester else "Unknown",
            'hospital': r.hospital_address,
            'contact': r.contact_number,
            'date': r.created_at
        })
    return render_template('admin/details.html', title="Active Requests", data=display_data, type="active")

@app.route('/admin/view/history')
@login_required
def view_history():
    if current_user.role != 'admin': return redirect(url_for('home'))
    reqs = BloodRequest.query.filter_by(status='Fulfilled').order_by(BloodRequest.created_at.desc()).all()
    
    display_data = []
    for r in reqs:
        requester = User.query.get(r.requester_id)
        donor = User.query.get(r.fulfilled_by_id) if r.fulfilled_by_id else None
        
        display_data.append({
            'id': r.id,
            'patient_name': r.patient_name,
            'blood_group': r.blood_group,
            'requester_name': requester.name if requester else "Unknown",
            'donor_name': donor.name if donor else "Unknown Donor", 
            'donor_contact': donor.phone if donor else "-",
            'hospital': r.hospital_address,
            'date': r.created_at
        })
    return render_template('admin/details.html', title="Lives Saved (History)", data=display_data, type="history")    
    
#=====================================
# 11. BLOCKCHAIN & TOOLS
#=====================================
@app.route('/track_blood/<int:req_id>')
def track_blood_chain(req_id): 
    chain = BloodChain.query.filter_by(req_id=req_id).order_by(BloodChain.id.asc()).all()
    req = BloodRequest.query.get_or_404(req_id)
    return render_template('blockchain.html', chain=chain, request=req)

@app.route('/generate_qrcode')
@login_required
def generate_qrcode():

    qr_data = f"""
    🩸 BloodAI Donor Card
    -------------------
    ID: #{current_user.donor_id}
    Name: {current_user.name}
    Group: {current_user.blood_group}
    Phone: {current_user.phone}
    Status: {'✅ Available' if current_user.is_available else '💤 Unavailable'}
    """
    
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

@app.route('/generate_packet_qr/<int:req_id>')
def generate_packet_qr(req_id):
    
    link = url_for('track_blood_chain', req_id=req_id, _external=True)
    
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(link)
    qr.make(fit=True)
    img = qr.make_image(fill='darkred', back_color='white')
    
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

#=====================================
# 12. API ROUTES (LOCATION)
#=====================================
@app.route('/api/update_location', methods=['POST'])
@login_required
@limiter.exempt
def update_location():
    if current_user.role != 'donor': return {'status': 'error'}, 403
    
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if lat and lon:
        current_user.lat = lat
        current_user.lon = lon
        db.session.commit()
        return {'status': 'updated', 'lat': lat, 'lon': lon}
    return {'status': 'fail'}, 400

@app.route('/api/get_donor_location/<int:donor_id>')
@limiter.exempt
def get_donor_location(donor_id):
    donor = User.query.get_or_404(donor_id)
    return {
        'name': donor.name,
        'lat': donor.lat,
        'lon': donor.lon,
        'phone': donor.phone,
        'updated_at': str(datetime.datetime.now())
    }

#=====================================
# 13. SECURITY HEADERS
#=====================================
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'

    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
  
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response    

#=====================================
# 14. MAIN EXECUTION (Updated for Production)
#=====================================

def setup_app():
   
    with app.app_context():
        try:
            
            load_or_train_model()
            
            db.create_all() 
            
            admin_user = User.query.filter_by(role='admin').first()
            if not admin_user:
                pw = generate_password_hash('admin123', method='pbkdf2:sha256')
                admin = User(name='Admin', email='admin@bloodai.com', password=pw, role='admin')
                db.session.add(admin)
                db.session.commit()
            
            if BloodStock.query.count() == 0:
                groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
                for g in groups:
                    db.session.add(BloodStock(blood_group=g, units=0))
                db.session.commit()
            print("🚀 App Setup Completed (DB & ML Ready)")
        except Exception as e:
            print(f"❌ Setup Error: {e}")

setup_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)