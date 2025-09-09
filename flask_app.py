"""
🏥 Health Tracker Web Application - Flask Implementation
A comprehensive health tracking platform with user authentication, 
dashboard, manual data input, and wearable device integration.

Tech Stack: Flask, Pandas, NumPy, matplotlib, SQLAlchemy
Features: User registration, authentication, health metrics dashboard,
         manual input, wearable sync, progress tracking over time
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import random
from pathlib import Path
import os

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'health-tracker-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access your health dashboard.'

# Database Models
class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    height = db.Column(db.Float)  # in cm
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    health_metrics = db.relationship('HealthMetric', backref='user', lazy=True)
    wearable_data = db.relationship('WearableData', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class HealthMetric(db.Model):
    """Health metrics manually entered by users"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    metric_type = db.Column(db.String(50), nullable=False)  # weight, blood_pressure, heart_rate, etc.
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=False)
    notes = db.Column(db.Text)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # For blood pressure (systolic/diastolic)
    value_secondary = db.Column(db.Float)  # for diastolic BP

class WearableData(db.Model):
    """Data synced from wearable devices"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    device_type = db.Column(db.String(50), nullable=False)  # fitbit, apple_watch, garmin
    data_type = db.Column(db.String(50), nullable=False)  # steps, heart_rate, sleep, calories
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=False)
    sync_time = db.Column(db.DateTime, default=datetime.utcnow)
    date_recorded = db.Column(db.Date, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper Functions
def generate_sample_wearable_data(user_id, days=30):
    """Generate sample wearable data for demonstration"""
    device_types = ['fitbit', 'apple_watch', 'garmin']
    data_types = ['steps', 'heart_rate', 'calories_burned', 'sleep_hours', 'active_minutes']
    
    sample_data = []
    end_date = datetime.now().date()
    
    for day in range(days):
        current_date = end_date - timedelta(days=day)
        device = random.choice(device_types)
        
        # Generate realistic health data
        daily_data = {
            'steps': random.randint(3000, 15000),
            'heart_rate': random.randint(60, 100),
            'calories_burned': random.randint(1200, 3500),
            'sleep_hours': round(random.uniform(5.5, 9.5), 1),
            'active_minutes': random.randint(20, 120)
        }
        
        for data_type, value in daily_data.items():
            unit_map = {
                'steps': 'steps',
                'heart_rate': 'bpm',
                'calories_burned': 'cal',
                'sleep_hours': 'hours',
                'active_minutes': 'minutes'
            }
            
            sample_data.append(WearableData(
                user_id=user_id,
                device_type=device,
                data_type=data_type,
                value=value,
                unit=unit_map[data_type],
                date_recorded=current_date
            ))
    
    return sample_data

def create_dashboard_charts(user_id):
    """Create interactive charts for the dashboard using Plotly"""
    charts = {}
    
    # Get recent data (last 30 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    # Steps chart
    steps_data = WearableData.query.filter(
        WearableData.user_id == user_id,
        WearableData.data_type == 'steps',
        WearableData.date_recorded >= start_date
    ).order_by(WearableData.date_recorded).all()
    
    if steps_data:
        dates = [data.date_recorded for data in steps_data]
        steps = [data.value for data in steps_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=steps, mode='lines+markers', name='Daily Steps',
                                line=dict(color='#1f77b4', width=3)))
        fig.update_layout(title='Daily Steps - Last 30 Days', xaxis_title='Date', yaxis_title='Steps',
                         template='plotly_white', height=400)
        charts['steps'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    # Heart Rate chart
    hr_data = WearableData.query.filter(
        WearableData.user_id == user_id,
        WearableData.data_type == 'heart_rate',
        WearableData.date_recorded >= start_date
    ).order_by(WearableData.date_recorded).all()
    
    if hr_data:
        dates = [data.date_recorded for data in hr_data]
        heart_rates = [data.value for data in hr_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=heart_rates, mode='lines+markers', name='Resting Heart Rate',
                                line=dict(color='#ff7f0e', width=3)))
        fig.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Normal Lower")
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Normal Upper")
        fig.update_layout(title='Heart Rate Trends - Last 30 Days', xaxis_title='Date', yaxis_title='BPM',
                         template='plotly_white', height=400)
        charts['heart_rate'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    # Weight tracking (from manual entries)
    weight_data = HealthMetric.query.filter(
        HealthMetric.user_id == user_id,
        HealthMetric.metric_type == 'weight',
        HealthMetric.recorded_at >= datetime.combine(start_date, datetime.min.time())
    ).order_by(HealthMetric.recorded_at).all()
    
    if weight_data:
        dates = [data.recorded_at.date() for data in weight_data]
        weights = [data.value for data in weight_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=weights, mode='lines+markers', name='Weight',
                                line=dict(color='#2ca02c', width=3)))
        fig.update_layout(title='Weight Progress - Last 30 Days', xaxis_title='Date', yaxis_title='Weight (kg)',
                         template='plotly_white', height=400)
        charts['weight'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    # Sleep hours chart
    sleep_data = WearableData.query.filter(
        WearableData.user_id == user_id,
        WearableData.data_type == 'sleep_hours',
        WearableData.date_recorded >= start_date
    ).order_by(WearableData.date_recorded).all()
    
    if sleep_data:
        dates = [data.date_recorded for data in sleep_data]
        sleep_hours = [data.value for data in sleep_data]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=sleep_hours, name='Sleep Hours',
                            marker_color='#9467bd'))
        fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Recommended: 7-9 hours")
        fig.add_hline(y=9, line_dash="dash", line_color="green")
        fig.update_layout(title='Sleep Patterns - Last 30 Days', xaxis_title='Date', yaxis_title='Hours',
                         template='plotly_white', height=400)
        charts['sleep'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    return charts

def get_health_summary(user_id):
    """Get summary statistics for dashboard"""
    summary = {}
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    # Steps summary
    steps_data = WearableData.query.filter(
        WearableData.user_id == user_id,
        WearableData.data_type == 'steps',
        WearableData.date_recorded >= start_date
    ).all()
    
    if steps_data:
        steps_values = [data.value for data in steps_data]
        summary['avg_steps'] = int(np.mean(steps_values))
        summary['total_steps'] = int(np.sum(steps_values))
        summary['step_goal_achievement'] = len([s for s in steps_values if s >= 10000])
    
    # Weight change
    recent_weight = HealthMetric.query.filter(
        HealthMetric.user_id == user_id,
        HealthMetric.metric_type == 'weight'
    ).order_by(HealthMetric.recorded_at.desc()).first()
    
    if recent_weight:
        summary['current_weight'] = recent_weight.value
        summary['weight_unit'] = recent_weight.unit
    
    # Sleep average
    sleep_data = WearableData.query.filter(
        WearableData.user_id == user_id,
        WearableData.data_type == 'sleep_hours',
        WearableData.date_recorded >= start_date
    ).all()
    
    if sleep_data:
        sleep_values = [data.value for data in sleep_data]
        summary['avg_sleep'] = round(np.mean(sleep_values), 1)
    
    return summary

# Routes
@app.route('/')
def index():
    """Home page - redirect to dashboard if logged in, otherwise show landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Generate sample wearable data for demo
        sample_data = generate_sample_wearable_data(user.id)
        for data in sample_data:
            db.session.add(data)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main health dashboard with charts and summary"""
    charts = create_dashboard_charts(current_user.id)
    summary = get_health_summary(current_user.id)
    
    return render_template('dashboard.html', 
                         user=current_user, 
                         charts=charts, 
                         summary=summary)

@app.route('/input_health', methods=['GET', 'POST'])
@login_required
def input_health():
    """Manual health data input form"""
    if request.method == 'POST':
        metric_type = request.form['metric_type']
        value = float(request.form['value'])
        unit = request.form['unit']
        notes = request.form.get('notes', '')
        
        # Handle blood pressure (systolic/diastolic)
        value_secondary = None
        if metric_type == 'blood_pressure':
            value_secondary = float(request.form['value_secondary'])
        
        health_metric = HealthMetric(
            user_id=current_user.id,
            metric_type=metric_type,
            value=value,
            value_secondary=value_secondary,
            unit=unit,
            notes=notes
        )
        
        db.session.add(health_metric)
        db.session.commit()
        
        flash(f'{metric_type.replace("_", " ").title()} recorded successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('input_health.html')

@app.route('/sync_wearable', methods=['GET', 'POST'])
@login_required
def sync_wearable():
    """Simulate wearable device synchronization"""
    if request.method == 'POST':
        device_type = request.form['device_type']
        
        # Simulate data sync by generating today's data
        today = datetime.now().date()
        
        # Generate realistic data for today
        sync_data = {
            'steps': random.randint(3000, 15000),
            'heart_rate': random.randint(60, 100),
            'calories_burned': random.randint(1200, 3500),
            'sleep_hours': round(random.uniform(5.5, 9.5), 1),
            'active_minutes': random.randint(20, 120)
        }
        
        unit_map = {
            'steps': 'steps',
            'heart_rate': 'bpm',
            'calories_burned': 'cal',
            'sleep_hours': 'hours',
            'active_minutes': 'minutes'
        }
        
        # Remove existing data for today
        WearableData.query.filter(
            WearableData.user_id == current_user.id,
            WearableData.date_recorded == today,
            WearableData.device_type == device_type
        ).delete()
        
        # Add new synced data
        for data_type, value in sync_data.items():
            wearable_data = WearableData(
                user_id=current_user.id,
                device_type=device_type,
                data_type=data_type,
                value=value,
                unit=unit_map[data_type],
                date_recorded=today
            )
            db.session.add(wearable_data)
        
        db.session.commit()
        
        flash(f'Successfully synced data from {device_type}!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('sync_wearable.html')

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html', user=current_user)

@app.route('/history')
@login_required
def history():
    """View historical health data"""
    # Get manual entries
    manual_data = HealthMetric.query.filter_by(user_id=current_user.id)\
                                   .order_by(HealthMetric.recorded_at.desc())\
                                   .limit(50).all()
    
    # Get recent wearable data
    wearable_data = WearableData.query.filter_by(user_id=current_user.id)\
                                     .order_by(WearableData.sync_time.desc())\
                                     .limit(100).all()
    
    return render_template('history.html', 
                         manual_data=manual_data, 
                         wearable_data=wearable_data)

# API Endpoints for AJAX requests
@app.route('/api/health_data/<metric_type>')
@login_required
def api_health_data(metric_type):
    """API endpoint to get health data for charts"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    if metric_type in ['steps', 'heart_rate', 'sleep_hours', 'calories_burned']:
        data = WearableData.query.filter(
            WearableData.user_id == current_user.id,
            WearableData.data_type == metric_type,
            WearableData.date_recorded >= start_date
        ).order_by(WearableData.date_recorded).all()
        
        return jsonify([{
            'date': d.date_recorded.isoformat(),
            'value': d.value,
            'unit': d.unit
        } for d in data])
    
    else:
        data = HealthMetric.query.filter(
            HealthMetric.user_id == current_user.id,
            HealthMetric.metric_type == metric_type,
            HealthMetric.recorded_at >= datetime.combine(start_date, datetime.min.time())
        ).order_by(HealthMetric.recorded_at).all()
        
        return jsonify([{
            'date': d.recorded_at.date().isoformat(),
            'value': d.value,
            'value_secondary': d.value_secondary,
            'unit': d.unit
        } for d in data])

# Template creation functions (since we can't create template files)
def create_templates():
    """Create basic HTML templates as strings (in real app, these would be separate files)"""
    pass

# Initialize database
def init_db():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    init_db()
    
    print("🏥 Health Tracker Web Application Starting...")
    print("📊 Features: User Auth, Dashboard, Manual Input, Wearable Sync")
    print("🔗 Visit: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
