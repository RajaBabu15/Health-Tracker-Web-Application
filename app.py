"""
Health Tracker Web Application - Flask Backend API

A comprehensive health tracking application with user authentication, 
health metric management, wearable device integration, and analytics.
"""

import os
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import secrets
import json
import uuid
from dataclasses import dataclass
from functools import wraps

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, verify_jwt_in_request, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import BadRequest, Unauthorized, NotFound
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, and_, or_, text

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from apscheduler.schedulers.background import BackgroundScheduler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask extensions (will be configured in create_app)
db = SQLAlchemy()
jwt = JWTManager()
limiter = Limiter(key_func=get_remote_address)

# Database Models
class User(db.Model):
    """User account model with authentication and profile info"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    health_metrics = db.relationship('HealthMetric', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    device_connections = db.relationship('DeviceConnection', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password: str):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Check if password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding sensitive info)"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }

class HealthMetric(db.Model):
    """Health metrics data model"""
    __tablename__ = 'health_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    metric_type = db.Column(db.String(50), nullable=False, index=True)
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    source = db.Column(db.String(50), default='manual')  # manual, fitbit, apple_watch, garmin, etc.
    notes = db.Column(db.Text)
    
    # Add composite index for efficient queries
    __table_args__ = (
        db.Index('idx_user_metric_date', 'user_id', 'metric_type', 'recorded_at'),
    )
    
    VALID_METRICS = {
        'weight': {'units': ['kg', 'lbs'], 'min_value': 0, 'max_value': 1000},
        'height': {'units': ['cm', 'inches'], 'min_value': 0, 'max_value': 300},
        'heart_rate': {'units': ['bpm'], 'min_value': 30, 'max_value': 250},
        'blood_pressure': {'units': ['mmHg'], 'min_value': 50, 'max_value': 300},
        'steps': {'units': ['count'], 'min_value': 0, 'max_value': 100000},
        'sleep_hours': {'units': ['hours'], 'min_value': 0, 'max_value': 24},
        'water_intake': {'units': ['ml', 'oz'], 'min_value': 0, 'max_value': 10000},
        'calories': {'units': ['kcal'], 'min_value': 0, 'max_value': 10000},
        'exercise_minutes': {'units': ['minutes'], 'min_value': 0, 'max_value': 1440},
        'body_temperature': {'units': ['celsius', 'fahrenheit'], 'min_value': 30, 'max_value': 45},
    }
    
    def validate_metric(self) -> bool:
        """Validate metric type, value, and unit"""
        if self.metric_type not in self.VALID_METRICS:
            return False
        
        metric_config = self.VALID_METRICS[self.metric_type]
        
        # Check unit
        if self.unit not in metric_config['units']:
            return False
        
        # Check value range
        if not (metric_config['min_value'] <= self.value <= metric_config['max_value']):
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health metric to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'metric_type': self.metric_type,
            'value': self.value,
            'unit': self.unit,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
            'source': self.source,
            'notes': self.notes
        }

class DeviceConnection(db.Model):
    """Wearable device connections and sync history"""
    __tablename__ = 'device_connections'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    device_type = db.Column(db.String(50), nullable=False)  # fitbit, apple_watch, garmin, etc.
    device_id = db.Column(db.String(100))
    access_token = db.Column(db.String(500))  # Encrypted in production
    refresh_token = db.Column(db.String(500))  # Encrypted in production
    connected_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_sync = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    sync_frequency = db.Column(db.Integer, default=60)  # minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert device connection to dictionary"""
        return {
            'id': self.id,
            'device_type': self.device_type,
            'device_id': self.device_id,
            'connected_at': self.connected_at.isoformat() if self.connected_at else None,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'is_active': self.is_active,
            'sync_frequency': self.sync_frequency
        }

# Validation Functions
def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

# Authentication Decorators
def jwt_required_custom(f):
    """Custom JWT required decorator with better error handling"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            return jsonify({'error': 'Invalid or missing token'}), 401
    return decorated

# Device Integration Services (Stubs)
class DeviceSyncService:
    """Service for syncing data from wearable devices"""
    
    @staticmethod
    def sync_fitbit_data(user_id: int, access_token: str) -> Dict[str, Any]:
        """Sync data from Fitbit API (stub implementation)"""
        logger.info(f"Syncing Fitbit data for user {user_id}")
        
        # Stub: Simulate successful sync with mock data
        mock_metrics = [
            {'metric_type': 'steps', 'value': 8547, 'unit': 'count', 'source': 'fitbit'},
            {'metric_type': 'heart_rate', 'value': 72, 'unit': 'bpm', 'source': 'fitbit'},
            {'metric_type': 'sleep_hours', 'value': 7.5, 'unit': 'hours', 'source': 'fitbit'},
            {'metric_type': 'calories', 'value': 2340, 'unit': 'kcal', 'source': 'fitbit'}
        ]
        
        # Save metrics to database
        synced_count = 0
        for metric_data in mock_metrics:
            metric = HealthMetric(
                user_id=user_id,
                metric_type=metric_data['metric_type'],
                value=metric_data['value'],
                unit=metric_data['unit'],
                source=metric_data['source'],
                notes=f"Synced from Fitbit at {datetime.utcnow()}"
            )
            if metric.validate_metric():
                db.session.add(metric)
                synced_count += 1
        
        try:
            db.session.commit()
            logger.info(f"Successfully synced {synced_count} Fitbit metrics for user {user_id}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving Fitbit sync data: {str(e)}")
            return {'status': 'error', 'message': 'Database error during sync'}
        
        return {
            'status': 'success',
            'message': f'Successfully synced {synced_count} metrics from fitbit',
            'synced_metrics': synced_count,
            'last_sync': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def sync_apple_watch_data(user_id: int, access_token: str) -> Dict[str, Any]:
        """Sync data from Apple HealthKit (stub implementation)"""
        logger.info(f"Syncing Apple Watch data for user {user_id}")
        
        # Stub: Simulate successful sync with mock data
        mock_metrics = [
            {'metric_type': 'steps', 'value': 9234, 'unit': 'count', 'source': 'apple_watch'},
            {'metric_type': 'heart_rate', 'value': 68, 'unit': 'bpm', 'source': 'apple_watch'},
            {'metric_type': 'exercise_minutes', 'value': 45, 'unit': 'minutes', 'source': 'apple_watch'},
            {'metric_type': 'calories', 'value': 2450, 'unit': 'kcal', 'source': 'apple_watch'}
        ]
        
        # Save metrics to database
        synced_count = 0
        for metric_data in mock_metrics:
            metric = HealthMetric(
                user_id=user_id,
                metric_type=metric_data['metric_type'],
                value=metric_data['value'],
                unit=metric_data['unit'],
                source=metric_data['source'],
                notes=f"Synced from Apple Watch at {datetime.utcnow()}"
            )
            if metric.validate_metric():
                db.session.add(metric)
                synced_count += 1
        
        try:
            db.session.commit()
            logger.info(f"Successfully synced {synced_count} Apple Watch metrics for user {user_id}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving Apple Watch sync data: {str(e)}")
            return {'status': 'error', 'message': 'Database error during sync'}
        
        return {
            'status': 'success',
            'message': f'Successfully synced {synced_count} metrics from apple_watch',
            'synced_metrics': synced_count,
            'last_sync': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def sync_garmin_data(user_id: int, access_token: str) -> Dict[str, Any]:
        """Sync data from Garmin Connect (stub implementation)"""
        logger.info(f"Syncing Garmin data for user {user_id}")
        
        # Stub: Simulate successful sync with mock data
        mock_metrics = [
            {'metric_type': 'steps', 'value': 7832, 'unit': 'count', 'source': 'garmin'},
            {'metric_type': 'heart_rate', 'value': 75, 'unit': 'bpm', 'source': 'garmin'},
            {'metric_type': 'sleep_hours', 'value': 8.2, 'unit': 'hours', 'source': 'garmin'},
            {'metric_type': 'exercise_minutes', 'value': 60, 'unit': 'minutes', 'source': 'garmin'}
        ]
        
        # Save metrics to database
        synced_count = 0
        for metric_data in mock_metrics:
            metric = HealthMetric(
                user_id=user_id,
                metric_type=metric_data['metric_type'],
                value=metric_data['value'],
                unit=metric_data['unit'],
                source=metric_data['source'],
                notes=f"Synced from Garmin at {datetime.utcnow()}"
            )
            if metric.validate_metric():
                db.session.add(metric)
                synced_count += 1
        
        try:
            db.session.commit()
            logger.info(f"Successfully synced {synced_count} Garmin metrics for user {user_id}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving Garmin sync data: {str(e)}")
            return {'status': 'error', 'message': 'Database error during sync'}
        
        return {
            'status': 'success',
            'message': f'Successfully synced {synced_count} metrics from garmin',
            'synced_metrics': synced_count,
            'last_sync': datetime.utcnow().isoformat()
        }

# Analytics and Visualization
class HealthAnalytics:
    """Service for generating health insights and visualizations"""
    
    @staticmethod
    def generate_chart(user_id: int, metric_type: str, days: int = 30) -> Optional[str]:
        """Generate a chart for health metrics"""
        try:
            # Query metrics for the specified period
            start_date = datetime.utcnow() - timedelta(days=days)
            metrics = HealthMetric.query.filter(
                and_(
                    HealthMetric.user_id == user_id,
                    HealthMetric.metric_type == metric_type,
                    HealthMetric.recorded_at >= start_date
                )
            ).order_by(HealthMetric.recorded_at.asc()).all()
            
            if not metrics:
                logger.warning(f"No data found for user {user_id}, metric {metric_type}")
                return None
            
            # Convert to pandas DataFrame
            data = pd.DataFrame([m.to_dict() for m in metrics])
            data['recorded_at'] = pd.to_datetime(data['recorded_at'])
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(data['recorded_at'], data['value'], marker='o', linewidth=2, markersize=4)
            plt.title(f'{metric_type.replace("_", " ").title()} Over Time')
            plt.xlabel('Date')
            plt.ylabel(f'{metric_type.replace("_", " ").title()} ({data["unit"].iloc[0]})')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save chart
            charts_dir = os.path.join(os.path.dirname(__file__), 'static', 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            filename = f"{user_id}_{metric_type}_{days}days_{int(datetime.utcnow().timestamp())}.png"
            filepath = os.path.join(charts_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated chart for user {user_id}: {filename}")
            return f'/static/charts/{filename}'
            
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            return None

# Application Factory
def create_app(config_name: Optional[str] = None) -> tuple[Flask, SQLAlchemy]:
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL', 
        'sqlite:///health_tracker.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
    
    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    limiter.init_app(app)
    CORS(app)
    
    # Ensure static directories exist
    os.makedirs(os.path.join(app.root_path, 'static', 'charts'), exist_ok=True)
    
    # API Routes
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        })
    
    # Authentication Routes
    @app.route('/api/auth/register', methods=['POST'])
    @limiter.limit("5 per minute")
    def register():
        """User registration endpoint"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            name = data.get('name', '').strip()
            
            # Validation
            if not all([email, password, name]):
                return jsonify({'error': 'Email, password, and name are required'}), 400
            
            if not validate_email(email):
                return jsonify({'error': 'Invalid email format'}), 400
            
            is_valid_password, password_message = validate_password(password)
            if not is_valid_password:
                return jsonify({'error': password_message}), 400
            
            # Check if user exists
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already registered'}), 400
            
            # Create user
            user = User(email=email, name=name)
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            logger.info(f"New user registered: {email}")
            
            return jsonify({
                'message': 'User created successfully',
                'user_id': user.id
            }), 201
            
        except IntegrityError:
            db.session.rollback()
            return jsonify({'error': 'Email already registered'}), 400
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/api/auth/login', methods=['POST'])
    @limiter.limit("10 per minute")
    def login():
        """User login endpoint"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            
            if not all([email, password]):
                return jsonify({'error': 'Email and password are required'}), 400
            
            user = User.query.filter_by(email=email).first()
            
            if not user or not user.check_password(password):
                return jsonify({'error': 'Invalid email or password'}), 401
            
            if not user.is_active:
                return jsonify({'error': 'Account is deactivated'}), 401
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # Create access token
            access_token = create_access_token(identity=str(user.id))
            
            logger.info(f"User logged in: {email}")
            
            return jsonify({
                'access_token': access_token,
                'user_id': user.id,
                'user': user.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # Health Metrics Routes
    @app.route('/api/metrics', methods=['POST'])
    @jwt_required_custom
    @limiter.limit("30 per minute")
    def create_metric():
        """Create a new health metric"""
        try:
            user_id = int(get_jwt_identity())
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            metric_type = data.get('metric_type')
            value = data.get('value')
            unit = data.get('unit')
            notes = data.get('notes', '')
            
            if not all([metric_type, value is not None, unit]):
                return jsonify({'error': 'metric_type, value, and unit are required'}), 400
            
            try:
                value = float(value)
            except (ValueError, TypeError):
                return jsonify({'error': 'Value must be a number'}), 400
            
            # Create and validate metric
            metric = HealthMetric(
                user_id=user_id,
                metric_type=metric_type,
                value=value,
                unit=unit,
                notes=notes,
                source='manual'
            )
            
            if not metric.validate_metric():
                return jsonify({'error': 'Invalid metric type, value, or unit'}), 400
            
            db.session.add(metric)
            db.session.commit()
            
            logger.info(f"New metric created by user {user_id}: {metric_type}")
            
            return jsonify({
                'message': 'Metric created successfully',
                'id': metric.id,
                'metric': metric.to_dict()
            }), 201
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Create metric error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/api/metrics', methods=['GET'])
    @jwt_required_custom
    def get_metrics():
        """Get user's health metrics with optional filtering"""
        try:
            user_id = int(get_jwt_identity())
            
            # Query parameters
            metric_type = request.args.get('metric_type')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            limit = request.args.get('limit', 100, type=int)
            
            query = HealthMetric.query.filter_by(user_id=user_id)
            
            if metric_type:
                query = query.filter_by(metric_type=metric_type)
            
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    query = query.filter(HealthMetric.recorded_at >= start_dt)
                except ValueError:
                    return jsonify({'error': 'Invalid start_date format'}), 400
            
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    query = query.filter(HealthMetric.recorded_at <= end_dt)
                except ValueError:
                    return jsonify({'error': 'Invalid end_date format'}), 400
            
            metrics = query.order_by(HealthMetric.recorded_at.desc()).limit(limit).all()
            
            return jsonify([metric.to_dict() for metric in metrics])
            
        except Exception as e:
            logger.error(f"Get metrics error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # Device Sync Routes
    @app.route('/api/sync/devices', methods=['POST'])
    @jwt_required_custom
    @limiter.limit("5 per minute")
    def sync_device():
        """Sync data from wearable devices"""
        try:
            user_id = int(get_jwt_identity())
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            device_type = data.get('device_type')
            auth_token = data.get('auth_token')
            
            if not all([device_type, auth_token]):
                return jsonify({'error': 'device_type and auth_token are required'}), 400
            
            # Validate device type
            supported_devices = ['fitbit', 'apple_watch', 'garmin']
            if device_type not in supported_devices:
                return jsonify({'error': f'Unsupported device type. Supported: {supported_devices}'}), 400
            
            # Call appropriate sync service
            if device_type == 'fitbit':
                result = DeviceSyncService.sync_fitbit_data(user_id, auth_token)
            elif device_type == 'apple_watch':
                result = DeviceSyncService.sync_apple_watch_data(user_id, auth_token)
            elif device_type == 'garmin':
                result = DeviceSyncService.sync_garmin_data(user_id, auth_token)
            
            logger.info(f"Device sync completed for user {user_id}: {device_type}")
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Device sync error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # Dashboard and Analytics Routes
    @app.route('/api/dashboard/chart', methods=['GET'])
    @jwt_required_custom
    def generate_chart():
        """Generate charts for health metrics"""
        try:
            user_id = int(get_jwt_identity())
            
            metric_type = request.args.get('metric_type')
            days = request.args.get('days', 30, type=int)
            
            if not metric_type:
                return jsonify({'error': 'metric_type parameter is required'}), 400
            
            # Validate metric type
            if metric_type not in HealthMetric.VALID_METRICS:
                return jsonify({'error': f'Invalid metric type: {metric_type}'}), 400
            
            chart_url = HealthAnalytics.generate_chart(user_id, metric_type, days)
            
            if not chart_url:
                return jsonify({'error': 'No data available for chart generation'}), 404
            
            return jsonify({
                'chart_url': chart_url,
                'metric_type': metric_type,
                'days': days,
                'generated_at': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Chart generation error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # Static file serving
    @app.route('/static/charts/<filename>')
    def serve_chart(filename):
        """Serve generated chart files"""
        charts_dir = os.path.join(app.root_path, 'static', 'charts')
        return send_from_directory(charts_dir, filename)
    
    # Error Handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({'error': 'Unauthorized'}), 401
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500
    
    # Initialize database and create demo data
    with app.app_context():
        db.create_all()
        
        # Create demo user if it doesn't exist
        demo_user = User.query.filter_by(email='demo@healthtracker.com').first()
        if not demo_user:
            demo_user = User(
                email='demo@healthtracker.com',
                name='Demo User'
            )
            demo_user.set_password('password')
            db.session.add(demo_user)
            
            # Add some sample metrics
            sample_metrics = [
                HealthMetric(user_id=1, metric_type='weight', value=70.5, unit='kg'),
                HealthMetric(user_id=1, metric_type='heart_rate', value=72, unit='bpm'),
                HealthMetric(user_id=1, metric_type='steps', value=8500, unit='count'),
                HealthMetric(user_id=1, metric_type='sleep_hours', value=7.5, unit='hours'),
            ]
            
            for metric in sample_metrics:
                db.session.add(metric)
            
            try:
                db.session.commit()
                logger.info("Demo user and sample data created")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error creating demo data: {str(e)}")
    
    return app, db

# Main application entry point
if __name__ == '__main__':
    app, db_instance = create_app()
    
    # Development server
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=debug_mode
    )
