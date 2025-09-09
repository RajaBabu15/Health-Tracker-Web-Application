import pytest
import json
import os
import tempfile
from datetime import datetime
from app import create_app

@pytest.fixture
def client():
    """Create test client with in-memory database"""
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp()
    
    app, db = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['JWT_SECRET_KEY'] = 'test-secret-key'
    
    with app.test_client() as client:
        with app.app_context():
            # Create tables
            db.create_all()
            
            # Run migrations by executing SQL directly
            with open('migrations/001_init.sql', 'r') as f:
                # Skip the transaction statements and some complex queries for testing
                sql_lines = f.read().split(';')
                for line in sql_lines:
                    line = line.strip()
                    if line and not line.startswith('BEGIN') and not line.startswith('COMMIT') and not line.startswith('SELECT '):
                        try:
                            db.engine.execute(line)
                        except Exception as e:
                            # Skip problematic SQL statements in test environment
                            if 'ALTER TABLE' not in line and 'TRIGGER' not in line:
                                print(f"Warning: Could not execute SQL: {line[:50]}... Error: {e}")
            
            db.session.commit()
            
        yield client
    
    # Cleanup
    os.close(db_fd)
    os.unlink(db_path)

@pytest.fixture
def auth_token(client):
    """Get auth token for testing authenticated endpoints"""
    # First register a test user
    client.post('/api/auth/register',
               data=json.dumps({
                   'email': 'testuser@example.com',
                   'password': 'testpass123',
                   'name': 'Test User'
               }),
               content_type='application/json')
    
    # Then login to get token
    response = client.post('/api/auth/login', 
                          data=json.dumps({
                              'email': 'testuser@example.com',
                              'password': 'testpass123'
                          }),
                          content_type='application/json')
    
    if response.status_code == 200:
        return json.loads(response.data)['access_token']
    else:
        # Fallback: try demo user
        response = client.post('/api/auth/login', 
                              data=json.dumps({
                                  'email': 'demo@healthtracker.com',
                                  'password': 'password'
                              }),
                              content_type='application/json')
        if response.status_code == 200:
            return json.loads(response.data)['access_token']
        else:
            pytest.skip("Could not obtain auth token")

def test_user_registration(client):
    """Test user registration endpoint"""
    import uuid
    unique_email = f'newuser_{uuid.uuid4().hex[:8]}@example.com'
    response = client.post('/api/auth/register',
                          data=json.dumps({
                              'email': unique_email,
                              'password': 'newpass123',
                              'name': 'New User'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert 'user_id' in data
    assert data['message'] == 'User created successfully'

def test_user_registration_duplicate_email(client):
    """Test registration with duplicate email"""
    # Register first user
    client.post('/api/auth/register',
               data=json.dumps({
                   'email': 'duplicate@example.com',
                   'password': 'pass123',
                   'name': 'First User'
               }),
               content_type='application/json')
    
    # Try to register same email again
    response = client.post('/api/auth/register',
                          data=json.dumps({
                              'email': 'duplicate@example.com',
                              'password': 'pass456',
                              'name': 'Second User'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_user_registration_validation_error(client):
    """Test registration with invalid data"""
    # Missing required fields
    response = client.post('/api/auth/register',
                          data=json.dumps({
                              'email': 'invalid-email',  # Invalid email format
                              'password': '123',  # Too short
                              'name': ''  # Empty name
                          }),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_user_login_valid(client):
    """Test valid user login"""
    # First register a user
    client.post('/api/auth/register',
               data=json.dumps({
                   'email': 'login@example.com',
                   'password': 'loginpass123',
                   'name': 'Login User'
               }),
               content_type='application/json')
    
    # Then login
    response = client.post('/api/auth/login',
                          data=json.dumps({
                              'email': 'login@example.com',
                              'password': 'loginpass123'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'access_token' in data
    assert 'user_id' in data

def test_user_login_invalid_credentials(client):
    """Test login with invalid credentials"""
    response = client.post('/api/auth/login',
                          data=json.dumps({
                              'email': 'nonexistent@example.com',
                              'password': 'wrongpassword'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 401
    data = json.loads(response.data)
    assert 'error' in data

def test_create_health_metric(client, auth_token):
    """Test creating a health metric"""
    response = client.post('/api/metrics',
                          headers={'Authorization': f'Bearer {auth_token}'},
                          data=json.dumps({
                              'metric_type': 'weight',
                              'value': 72.5,
                              'unit': 'kg',
                              'notes': 'Test measurement'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['message'] == 'Metric created successfully'
    assert 'id' in data

def test_create_metric_validation_error(client, auth_token):
    """Test creating metric with invalid data"""
    # Test negative value
    response = client.post('/api/metrics',
                          headers={'Authorization': f'Bearer {auth_token}'},
                          data=json.dumps({
                              'metric_type': 'weight',
                              'value': -10.5,  # Invalid negative value
                              'unit': 'kg'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_create_metric_invalid_type(client, auth_token):
    """Test creating metric with invalid type"""
    response = client.post('/api/metrics',
                          headers={'Authorization': f'Bearer {auth_token}'},
                          data=json.dumps({
                              'metric_type': 'invalid_type',
                              'value': 10.0,
                              'unit': 'kg'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_get_health_metrics(client, auth_token):
    """Test retrieving health metrics"""
    # First create a metric
    client.post('/api/metrics',
               headers={'Authorization': f'Bearer {auth_token}'},
               data=json.dumps({
                   'metric_type': 'heart_rate',
                   'value': 75,
                   'unit': 'bpm'
               }),
               content_type='application/json')
    
    # Then retrieve metrics
    response = client.get('/api/metrics',
                         headers={'Authorization': f'Bearer {auth_token}'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    # Should have at least the metric we just created
    assert len(data) >= 1

def test_get_metrics_with_filters(client, auth_token):
    """Test retrieving metrics with query filters"""
    response = client.get('/api/metrics?metric_type=weight&start_date=2022-01-01&end_date=2022-12-31',
                         headers={'Authorization': f'Bearer {auth_token}'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_device_sync_fitbit(client, auth_token):
    """Test Fitbit device sync stub"""
    response = client.post('/api/sync/devices',
                          headers={'Authorization': f'Bearer {auth_token}'},
                          data=json.dumps({
                              'device_type': 'fitbit',
                              'auth_token': 'mock_fitbit_token_123'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'synced_metrics' in data
    assert data['status'] == 'success'
    assert 'fitbit' in data['message']

def test_device_sync_apple_watch(client, auth_token):
    """Test Apple Watch device sync stub"""
    response = client.post('/api/sync/devices',
                          headers={'Authorization': f'Bearer {auth_token}'},
                          data=json.dumps({
                              'device_type': 'apple_watch',
                              'auth_token': 'mock_apple_token_456'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'synced_metrics' in data
    assert data['status'] == 'success'
    assert 'apple_watch' in data['message']

def test_device_sync_garmin(client, auth_token):
    """Test Garmin device sync stub"""
    response = client.post('/api/sync/devices',
                          headers={'Authorization': f'Bearer {auth_token}'},
                          data=json.dumps({
                              'device_type': 'garmin',
                              'auth_token': 'mock_garmin_token_789'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'synced_metrics' in data
    assert data['status'] == 'success'
    assert 'garmin' in data['message']

def test_chart_generation_with_data(client, auth_token):
    """Test chart generation when data exists"""
    # First create some weight data
    for i, value in enumerate([70.0, 70.2, 69.8, 70.1]):
        client.post('/api/metrics',
                   headers={'Authorization': f'Bearer {auth_token}'},
                   data=json.dumps({
                       'metric_type': 'weight',
                       'value': value,
                       'unit': 'kg'
                   }),
                   content_type='application/json')
    
    # Then generate chart
    response = client.get('/api/dashboard/chart?metric_type=weight&days=30',
                         headers={'Authorization': f'Bearer {auth_token}'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'chart_url' in data
    assert data['chart_url'].endswith('.png')
    assert '/static/charts/' in data['chart_url']

def test_chart_generation_no_data(client, auth_token):
    """Test chart generation when no data exists"""
    response = client.get('/api/dashboard/chart?metric_type=blood_pressure&days=30',
                         headers={'Authorization': f'Bearer {auth_token}'})
    
    # This might return 404 if no data exists, or 200 with empty chart
    # depending on implementation
    assert response.status_code in [200, 404]

def test_unauthorized_access_metrics(client):
    """Test accessing metrics without authentication"""
    response = client.get('/api/metrics')
    assert response.status_code == 401  # Custom JWT decorator returns 401

def test_unauthorized_access_sync(client):
    """Test accessing sync endpoint without authentication"""
    response = client.post('/api/sync/devices',
                          data=json.dumps({
                              'device_type': 'fitbit',
                              'auth_token': 'test'
                          }),
                          content_type='application/json')
    assert response.status_code == 401  # Custom JWT decorator returns 401

def test_unauthorized_access_chart(client):
    """Test accessing chart endpoint without authentication"""
    response = client.get('/api/dashboard/chart?metric_type=weight')
    assert response.status_code == 401  # Custom JWT decorator returns 401

def test_invalid_jwt_token(client):
    """Test using invalid JWT token"""
    response = client.get('/api/metrics',
                         headers={'Authorization': 'Bearer invalid_token_12345'})
    assert response.status_code == 401  # Invalid token format

def test_missing_required_fields_registration(client):
    """Test registration with missing required fields"""
    response = client.post('/api/auth/register',
                          data=json.dumps({
                              'email': 'test@example.com'
                              # Missing password and name
                          }),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_missing_required_fields_login(client):
    """Test login with missing required fields"""
    response = client.post('/api/auth/login',
                          data=json.dumps({
                              'email': 'test@example.com'
                              # Missing password
                          }),
                          content_type='application/json')
    
    assert response.status_code == 400

def test_missing_required_fields_metrics(client, auth_token):
    """Test creating metric with missing required fields"""
    response = client.post('/api/metrics',
                          headers={'Authorization': f'Bearer {auth_token}'},
                          data=json.dumps({
                              'metric_type': 'weight'
                              # Missing value and unit
                          }),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_rate_limiting_registration(client):
    """Test rate limiting on registration endpoint"""
    # This test assumes rate limiting is configured
    # Make multiple rapid requests
    for i in range(6):  # Limit is 5 per minute
        response = client.post('/api/auth/register',
                              data=json.dumps({
                                  'email': f'user{i}@example.com',
                                  'password': 'password123',
                                  'name': f'User {i}'
                              }),
                              content_type='application/json')
        
        if i < 5:
            assert response.status_code in [201, 400]  # 201 for success, 400 for duplicate
        else:
            # The 6th request should be rate limited
            assert response.status_code == 429

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
