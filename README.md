# Health Tracker Web Application

A Flask-based health tracking web application with user authentication, health metrics management, and wearable device integration.

## Features
- 🔐 User authentication (JWT)
- 📊 Health metrics tracking (10+ metrics)
- 📱 Wearable device sync (Fitbit, Apple Watch, Garmin)
- 📈 Data visualization and charts
- 📓 Jupyter analysis notebooks
- 🔗 REST API endpoints

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Application runs on `http://localhost:5000`

## Tech Stack
- Flask, SQLAlchemy, JWT
- Pandas, NumPy, Matplotlib
- SQLite/PostgreSQL
- AWS deployment ready

## API Endpoints
- `POST /api/auth/register` - Register user
- `POST /api/auth/login` - Login user  
- `GET /api/metrics` - Get metrics
- `POST /api/metrics` - Add metric
- `POST /api/sync/devices` - Sync devices
- `GET /api/dashboard/chart` - Generate charts

## Deployment
AWS Elastic Beanstalk ready with included configuration files.

## License
MIT
