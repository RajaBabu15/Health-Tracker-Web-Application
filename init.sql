-- Health Tracker Database Initialization Script

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create database if not exists (handled by Docker)
-- The database 'healthtracker' is created by the postgres Docker container

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create health_metrics table
CREATE TABLE IF NOT EXISTS health_metrics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    metric_type VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    unit VARCHAR(20) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(50) DEFAULT 'manual',
    notes TEXT
);

-- Create device_connections table
CREATE TABLE IF NOT EXISTS device_connections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_type VARCHAR(50) NOT NULL,
    device_id VARCHAR(100),
    access_token VARCHAR(500),
    refresh_token VARCHAR(500),
    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_sync TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    sync_frequency INTEGER DEFAULT 60
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_health_metrics_user_type ON health_metrics(user_id, metric_type);
CREATE INDEX IF NOT EXISTS idx_health_metrics_recorded_at ON health_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_health_metrics_user_recorded ON health_metrics(user_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_device_connections_user ON device_connections(user_id);

-- Insert demo data
INSERT INTO users (email, name, password_hash) 
VALUES ('demo@healthtracker.com', 'Demo User', 'pbkdf2:sha256:260000$demo$salt$hash')
ON CONFLICT (email) DO NOTHING;

-- Insert sample health metrics for demo user
WITH demo_user AS (
    SELECT id FROM users WHERE email = 'demo@healthtracker.com'
)
INSERT INTO health_metrics (user_id, metric_type, value, unit, source, notes)
SELECT 
    demo_user.id,
    metric_type,
    value,
    unit,
    'manual' as source,
    'Demo data' as notes
FROM demo_user,
(VALUES 
    ('weight', 70.5, 'kg'),
    ('height', 175, 'cm'),
    ('heart_rate', 72, 'bpm'),
    ('steps', 8500, 'count'),
    ('sleep_hours', 7.5, 'hours'),
    ('water_intake', 2000, 'ml'),
    ('calories', 2200, 'kcal'),
    ('exercise_minutes', 45, 'minutes')
) AS metrics(metric_type, value, unit)
ON CONFLICT DO NOTHING;

-- Create a function to clean old chart files (to be called periodically)
CREATE OR REPLACE FUNCTION cleanup_old_charts()
RETURNS void AS $$
BEGIN
    -- This would be implemented in the application layer
    -- as PostgreSQL can't directly access file system
    RAISE NOTICE 'Chart cleanup should be implemented in application layer';
END;
$$ LANGUAGE plpgsql;
