-- Health Tracker Database Schema
-- Version: 1.0.0
-- Date: 2022-10-01
-- Description: Initial schema for health tracker application

BEGIN TRANSACTION;

-- Users table for authentication
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Health metrics table for user data
CREATE TABLE health_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    value DECIMAL(10,2) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    notes TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Device sync tracking table
CREATE TABLE device_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    device_type VARCHAR(50) NOT NULL,
    device_id VARCHAR(100),
    auth_token VARCHAR(255),
    last_sync_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    sync_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Indexes for performance optimization
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);

CREATE INDEX idx_health_metrics_user_id ON health_metrics(user_id);
CREATE INDEX idx_health_metrics_type ON health_metrics(metric_type);
CREATE INDEX idx_health_metrics_date ON health_metrics(recorded_at);
CREATE INDEX idx_health_metrics_user_type ON health_metrics(user_id, metric_type);
CREATE INDEX idx_health_metrics_user_date ON health_metrics(user_id, recorded_at);

CREATE INDEX idx_device_links_user_id ON device_links(user_id);
CREATE INDEX idx_device_links_type ON device_links(device_type);
CREATE INDEX idx_device_links_active ON device_links(is_active);

-- Constraints for data integrity
ALTER TABLE health_metrics ADD CONSTRAINT chk_value_positive CHECK (value >= 0);
ALTER TABLE health_metrics ADD CONSTRAINT chk_metric_type CHECK (
    metric_type IN ('weight', 'blood_pressure', 'heart_rate', 'steps')
);
ALTER TABLE device_links ADD CONSTRAINT chk_device_type CHECK (
    device_type IN ('fitbit', 'apple_watch', 'garmin')
);
ALTER TABLE device_links ADD CONSTRAINT chk_sync_count_positive CHECK (sync_count >= 0);

-- Sample data for testing and demo purposes
INSERT INTO users (email, password_hash, name) VALUES 
('demo@healthtracker.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj0kfJb6/q.C', 'Demo User'),
('test@example.com', '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', 'Test User');

INSERT INTO health_metrics (user_id, metric_type, value, unit, notes, recorded_at) VALUES 
-- Demo user data (user_id = 1)
(1, 'weight', 70.5, 'kg', 'Morning measurement', '2022-11-01 08:00:00'),
(1, 'weight', 70.2, 'kg', 'Morning measurement', '2022-11-02 08:00:00'),
(1, 'weight', 70.8, 'kg', 'Morning measurement', '2022-11-03 08:00:00'),
(1, 'weight', 70.1, 'kg', 'Morning measurement', '2022-11-04 08:00:00'),
(1, 'weight', 69.9, 'kg', 'Morning measurement', '2022-11-05 08:00:00'),

(1, 'heart_rate', 72, 'bpm', 'Resting heart rate', '2022-11-01 09:00:00'),
(1, 'heart_rate', 75, 'bpm', 'Resting heart rate', '2022-11-02 09:00:00'),
(1, 'heart_rate', 73, 'bpm', 'Resting heart rate', '2022-11-03 09:00:00'),
(1, 'heart_rate', 71, 'bpm', 'Resting heart rate', '2022-11-04 09:00:00'),
(1, 'heart_rate', 74, 'bpm', 'Resting heart rate', '2022-11-05 09:00:00'),

(1, 'steps', 8500, 'count', 'Daily step count', '2022-11-01 23:59:00'),
(1, 'steps', 9200, 'count', 'Daily step count', '2022-11-02 23:59:00'),
(1, 'steps', 7800, 'count', 'Daily step count', '2022-11-03 23:59:00'),
(1, 'steps', 10100, 'count', 'Daily step count', '2022-11-04 23:59:00'),
(1, 'steps', 8900, 'count', 'Daily step count', '2022-11-05 23:59:00'),

(1, 'blood_pressure', 120, 'mmHg', 'Systolic pressure', '2022-11-01 10:00:00'),
(1, 'blood_pressure', 118, 'mmHg', 'Systolic pressure', '2022-11-02 10:00:00'),
(1, 'blood_pressure', 122, 'mmHg', 'Systolic pressure', '2022-11-03 10:00:00'),

-- Test user data (user_id = 2)
(2, 'weight', 65.0, 'kg', 'Test measurement', '2022-11-01 08:30:00'),
(2, 'heart_rate', 68, 'bpm', 'Test measurement', '2022-11-01 09:30:00'),
(2, 'steps', 12000, 'count', 'Test measurement', '2022-11-01 23:30:00');

INSERT INTO device_links (user_id, device_type, device_id, last_sync_at, is_active, sync_count) VALUES 
(1, 'fitbit', 'FB123456789', '2022-11-05 12:00:00', TRUE, 25),
(1, 'apple_watch', 'AW987654321', '2022-11-05 11:30:00', TRUE, 18),
(2, 'garmin', 'GM456789123', '2022-11-01 10:00:00', TRUE, 5);

-- Create triggers for updated_at timestamps
CREATE TRIGGER update_users_timestamp 
    AFTER UPDATE ON users
    FOR EACH ROW
    BEGIN
        UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER update_device_links_timestamp 
    AFTER UPDATE ON device_links
    FOR EACH ROW
    BEGIN
        UPDATE device_links SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

-- Verify data integrity
-- Check that all foreign keys are valid
SELECT 'Foreign key validation' as check_name, 
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END as result
FROM health_metrics hm 
LEFT JOIN users u ON hm.user_id = u.id 
WHERE u.id IS NULL;

SELECT 'Device links validation' as check_name,
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END as result  
FROM device_links dl
LEFT JOIN users u ON dl.user_id = u.id
WHERE u.id IS NULL;

-- Show summary statistics
SELECT 'Database setup summary' as info;
SELECT COUNT(*) as user_count FROM users;
SELECT COUNT(*) as metrics_count FROM health_metrics;
SELECT COUNT(*) as device_links_count FROM device_links;

COMMIT;

-- End of migration 001_init.sql
