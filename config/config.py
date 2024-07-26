import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'AAAAB3NzaC1yc2EAAAADAQABAAAAgQDaKe2ZWyHFcS0UOCq+RN2RYP0vERmm4sNDqvVFv7GWumk2xcXCjdbdNtm3cQJewGgMP0rxQF726k/qdATy0CxHW/uQzyTThS4K7wAnorGv8kHsT2RmsOQkPMTBCNNO0z1Q2T6VrPErLcL/caK49YmiDKnn/aTisdQZ/kizPW20bw==')
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///your_database.db')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    DATABASE_URI = os.getenv('DEV_DATABASE_URI', 'sqlite:///dev_database.db')

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DATABASE_URI = os.getenv('TEST_DATABASE_URI', 'sqlite:///test_database.db')

class ProductionConfig(Config):
    """Production configuration."""
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///prod_database.db')

def get_config(env):
    """Retrieve the configuration class based on the environment."""
    config_mapping = {
        'development': DevelopmentConfig,
        'testing': TestingConfig,
        'production': ProductionConfig
    }
    return config_mapping.get(env, ProductionConfig)