#!/usr/bin/env python3
"""
AWS Deployment Configuration for Health Tracker Web Application
Supports deployment to AWS Elastic Beanstalk, EC2, and RDS

Usage:
    python deploy_aws.py --platform eb  # Elastic Beanstalk
    python deploy_aws.py --platform ec2 # EC2 instance
"""

import os
import json
import boto3
import argparse
from pathlib import Path

class HealthTrackerAWSDeployment:
    """AWS deployment configuration for Health Tracker"""
    
    def __init__(self):
        self.app_name = "health-tracker-web-app"
        self.environment_name = "health-tracker-env"
        self.region = "us-east-1"  # Default region
        
    def create_eb_config(self):
        """Create Elastic Beanstalk configuration files"""
        
        # Create .ebextensions directory
        eb_dir = Path(".ebextensions")
        eb_dir.mkdir(exist_ok=True)
        
        # Database configuration
        db_config = {
            "Resources": {
                "AWSEBAutoScalingGroup": {
                    "Metadata": {
                        "AWS::CloudFormation::Authentication": {
                            "S3Auth": {
                                "type": "s3",
                                "buckets": ["elasticbeanstalk-*"],
                                "roleName": "aws-elasticbeanstalk-ec2-role"
                            }
                        }
                    }
                }
            },
            "option_settings": {
                "aws:rds:dbinstance": {
                    "DBEngine": "postgres",
                    "DBEngineVersion": "13.7",
                    "DBInstanceClass": "db.t3.micro",
                    "DBAllocatedStorage": "20",
                    "MultiAZDatabase": "false",
                    "DBDeletionPolicy": "Snapshot"
                },
                "aws:elasticbeanstalk:application:environment": {
                    "FLASK_ENV": "production",
                    "DATABASE_URL": "postgresql://$(RDS_HOSTNAME):$(RDS_PORT)/$(RDS_DB_NAME)",
                    "SECRET_KEY": "health-tracker-production-key-change-me"
                }
            }
        }
        
        with open(eb_dir / "01_database.config", 'w') as f:
            json.dump(db_config, f, indent=2)
        
        # Python packages configuration
        packages_config = {
            "packages": {
                "yum": {
                    "postgresql-devel": [],
                    "python3-devel": [],
                    "gcc": []
                }
            }
        }
        
        with open(eb_dir / "02_packages.config", 'w') as f:
            json.dump(packages_config, f, indent=2)
        
        # WSGI configuration
        wsgi_config = {
            "files": {
                "/opt/python/current/app/application.py": {
                    "mode": "000644",
                    "owner": "root",
                    "group": "root",
                    "content": """
from flask_app import app as application

if __name__ == '__main__':
    application.run(debug=False)
"""
                }
            }
        }
        
        with open(eb_dir / "03_wsgi.config", 'w') as f:
            json.dump(wsgi_config, f, indent=2)
        
        print("✅ Elastic Beanstalk configuration created")
        
    def create_dockerfile(self):
        """Create Dockerfile for containerized deployment"""
        
        dockerfile_content = """
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    postgresql-client \\
    gcc \\
    python3-dev \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_new.txt .
RUN pip install --no-cache-dir -r requirements_new.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -r healthtracker && chown -R healthtracker:healthtracker /app
USER healthtracker

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=flask_app.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "flask_app.py"]
"""
        
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content.strip())
        
        # Create .dockerignore
        dockerignore_content = """
__pycache__/
*.pyc
.git/
.pytest_cache/
.coverage
.env
*.log
health_tracker.db
*.sqlite
.DS_Store
"""
        
        with open(".dockerignore", 'w') as f:
            f.write(dockerignore_content.strip())
        
        print("✅ Docker configuration created")
        
    def create_cloudformation_template(self):
        """Create CloudFormation template for full stack deployment"""
        
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "Health Tracker Web Application - Full Stack",
            "Parameters": {
                "KeyName": {
                    "Description": "EC2 Key Pair for SSH access",
                    "Type": "AWS::EC2::KeyPair::KeyName"
                },
                "InstanceType": {
                    "Description": "EC2 instance type",
                    "Type": "String",
                    "Default": "t3.micro",
                    "AllowedValues": ["t3.micro", "t3.small", "t3.medium"]
                }
            },
            "Resources": {
                "HealthTrackerVPC": {
                    "Type": "AWS::EC2::VPC",
                    "Properties": {
                        "CidrBlock": "10.0.0.0/16",
                        "EnableDnsHostnames": True,
                        "Tags": [{"Key": "Name", "Value": "HealthTracker-VPC"}]
                    }
                },
                "PublicSubnet": {
                    "Type": "AWS::EC2::Subnet",
                    "Properties": {
                        "VpcId": {"Ref": "HealthTrackerVPC"},
                        "CidrBlock": "10.0.1.0/24",
                        "AvailabilityZone": {"Fn::Select": [0, {"Fn::GetAZs": ""}]},
                        "MapPublicIpOnLaunch": True
                    }
                },
                "PrivateSubnet": {
                    "Type": "AWS::EC2::Subnet",
                    "Properties": {
                        "VpcId": {"Ref": "HealthTrackerVPC"},
                        "CidrBlock": "10.0.2.0/24",
                        "AvailabilityZone": {"Fn::Select": [1, {"Fn::GetAZs": ""}]}
                    }
                },
                "InternetGateway": {
                    "Type": "AWS::EC2::InternetGateway"
                },
                "AttachGateway": {
                    "Type": "AWS::EC2::VPCGatewayAttachment",
                    "Properties": {
                        "VpcId": {"Ref": "HealthTrackerVPC"},
                        "InternetGatewayId": {"Ref": "InternetGateway"}
                    }
                },
                "PublicRouteTable": {
                    "Type": "AWS::EC2::RouteTable",
                    "Properties": {
                        "VpcId": {"Ref": "HealthTrackerVPC"}
                    }
                },
                "PublicRoute": {
                    "Type": "AWS::EC2::Route",
                    "DependsOn": "AttachGateway",
                    "Properties": {
                        "RouteTableId": {"Ref": "PublicRouteTable"},
                        "DestinationCidrBlock": "0.0.0.0/0",
                        "GatewayId": {"Ref": "InternetGateway"}
                    }
                },
                "PublicSubnetRouteTableAssociation": {
                    "Type": "AWS::EC2::SubnetRouteTableAssociation",
                    "Properties": {
                        "SubnetId": {"Ref": "PublicSubnet"},
                        "RouteTableId": {"Ref": "PublicRouteTable"}
                    }
                },
                "WebServerSecurityGroup": {
                    "Type": "AWS::EC2::SecurityGroup",
                    "Properties": {
                        "GroupDescription": "Health Tracker Web Server Security Group",
                        "VpcId": {"Ref": "HealthTrackerVPC"},
                        "SecurityGroupIngress": [
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 80,
                                "ToPort": 80,
                                "CidrIp": "0.0.0.0/0"
                            },
                            {
                                "IpProtocol": "tcp", 
                                "FromPort": 443,
                                "ToPort": 443,
                                "CidrIp": "0.0.0.0/0"
                            },
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 22,
                                "ToPort": 22,
                                "CidrIp": "0.0.0.0/0"
                            },
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 5000,
                                "ToPort": 5000,
                                "CidrIp": "0.0.0.0/0"
                            }
                        ]
                    }
                },
                "DatabaseSecurityGroup": {
                    "Type": "AWS::EC2::SecurityGroup",
                    "Properties": {
                        "GroupDescription": "Health Tracker Database Security Group",
                        "VpcId": {"Ref": "HealthTrackerVPC"},
                        "SecurityGroupIngress": [
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 5432,
                                "ToPort": 5432,
                                "SourceSecurityGroupId": {"Ref": "WebServerSecurityGroup"}
                            }
                        ]
                    }
                },
                "DBSubnetGroup": {
                    "Type": "AWS::RDS::DBSubnetGroup",
                    "Properties": {
                        "DBSubnetGroupDescription": "Subnet group for Health Tracker RDS",
                        "SubnetIds": [{"Ref": "PublicSubnet"}, {"Ref": "PrivateSubnet"}]
                    }
                },
                "HealthTrackerDatabase": {
                    "Type": "AWS::RDS::DBInstance",
                    "Properties": {
                        "DBInstanceIdentifier": "health-tracker-db",
                        "DBName": "healthtracker",
                        "Engine": "postgres",
                        "MasterUsername": "healthtracker",
                        "MasterUserPassword": "ChangeMe123!",
                        "DBInstanceClass": "db.t3.micro",
                        "AllocatedStorage": "20",
                        "VPCSecurityGroups": [{"Ref": "DatabaseSecurityGroup"}],
                        "DBSubnetGroupName": {"Ref": "DBSubnetGroup"},
                        "MultiAZ": False,
                        "PubliclyAccessible": False,
                        "DeletionProtection": False
                    }
                },
                "HealthTrackerEC2Instance": {
                    "Type": "AWS::EC2::Instance",
                    "Properties": {
                        "ImageId": "ami-0c55b159cbfafe1d0",  # Amazon Linux 2
                        "InstanceType": {"Ref": "InstanceType"},
                        "KeyName": {"Ref": "KeyName"},
                        "SecurityGroupIds": [{"Ref": "WebServerSecurityGroup"}],
                        "SubnetId": {"Ref": "PublicSubnet"},
                        "UserData": {
                            "Fn::Base64": {
                                "Fn::Join": ["", [
                                    "#!/bin/bash\\n",
                                    "yum update -y\\n",
                                    "yum install -y python3 python3-pip git postgresql\\n",
                                    "pip3 install flask flask-sqlalchemy flask-login plotly pandas numpy matplotlib\\n",
                                    "cd /home/ec2-user\\n",
                                    "git clone https://github.com/username/Health-Tracker-Web-Application.git\\n",
                                    "cd Health-Tracker-Web-Application\\n",
                                    "export DATABASE_URL=postgresql://healthtracker:ChangeMe123!@",
                                    {"Fn::GetAtt": ["HealthTrackerDatabase", "Endpoint.Address"]},
                                    ":5432/healthtracker\\n",
                                    "python3 flask_app.py &\\n"
                                ]]
                            }
                        },
                        "Tags": [{"Key": "Name", "Value": "HealthTracker-WebServer"}]
                    }
                }
            },
            "Outputs": {
                "WebsiteURL": {
                    "Description": "Health Tracker Application URL",
                    "Value": {
                        "Fn::Join": ["", [
                            "http://",
                            {"Fn::GetAtt": ["HealthTrackerEC2Instance", "PublicDnsName"]},
                            ":5000"
                        ]]
                    }
                },
                "DatabaseEndpoint": {
                    "Description": "RDS Database Endpoint",
                    "Value": {"Fn::GetAtt": ["HealthTrackerDatabase", "Endpoint.Address"]}
                }
            }
        }
        
        with open("cloudformation-template.json", 'w') as f:
            json.dump(template, f, indent=2)
        
        print("✅ CloudFormation template created")
        
    def create_env_config(self):
        """Create environment configuration files"""
        
        # Production environment variables
        prod_env = """
# Production Environment Configuration
FLASK_ENV=production
FLASK_DEBUG=False

# Database Configuration (update with your RDS endpoint)
DATABASE_URL=postgresql://healthtracker:password@your-rds-endpoint.amazonaws.com:5432/healthtracker

# Security
SECRET_KEY=your-super-secret-key-change-me-in-production

# Application Settings
HEALTH_TRACKER_ENV=production
MAX_CONTENT_LENGTH=16777216  # 16MB max file upload

# AWS Settings (if using S3 for file storage)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=health-tracker-data
AWS_REGION=us-east-1

# Email Configuration (if using SES)
MAIL_SERVER=email-smtp.us-east-1.amazonaws.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_ses_username
MAIL_PASSWORD=your_ses_password

# Monitoring
SENTRY_DSN=your_sentry_dsn_for_error_tracking
"""
        
        with open(".env.production", 'w') as f:
            f.write(prod_env.strip())
        
        # Development environment
        dev_env = """
# Development Environment Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///health_tracker_dev.db

# Security
SECRET_KEY=dev-secret-key

# Application Settings
HEALTH_TRACKER_ENV=development
"""
        
        with open(".env.development", 'w') as f:
            f.write(dev_env.strip())
        
        print("✅ Environment configuration files created")
        
    def create_deployment_scripts(self):
        """Create deployment automation scripts"""
        
        # Deployment script
        deploy_script = """#!/bin/bash
# Health Tracker AWS Deployment Script

set -e

echo "🚀 Starting Health Tracker AWS Deployment..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if logged into AWS
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ Not logged into AWS. Please run 'aws configure' first."
    exit 1
fi

# Set variables
STACK_NAME="health-tracker-stack"
TEMPLATE_FILE="cloudformation-template.json"

echo "📦 Validating CloudFormation template..."
aws cloudformation validate-template --template-body file://$TEMPLATE_FILE

echo "🏗️ Deploying CloudFormation stack..."
aws cloudformation deploy \\
    --template-file $TEMPLATE_FILE \\
    --stack-name $STACK_NAME \\
    --parameter-overrides InstanceType=t3.micro \\
    --capabilities CAPABILITY_IAM \\
    --region us-east-1

echo "📋 Getting stack outputs..."
aws cloudformation describe-stacks \\
    --stack-name $STACK_NAME \\
    --query 'Stacks[0].Outputs' \\
    --output table

echo "✅ Deployment complete!"
echo "🌐 Your Health Tracker application should be accessible at the WebsiteURL above"
echo "⚠️ Remember to update the database password and secret key for production!"
"""
        
        with open("deploy.sh", 'w') as f:
            f.write(deploy_script.strip())
        
        os.chmod("deploy.sh", 0o755)  # Make executable
        
        # Cleanup script
        cleanup_script = """#!/bin/bash
# Health Tracker AWS Cleanup Script

set -e

echo "🗑️ Cleaning up Health Tracker AWS resources..."

STACK_NAME="health-tracker-stack"

echo "🔍 Checking if stack exists..."
if aws cloudformation describe-stacks --stack-name $STACK_NAME &> /dev/null; then
    echo "🗂️ Deleting CloudFormation stack..."
    aws cloudformation delete-stack --stack-name $STACK_NAME
    
    echo "⏳ Waiting for stack deletion to complete..."
    aws cloudformation wait stack-delete-complete --stack-name $STACK_NAME
    
    echo "✅ Stack deleted successfully!"
else
    echo "⚠️ Stack $STACK_NAME does not exist."
fi

echo "🧹 Cleanup complete!"
"""
        
        with open("cleanup.sh", 'w') as f:
            f.write(cleanup_script.strip())
        
        os.chmod("cleanup.sh", 0o755)  # Make executable
        
        print("✅ Deployment scripts created")
        
    def create_deployment_guide(self):
        """Create comprehensive deployment guide"""
        
        guide = """# 🏥 Health Tracker AWS Deployment Guide

## Overview
This guide covers deploying the Health Tracker web application to AWS using multiple deployment options.

## Prerequisites
1. AWS CLI installed and configured
2. Appropriate AWS IAM permissions
3. Python 3.8+ and pip installed
4. Git repository with your Health Tracker code

## Deployment Options

### Option 1: Elastic Beanstalk (Recommended)
Elastic Beanstalk provides easy deployment and management.

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init health-tracker --platform python-3.8 --region us-east-1

# Create environment and deploy
eb create health-tracker-env --database.engine postgres --database.username healthtracker

# Deploy updates
eb deploy
```

### Option 2: EC2 + CloudFormation (Full Control)
Use the provided CloudFormation template for complete infrastructure control.

```bash
# Deploy the full stack
./deploy.sh

# Or manually:
aws cloudformation deploy \\
    --template-file cloudformation-template.json \\
    --stack-name health-tracker-stack \\
    --parameter-overrides InstanceType=t3.micro \\
    --capabilities CAPABILITY_IAM
```

### Option 3: Container Deployment (Docker)
Deploy using Docker containers on AWS ECS or EKS.

```bash
# Build Docker image
docker build -t health-tracker .

# Test locally
docker run -p 5000:5000 health-tracker

# Push to ECR (update with your repository)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag health-tracker:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/health-tracker:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/health-tracker:latest
```

## Environment Configuration

### 1. Database Setup
Update your production environment with RDS details:

```bash
# Set environment variables
export DATABASE_URL="postgresql://username:password@your-rds-endpoint.amazonaws.com:5432/healthtracker"
export SECRET_KEY="your-super-secret-production-key"
```

### 2. Security Configuration
- Change default passwords
- Update secret keys
- Configure HTTPS/SSL certificates
- Set up proper IAM roles

### 3. Monitoring & Logging
- Enable CloudWatch logging
- Set up health checks
- Configure alerts for critical metrics

## Post-Deployment Checklist

### Immediate Tasks
- [ ] Test user registration and login
- [ ] Verify database connectivity
- [ ] Check health data input/output
- [ ] Test wearable data sync simulation
- [ ] Validate Jupyter notebook access

### Security Tasks
- [ ] Change all default passwords
- [ ] Update security groups (restrict access)
- [ ] Enable HTTPS with SSL certificate
- [ ] Set up Web Application Firewall (WAF)
- [ ] Configure backup strategy

### Performance Tasks
- [ ] Set up CloudFront CDN
- [ ] Configure auto-scaling
- [ ] Optimize database performance
- [ ] Set up monitoring dashboards

## Troubleshooting

### Common Issues
1. **Database Connection Errors**
   - Check security groups allow database access
   - Verify connection string format
   - Ensure database is in same VPC

2. **Application Not Starting**
   - Check CloudWatch logs
   - Verify all dependencies installed
   - Check environment variables

3. **Performance Issues**
   - Monitor CPU and memory usage
   - Check database query performance
   - Consider scaling instance size

### Useful Commands
```bash
# Check application logs
aws logs describe-log-groups
aws logs tail /aws/elasticbeanstalk/health-tracker-env/var/log/eb-engine.log

# Check database connection
aws rds describe-db-instances --db-instance-identifier health-tracker-db

# Monitor application health
aws elbv2 describe-target-health --target-group-arn your-target-group-arn
```

## Cost Optimization
- Use t3.micro instances for development
- Schedule non-production resources to stop outside business hours
- Use RDS reserved instances for production
- Implement S3 lifecycle policies for log retention

## Backup Strategy
1. **Database Backups**
   - Enable automated RDS backups
   - Set appropriate retention period
   - Test backup restoration process

2. **Application Backups**
   - Version control all code changes
   - Backup configuration files
   - Document deployment procedures

## Support
- Check AWS documentation for service-specific issues
- Review CloudWatch logs for application errors
- Monitor AWS Health Dashboard for service status

## Cleanup
To remove all AWS resources:
```bash
./cleanup.sh
```

---
**Note**: Always test deployments in a staging environment before production!
"""
        
        with open("AWS_DEPLOYMENT_GUIDE.md", 'w') as f:
            f.write(guide.strip())
        
        print("✅ Deployment guide created")

def main():
    parser = argparse.ArgumentParser(description="Deploy Health Tracker to AWS")
    parser.add_argument("--platform", choices=["eb", "ec2", "docker"], default="eb",
                       help="Deployment platform (default: eb)")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only create configuration files, don't deploy")
    
    args = parser.parse_args()
    
    deployer = HealthTrackerAWSDeployment()
    
    print("🏥 Setting up Health Tracker AWS Deployment Configuration...")
    
    # Create all configuration files
    deployer.create_eb_config()
    deployer.create_dockerfile()
    deployer.create_cloudformation_template()
    deployer.create_env_config()
    deployer.create_deployment_scripts()
    deployer.create_deployment_guide()
    
    print("\\n✅ AWS deployment configuration complete!")
    print("\\n📋 Files created:")
    print("   • .ebextensions/ - Elastic Beanstalk config")
    print("   • Dockerfile - Container configuration")
    print("   • cloudformation-template.json - Infrastructure as code")
    print("   • .env.production/.env.development - Environment variables")
    print("   • deploy.sh/cleanup.sh - Deployment scripts")
    print("   • AWS_DEPLOYMENT_GUIDE.md - Comprehensive guide")
    
    if not args.setup_only:
        print("\\n🚀 Ready for deployment!")
        print(f"   Run: ./deploy.sh")
        print("   Or see AWS_DEPLOYMENT_GUIDE.md for detailed instructions")
    
    print("\\n⚠️ Important:")
    print("   1. Update environment variables with your AWS credentials")
    print("   2. Change default passwords and secret keys")
    print("   3. Review security groups and access permissions")
    print("   4. Test in staging environment before production deployment")

if __name__ == "__main__":
    main()
