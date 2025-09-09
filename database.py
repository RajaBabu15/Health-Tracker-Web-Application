"""
🗄️ Health Tracker Database Management & Analytics
Comprehensive database models and health data analysis functions for the Flask Health Tracker

Features:
- SQLAlchemy models for users, health metrics, wearable data
- Data preprocessing and analysis functions
- Time-series health trend calculations
- Integration with pandas and numpy for analytics
- Health goal tracking and achievement analytics
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HealthDataAnalyzer:
    """Advanced health data analysis and insights generation"""
    
    def __init__(self, db_session):
        self.db = db_session
        
    def get_user_health_dataframe(self, user_id: int, days: int = 90) -> pd.DataFrame:
        """
        Create comprehensive health data DataFrame for analysis
        Combines manual entries and wearable data into unified dataset
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Import models (avoiding circular import)
        from app import HealthMetric, DeviceConnection
        
        # Get manual health metrics
        manual_data = HealthMetric.query.filter(
            HealthMetric.user_id == user_id,
            HealthMetric.recorded_at >= datetime.combine(start_date, datetime.min.time())
        ).all()
        
        # Get device connection data (stub - would integrate with actual device APIs)
        device_data = DeviceConnection.query.filter(
            DeviceConnection.user_id == user_id,
            DeviceConnection.is_active == True
        ).all()
        
        # Convert to DataFrames
        manual_df = self._convert_manual_to_df(manual_data)
        
        # For now, just use manual data (device integration would be added here)
        df = manual_df if not manual_df.empty else pd.DataFrame()
        
        if not df.empty:
            df = df.sort_values('date').reset_index(drop=True)
            df = self._calculate_derived_metrics(df)
        
        return df
    
    def _convert_manual_to_df(self, manual_data: List) -> pd.DataFrame:
        """Convert manual health metrics to DataFrame"""
        if not manual_data:
            return pd.DataFrame()
            
        data_dict = {}
        for entry in manual_data:
            date_key = entry.recorded_at.date()
            
            if date_key not in data_dict:
                data_dict[date_key] = {'date': date_key}
            
            if entry.metric_type == 'blood_pressure':
                data_dict[date_key]['systolic_bp'] = entry.value
                data_dict[date_key]['diastolic_bp'] = entry.value_secondary
            else:
                data_dict[date_key][entry.metric_type] = entry.value
        
        return pd.DataFrame(list(data_dict.values()))
    
    def _convert_wearable_to_df(self, wearable_data: List) -> pd.DataFrame:
        """Convert wearable data to DataFrame"""
        if not wearable_data:
            return pd.DataFrame()
            
        data_dict = {}
        for entry in wearable_data:
            date_key = entry.date_recorded
            
            if date_key not in data_dict:
                data_dict[date_key] = {'date': date_key}
            
            data_dict[date_key][entry.data_type] = entry.value
        
        return pd.DataFrame(list(data_dict.values()))
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived health metrics"""
        df = df.copy()
        
        # Calculate BMI if weight and height available
        if 'weight' in df.columns:
            # Would need height from user profile - simplified for demo
            df['weight_change'] = df['weight'].diff()
            df['weight_7day_avg'] = df['weight'].rolling(window=7, min_periods=1).mean()
        
        # Calculate step intensity levels
        if 'steps' in df.columns:
            df['step_intensity'] = pd.cut(df['steps'], 
                                        bins=[0, 5000, 10000, 15000, float('inf')],
                                        labels=['Low', 'Moderate', 'Active', 'Very Active'])
            df['steps_7day_avg'] = df['steps'].rolling(window=7, min_periods=1).mean()
        
        # Calculate heart rate zones
        if 'heart_rate' in df.columns:
            df['hr_zone'] = pd.cut(df['heart_rate'],
                                 bins=[0, 60, 70, 85, 100, float('inf')],
                                 labels=['Low', 'Normal', 'Elevated', 'High', 'Very High'])
        
        # Sleep quality assessment
        if 'sleep_hours' in df.columns:
            df['sleep_quality'] = pd.cut(df['sleep_hours'],
                                       bins=[0, 6, 7, 9, float('inf')],
                                       labels=['Poor', 'Fair', 'Good', 'Excessive'])
        
        # Calculate streak days for goals
        if 'steps' in df.columns:
            df['step_goal_met'] = df['steps'] >= 10000
            df['step_streak'] = self._calculate_streak(df['step_goal_met'])
        
        return df
    
    def _calculate_streak(self, boolean_series: pd.Series) -> pd.Series:
        """Calculate consecutive days of goal achievement"""
        groups = boolean_series.ne(boolean_series.shift()).cumsum()
        return boolean_series.groupby(groups).cumsum() * boolean_series
    
    def generate_health_insights(self, user_id: int) -> Dict[str, Any]:
        """Generate comprehensive health insights and recommendations"""
        df = self.get_user_health_dataframe(user_id)
        
        if df.empty:
            return {"message": "No data available for analysis"}
        
        insights = {
            "data_period": {
                "start_date": df['date'].min().isoformat(),
                "end_date": df['date'].max().isoformat(),
                "total_days": len(df)
            },
            "activity_insights": self._analyze_activity(df),
            "health_trends": self._analyze_trends(df),
            "goal_achievements": self._analyze_goals(df),
            "recommendations": self._generate_recommendations(df)
        }
        
        return insights
    
    def _analyze_activity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze activity patterns"""
        activity = {}
        
        if 'steps' in df.columns:
            steps_data = df['steps'].dropna()
            activity['steps'] = {
                "average_daily": int(steps_data.mean()) if not steps_data.empty else 0,
                "max_daily": int(steps_data.max()) if not steps_data.empty else 0,
                "days_above_10k": int((steps_data >= 10000).sum()) if not steps_data.empty else 0,
                "consistency_score": float(1 - (steps_data.std() / steps_data.mean())) if not steps_data.empty and steps_data.mean() > 0 else 0
            }
        
        if 'sleep_hours' in df.columns:
            sleep_data = df['sleep_hours'].dropna()
            activity['sleep'] = {
                "average_hours": round(float(sleep_data.mean()), 1) if not sleep_data.empty else 0,
                "nights_optimal": int(((sleep_data >= 7) & (sleep_data <= 9)).sum()) if not sleep_data.empty else 0,
                "sleep_debt": max(0, round(float(7 * len(sleep_data) - sleep_data.sum()), 1)) if not sleep_data.empty else 0
            }
        
        if 'heart_rate' in df.columns:
            hr_data = df['heart_rate'].dropna()
            activity['heart_rate'] = {
                "average_resting": int(hr_data.mean()) if not hr_data.empty else 0,
                "hr_variability": round(float(hr_data.std()), 1) if not hr_data.empty else 0,
                "days_normal_range": int(((hr_data >= 60) & (hr_data <= 100)).sum()) if not hr_data.empty else 0
            }
        
        return activity
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze health trends over time"""
        trends = {}
        
        # Weight trends
        if 'weight' in df.columns:
            weight_data = df[['date', 'weight']].dropna()
            if len(weight_data) > 1:
                # Calculate trend using linear regression
                x = np.arange(len(weight_data))
                y = weight_data['weight'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trends['weight'] = {
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "weekly_change_estimate": round(slope * 7, 2),
                    "correlation_strength": round(abs(r_value), 3),
                    "is_significant": p_value < 0.05
                }
        
        # Steps trends
        if 'steps' in df.columns:
            steps_data = df[['date', 'steps']].dropna()
            if len(steps_data) > 7:  # At least a week of data
                recent_avg = steps_data.tail(7)['steps'].mean()
                earlier_avg = steps_data.head(7)['steps'].mean()
                
                trends['activity'] = {
                    "recent_vs_earlier": "improving" if recent_avg > earlier_avg * 1.05 else "declining" if recent_avg < earlier_avg * 0.95 else "stable",
                    "percent_change": round(((recent_avg - earlier_avg) / earlier_avg) * 100, 1) if earlier_avg > 0 else 0
                }
        
        return trends
    
    def _analyze_goals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze goal achievements"""
        goals = {}
        
        if 'steps' in df.columns:
            steps_data = df['steps'].dropna()
            goals['daily_steps'] = {
                "goal": 10000,
                "achievement_rate": round(float((steps_data >= 10000).mean() * 100), 1) if not steps_data.empty else 0,
                "current_streak": int(df['step_streak'].iloc[-1]) if 'step_streak' in df.columns and not df.empty else 0,
                "best_streak": int(df['step_streak'].max()) if 'step_streak' in df.columns and not df.empty else 0
            }
        
        if 'sleep_hours' in df.columns:
            sleep_data = df['sleep_hours'].dropna()
            goals['sleep_quality'] = {
                "goal_range": "7-9 hours",
                "achievement_rate": round(float(((sleep_data >= 7) & (sleep_data <= 9)).mean() * 100), 1) if not sleep_data.empty else 0,
                "avg_sleep": round(float(sleep_data.mean()), 1) if not sleep_data.empty else 0
            }
        
        return goals
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate personalized health recommendations"""
        recommendations = []
        
        # Steps recommendations
        if 'steps' in df.columns:
            avg_steps = df['steps'].mean()
            if avg_steps < 8000:
                recommendations.append("🚶 Try to increase daily steps - aim for at least 8,000 steps per day")
            elif avg_steps < 10000:
                recommendations.append("🎯 You're close to the 10,000 step goal - add a 10-minute walk to reach it!")
            else:
                recommendations.append("✅ Great job maintaining high activity levels!")
        
        # Sleep recommendations
        if 'sleep_hours' in df.columns:
            avg_sleep = df['sleep_hours'].mean()
            if avg_sleep < 7:
                recommendations.append("😴 Consider improving sleep habits - aim for 7-9 hours per night")
            elif avg_sleep > 9:
                recommendations.append("⏰ You might be oversleeping - 7-9 hours is optimal for most adults")
        
        # Weight recommendations
        if 'weight' in df.columns and len(df['weight'].dropna()) > 7:
            weight_trend = df['weight'].diff().mean()
            if abs(weight_trend) > 0.1:  # More than 100g per day change
                recommendations.append("⚖️ Monitor weight changes - consult healthcare provider if concerned")
        
        # Heart rate recommendations
        if 'heart_rate' in df.columns:
            avg_hr = df['heart_rate'].mean()
            if avg_hr > 85:
                recommendations.append("❤️ Resting heart rate is elevated - consider stress management techniques")
            elif avg_hr < 60:
                recommendations.append("💪 Low resting heart rate suggests good cardiovascular fitness!")
        
        # General recommendations
        if len(df) > 30:  # Have at least a month of data
            consistency_metrics = []
            for col in ['steps', 'sleep_hours', 'heart_rate']:
                if col in df.columns:
                    data = df[col].dropna()
                    if not data.empty:
                        cv = data.std() / data.mean() if data.mean() > 0 else 0
                        consistency_metrics.append(cv)
            
            if consistency_metrics and np.mean(consistency_metrics) > 0.3:
                recommendations.append("📊 Focus on consistency - regular habits lead to better health outcomes")
        
        if not recommendations:
            recommendations.append("📈 Keep up the great work tracking your health metrics!")
        
        return recommendations
    
    def create_health_report(self, user_id: int, save_path: Optional[str] = None) -> str:
        """Generate comprehensive health report with visualizations"""
        df = self.get_user_health_dataframe(user_id, days=90)
        insights = self.generate_health_insights(user_id)
        
        if df.empty:
            return "No data available for report generation"
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Health Tracker - 90 Day Report', fontsize=16, fontweight='bold')
        
        # Steps trend
        if 'steps' in df.columns:
            steps_data = df[['date', 'steps']].dropna()
            axes[0, 0].plot(steps_data['date'], steps_data['steps'], marker='o', linewidth=2)
            axes[0, 0].axhline(y=10000, color='r', linestyle='--', label='Daily Goal')
            axes[0, 0].set_title('Daily Steps Trend')
            axes[0, 0].set_ylabel('Steps')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Sleep pattern
        if 'sleep_hours' in df.columns:
            sleep_data = df[['date', 'sleep_hours']].dropna()
            axes[0, 1].bar(range(len(sleep_data)), sleep_data['sleep_hours'], alpha=0.7, color='purple')
            axes[0, 1].axhline(y=7, color='g', linestyle='--', label='Minimum Recommended')
            axes[0, 1].axhline(y=9, color='g', linestyle='--', label='Maximum Recommended')
            axes[0, 1].set_title('Sleep Hours Pattern')
            axes[0, 1].set_ylabel('Hours')
            axes[0, 1].legend()
        
        # Heart rate distribution
        if 'heart_rate' in df.columns:
            hr_data = df['heart_rate'].dropna()
            axes[1, 0].hist(hr_data, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(x=hr_data.mean(), color='r', linestyle='--', label=f'Average: {hr_data.mean():.0f}')
            axes[1, 0].set_title('Heart Rate Distribution')
            axes[1, 0].set_xlabel('BPM')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Weight trend (if available)
        if 'weight' in df.columns:
            weight_data = df[['date', 'weight']].dropna()
            if not weight_data.empty:
                axes[1, 1].plot(weight_data['date'], weight_data['weight'], marker='s', linewidth=2, color='green')
                axes[1, 1].set_title('Weight Progress')
                axes[1, 1].set_ylabel('Weight (kg)')
        else:
            # Show calories burned if weight not available
            if 'calories_burned' in df.columns:
                cal_data = df[['date', 'calories_burned']].dropna()
                axes[1, 1].plot(cal_data['date'], cal_data['calories_burned'], marker='d', linewidth=2, color='red')
                axes[1, 1].set_title('Calories Burned Trend')
                axes[1, 1].set_ylabel('Calories')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return f"Health report generated successfully for {len(df)} days of data"
    
    def export_health_data(self, user_id: int, format: str = 'csv') -> str:
        """Export health data in various formats"""
        df = self.get_user_health_dataframe(user_id, days=365)  # Full year
        
        if df.empty:
            return "No data to export"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            filename = f"health_data_export_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif format.lower() == 'json':
            filename = f"health_data_export_{timestamp}.json"
            df.to_json(filename, orient='records', date_format='iso')
        elif format.lower() == 'excel':
            filename = f"health_data_export_{timestamp}.xlsx"
            df.to_excel(filename, index=False)
        
        return f"Data exported successfully to {filename}"

class HealthGoalTracker:
    """Track and manage health goals"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def set_user_goals(self, user_id: int, goals: Dict[str, float]):
        """Set health goals for user"""
        # In a full implementation, this would save to a UserGoals table
        # For now, we'll use default goals
        default_goals = {
            'daily_steps': 10000,
            'weekly_exercise_minutes': 150,
            'daily_water_intake': 2.0,  # liters
            'target_weight': None,  # user-defined
            'sleep_hours_min': 7,
            'sleep_hours_max': 9
        }
        return default_goals
    
    def calculate_goal_progress(self, user_id: int, analyzer: HealthDataAnalyzer) -> Dict[str, Any]:
        """Calculate progress towards goals"""
        df = analyzer.get_user_health_dataframe(user_id, days=30)
        goals = self.set_user_goals(user_id, {})
        
        progress = {}
        
        # Daily steps goal
        if 'steps' in df.columns and not df['steps'].empty:
            avg_steps = df['steps'].mean()
            progress['steps'] = {
                'goal': goals['daily_steps'],
                'current_average': int(avg_steps),
                'progress_percentage': min(100, round((avg_steps / goals['daily_steps']) * 100, 1)),
                'days_achieved': int((df['steps'] >= goals['daily_steps']).sum()),
                'total_days': len(df['steps'].dropna())
            }
        
        # Sleep goal
        if 'sleep_hours' in df.columns and not df['sleep_hours'].empty:
            sleep_in_range = ((df['sleep_hours'] >= goals['sleep_hours_min']) & 
                             (df['sleep_hours'] <= goals['sleep_hours_max'])).sum()
            total_sleep_days = len(df['sleep_hours'].dropna())
            
            progress['sleep'] = {
                'goal_range': f"{goals['sleep_hours_min']}-{goals['sleep_hours_max']} hours",
                'nights_in_range': int(sleep_in_range),
                'total_nights': int(total_sleep_days),
                'progress_percentage': round((sleep_in_range / total_sleep_days) * 100, 1) if total_sleep_days > 0 else 0,
                'average_sleep': round(df['sleep_hours'].mean(), 1)
            }
        
        return progress

# Utility functions for data preprocessing
def clean_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess health data"""
    df = df.copy()
    
    # Remove obvious outliers
    for col in ['steps', 'heart_rate', 'sleep_hours']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # For health metrics, also apply logical bounds
            if col == 'steps':
                lower_bound = max(0, lower_bound)
                upper_bound = min(50000, upper_bound)  # Max realistic daily steps
            elif col == 'heart_rate':
                lower_bound = max(30, lower_bound)  # Min viable heart rate
                upper_bound = min(200, upper_bound)  # Max heart rate
            elif col == 'sleep_hours':
                lower_bound = max(0, lower_bound)
                upper_bound = min(16, upper_bound)  # Max realistic sleep
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def generate_sample_health_data(user_id: int, start_date: date, end_date: date) -> List[Dict]:
    """Generate realistic sample health data for testing"""
    sample_data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Generate correlated health metrics
        base_activity = np.random.normal(0.7, 0.2)  # Base activity level
        base_activity = max(0.1, min(1.0, base_activity))
        
        daily_data = {
            'date': current_date,
            'steps': int(np.random.normal(8000 + base_activity * 5000, 2000)),
            'heart_rate': int(np.random.normal(75 - base_activity * 10, 8)),
            'sleep_hours': round(np.random.normal(7.5 + np.random.normal(0, 0.5), 1), 1),
            'calories_burned': int(np.random.normal(2000 + base_activity * 800, 300)),
            'active_minutes': int(np.random.normal(30 + base_activity * 60, 20))
        }
        
        # Ensure realistic bounds
        daily_data['steps'] = max(1000, min(25000, daily_data['steps']))
        daily_data['heart_rate'] = max(50, min(120, daily_data['heart_rate']))
        daily_data['sleep_hours'] = max(4.0, min(12.0, daily_data['sleep_hours']))
        daily_data['calories_burned'] = max(1200, min(4000, daily_data['calories_burned']))
        daily_data['active_minutes'] = max(0, min(300, daily_data['active_minutes']))
        
        sample_data.append(daily_data)
        current_date += timedelta(days=1)
    
    return sample_data
