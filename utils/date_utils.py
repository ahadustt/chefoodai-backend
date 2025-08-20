"""
ChefoodAI Date Utilities
Helper functions for date calculations and meal planning scheduling
"""

from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict, Any, Optional
import calendar

def get_week_dates(start_date: date = None) -> List[date]:
    """
    Get list of dates for a week starting from Monday
    
    Args:
        start_date: Starting date (defaults to current week)
        
    Returns:
        List of 7 dates starting from Monday
    """
    if start_date is None:
        start_date = date.today()
    
    # Find Monday of the week containing start_date
    days_since_monday = start_date.weekday()
    monday = start_date - timedelta(days=days_since_monday)
    
    return [monday + timedelta(days=i) for i in range(7)]

def get_month_dates(year: int = None, month: int = None) -> List[date]:
    """
    Get list of all dates in a month
    
    Args:
        year: Year (defaults to current year)
        month: Month (defaults to current month)
        
    Returns:
        List of all dates in the month
    """
    if year is None or month is None:
        today = date.today()
        year = year or today.year
        month = month or today.month
    
    # Get number of days in month
    _, num_days = calendar.monthrange(year, month)
    
    return [date(year, month, day) for day in range(1, num_days + 1)]

def get_date_range(start_date: date, end_date: date) -> List[date]:
    """
    Get list of dates between start and end date (inclusive)
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of dates in range
    """
    if start_date > end_date:
        return []
    
    dates = []
    current_date = start_date
    
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    return dates

def get_weekdays_in_range(start_date: date, end_date: date) -> List[date]:
    """
    Get list of weekdays (Monday-Friday) in date range
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of weekday dates
    """
    all_dates = get_date_range(start_date, end_date)
    return [d for d in all_dates if d.weekday() < 5]  # 0-4 are Mon-Fri

def get_weekends_in_range(start_date: date, end_date: date) -> List[date]:
    """
    Get list of weekend days (Saturday-Sunday) in date range
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of weekend dates
    """
    all_dates = get_date_range(start_date, end_date)
    return [d for d in all_dates if d.weekday() >= 5]  # 5-6 are Sat-Sun

def get_next_weekday(target_date: date, weekday: int) -> date:
    """
    Get next occurrence of specified weekday
    
    Args:
        target_date: Reference date
        weekday: Target weekday (0=Monday, 6=Sunday)
        
    Returns:
        Next date that falls on specified weekday
    """
    days_ahead = weekday - target_date.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    
    return target_date + timedelta(days=days_ahead)

def get_previous_weekday(target_date: date, weekday: int) -> date:
    """
    Get previous occurrence of specified weekday
    
    Args:
        target_date: Reference date
        weekday: Target weekday (0=Monday, 6=Sunday)
        
    Returns:
        Previous date that fell on specified weekday
    """
    days_behind = target_date.weekday() - weekday
    if days_behind <= 0:  # Target day hasn't happened this week
        days_behind += 7
    
    return target_date - timedelta(days=days_behind)

def format_date_for_display(target_date: date, format_type: str = 'friendly') -> str:
    """
    Format date for user-friendly display
    
    Args:
        target_date: Date to format
        format_type: Type of formatting ('friendly', 'short', 'long', 'iso')
        
    Returns:
        Formatted date string
    """
    today = date.today()
    
    if format_type == 'friendly':
        if target_date == today:
            return 'Today'
        elif target_date == today + timedelta(days=1):
            return 'Tomorrow'
        elif target_date == today - timedelta(days=1):
            return 'Yesterday'
        elif target_date.year == today.year:
            return target_date.strftime('%B %d')  # January 15
        else:
            return target_date.strftime('%B %d, %Y')  # January 15, 2024
    
    elif format_type == 'short':
        return target_date.strftime('%m/%d/%y')  # 01/15/24
    
    elif format_type == 'long':
        return target_date.strftime('%A, %B %d, %Y')  # Monday, January 15, 2024
    
    elif format_type == 'iso':
        return target_date.isoformat()  # 2024-01-15
    
    else:
        return str(target_date)

def get_meal_planning_weeks(start_date: date, num_weeks: int = 4) -> List[Dict[str, Any]]:
    """
    Get structured data for meal planning weeks
    
    Args:
        start_date: Starting date
        num_weeks: Number of weeks to generate
        
    Returns:
        List of week dictionaries with metadata
    """
    weeks = []
    current_monday = start_date - timedelta(days=start_date.weekday())
    
    for week_num in range(num_weeks):
        week_start = current_monday + timedelta(weeks=week_num)
        week_end = week_start + timedelta(days=6)
        
        week_data = {
            'week_number': week_num + 1,
            'start_date': week_start,
            'end_date': week_end,
            'dates': get_week_dates(week_start),
            'weekdays': [week_start + timedelta(days=i) for i in range(5)],
            'weekend': [week_start + timedelta(days=i) for i in range(5, 7)],
            'month_name': week_start.strftime('%B'),
            'year': week_start.year,
            'is_current_week': week_start <= date.today() <= week_end,
            'display_name': f"Week of {format_date_for_display(week_start, 'friendly')}"
        }
        
        weeks.append(week_data)
    
    return weeks

def calculate_meal_plan_duration(start_date: date, end_date: date) -> Dict[str, int]:
    """
    Calculate duration metrics for meal plan
    
    Args:
        start_date: Plan start date
        end_date: Plan end date
        
    Returns:
        Dictionary with duration metrics
    """
    if start_date > end_date:
        return {'total_days': 0, 'weekdays': 0, 'weekends': 0, 'weeks': 0}
    
    total_days = (end_date - start_date).days + 1
    weekdays = len(get_weekdays_in_range(start_date, end_date))
    weekends = len(get_weekends_in_range(start_date, end_date))
    weeks = (total_days + 6) // 7  # Round up to nearest week
    
    return {
        'total_days': total_days,
        'weekdays': weekdays,
        'weekends': weekends,
        'weeks': weeks
    }

def get_seasonal_info(target_date: date) -> Dict[str, Any]:
    """
    Get seasonal information for a date
    
    Args:
        target_date: Date to analyze
        
    Returns:
        Dictionary with seasonal information
    """
    month = target_date.month
    
    # Define seasons (Northern Hemisphere)
    if month in [12, 1, 2]:
        season = 'winter'
        seasonal_foods = ['root vegetables', 'citrus fruits', 'hearty stews', 'warm spices']
    elif month in [3, 4, 5]:
        season = 'spring'
        seasonal_foods = ['asparagus', 'peas', 'lettuce', 'strawberries', 'light salads']
    elif month in [6, 7, 8]:
        season = 'summer'
        seasonal_foods = ['tomatoes', 'berries', 'corn', 'zucchini', 'grilled foods']
    else:  # [9, 10, 11]
        season = 'fall'
        seasonal_foods = ['apples', 'pumpkin', 'squash', 'brussels sprouts', 'comfort foods']
    
    return {
        'season': season,
        'seasonal_foods': seasonal_foods,
        'month_name': target_date.strftime('%B'),
        'is_holiday_season': month in [11, 12]  # Thanksgiving/Christmas season
    }

def get_optimal_meal_prep_days(
    meal_plan_dates: List[date], 
    user_schedule: Dict[str, Any] = None
) -> Dict[str, List[date]]:
    """
    Suggest optimal meal prep days based on meal plan
    
    Args:
        meal_plan_dates: List of dates in meal plan
        user_schedule: User's schedule preferences
        
    Returns:
        Dictionary with suggested prep days
    """
    if not meal_plan_dates:
        return {'prep_days': [], 'cook_days': []}
    
    # Default to Sunday prep for the week
    prep_suggestions = []
    cook_days = []
    
    # Group dates by week
    weeks = {}
    for meal_date in meal_plan_dates:
        week_start = meal_date - timedelta(days=meal_date.weekday())
        if week_start not in weeks:
            weeks[week_start] = []
        weeks[week_start].append(meal_date)
    
    # Suggest prep days
    for week_start, week_dates in weeks.items():
        # Suggest Sunday before the week for prep
        prep_day = week_start - timedelta(days=1)  # Sunday
        prep_suggestions.append(prep_day)
        
        # Mark actual cooking days
        cook_days.extend(week_dates)
    
    return {
        'prep_days': prep_suggestions,
        'cook_days': cook_days,
        'batch_cooking_opportunities': prep_suggestions
    }

def is_business_day(target_date: date) -> bool:
    """Check if date is a business day (Monday-Friday)"""
    return target_date.weekday() < 5

def is_weekend(target_date: date) -> bool:
    """Check if date is a weekend (Saturday-Sunday)"""
    return target_date.weekday() >= 5

def get_days_until(target_date: date, reference_date: date = None) -> int:
    """
    Get number of days until target date
    
    Args:
        target_date: Target date
        reference_date: Reference date (defaults to today)
        
    Returns:
        Number of days (negative if in past)
    """
    if reference_date is None:
        reference_date = date.today()
    
    return (target_date - reference_date).days

def format_time_until(target_date: date, reference_date: date = None) -> str:
    """
    Format time until target date in human-readable form
    
    Args:
        target_date: Target date
        reference_date: Reference date (defaults to today)
        
    Returns:
        Human-readable time description
    """
    days = get_days_until(target_date, reference_date)
    
    if days == 0:
        return 'today'
    elif days == 1:
        return 'tomorrow'
    elif days == -1:
        return 'yesterday'
    elif days > 0:
        if days < 7:
            return f'in {days} days'
        elif days < 30:
            weeks = days // 7
            return f'in {weeks} week{"s" if weeks != 1 else ""}'
        else:
            months = days // 30
            return f'in {months} month{"s" if months != 1 else ""}'
    else:  # days < 0
        days = abs(days)
        if days < 7:
            return f'{days} days ago'
        elif days < 30:
            weeks = days // 7
            return f'{weeks} week{"s" if weeks != 1 else ""} ago'
        else:
            months = days // 30
            return f'{months} month{"s" if months != 1 else ""} ago'

def get_meal_timing_suggestions(
    meal_count: int = 3,
    start_time: str = '07:00',
    end_time: str = '20:00'
) -> Dict[str, str]:
    """
    Generate suggested meal times
    
    Args:
        meal_count: Number of meals per day
        start_time: First meal time (HH:MM format)
        end_time: Last meal time (HH:MM format)
        
    Returns:
        Dictionary mapping meal types to times
    """
    from datetime import datetime, time
    
    # Parse times
    start_hour, start_min = map(int, start_time.split(':'))
    end_hour, end_min = map(int, end_time.split(':'))
    
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min
    
    # Calculate intervals
    total_minutes = end_minutes - start_minutes
    interval = total_minutes // (meal_count - 1) if meal_count > 1 else 0
    
    # Generate meal times
    meal_types = ['breakfast', 'lunch', 'dinner', 'snack1', 'snack2']
    meal_times = {}
    
    for i in range(meal_count):
        if i < len(meal_types):
            meal_minutes = start_minutes + (i * interval)
            hour = meal_minutes // 60
            minute = meal_minutes % 60
            meal_times[meal_types[i]] = f"{hour:02d}:{minute:02d}"
    
    return meal_times