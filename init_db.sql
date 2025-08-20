-- ChefoodAI Database Initialization Script
-- Creates all necessary schemas, tables, and initial data

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS chefoodai;
USE chefoodai;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS recipes;
CREATE SCHEMA IF NOT EXISTS meal_planning;
CREATE SCHEMA IF NOT EXISTS shopping;
CREATE SCHEMA IF NOT EXISTS nutrition;
CREATE SCHEMA IF NOT EXISTS audit;

-- Users table
CREATE TABLE IF NOT EXISTS auth.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255),
    full_name VARCHAR(255),
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    is_premium BOOLEAN DEFAULT false,
    subscription_tier VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    last_login TIMESTAMP,
    login_count INTEGER DEFAULT 0,
    preferences JSONB DEFAULT '{}',
    dietary_restrictions TEXT[],
    allergens TEXT[],
    INDEX idx_users_email (email),
    INDEX idx_users_username (username)
);

-- Sessions table
CREATE TABLE IF NOT EXISTS auth.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    token VARCHAR(500) UNIQUE NOT NULL,
    refresh_token VARCHAR(500) UNIQUE,
    device_info JSONB,
    ip_address INET,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sessions_token (token),
    INDEX idx_sessions_user (user_id)
);

-- Organizations table
CREATE TABLE IF NOT EXISTS core.organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    owner_id UUID REFERENCES auth.users(id),
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_org_slug (slug)
);

-- Recipes table
CREATE TABLE IF NOT EXISTS recipes.recipes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    organization_id UUID REFERENCES core.organizations(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    ingredients JSONB NOT NULL,
    instructions JSONB NOT NULL,
    prep_time INTEGER,
    cook_time INTEGER,
    total_time INTEGER,
    servings INTEGER DEFAULT 4,
    difficulty VARCHAR(50),
    cuisine VARCHAR(100),
    category VARCHAR(100),
    tags TEXT[],
    image_url TEXT,
    nutrition_data JSONB,
    rating DECIMAL(3,2),
    rating_count INTEGER DEFAULT 0,
    is_public BOOLEAN DEFAULT false,
    is_ai_generated BOOLEAN DEFAULT false,
    ai_model VARCHAR(100),
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    INDEX idx_recipes_user (user_id),
    INDEX idx_recipes_title (title),
    FULLTEXT INDEX idx_recipes_search (title, description)
);

-- Meal Plans table
CREATE TABLE IF NOT EXISTS meal_planning.meal_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_days INTEGER NOT NULL,
    meals_per_day INTEGER DEFAULT 3,
    target_calories INTEGER,
    nutritional_goals JSONB,
    dietary_restrictions TEXT[],
    is_active BOOLEAN DEFAULT true,
    is_ai_optimized BOOLEAN DEFAULT false,
    optimization_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_meal_plans_user (user_id),
    INDEX idx_meal_plans_dates (start_date, end_date)
);

-- Meal Plan Days table
CREATE TABLE IF NOT EXISTS meal_planning.meal_plan_days (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meal_plan_id UUID REFERENCES meal_planning.meal_plans(id) ON DELETE CASCADE,
    day_number INTEGER NOT NULL,
    date DATE NOT NULL,
    total_calories INTEGER,
    total_protein DECIMAL(10,2),
    total_carbs DECIMAL(10,2),
    total_fat DECIMAL(10,2),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_meal_plan_days (meal_plan_id, day_number)
);

-- Meal Plan Meals table
CREATE TABLE IF NOT EXISTS meal_planning.meal_plan_meals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meal_plan_day_id UUID REFERENCES meal_planning.meal_plan_days(id) ON DELETE CASCADE,
    recipe_id UUID REFERENCES recipes.recipes(id),
    meal_type VARCHAR(50) NOT NULL, -- breakfast, lunch, dinner, snack
    custom_meal_data JSONB, -- For non-recipe meals
    servings INTEGER DEFAULT 1,
    calories INTEGER,
    notes TEXT,
    order_index INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_meal_plan_meals (meal_plan_day_id, meal_type)
);

-- Shopping Lists table
CREATE TABLE IF NOT EXISTS shopping.shopping_lists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    meal_plan_id UUID REFERENCES meal_planning.meal_plans(id),
    name VARCHAR(255) NOT NULL,
    items JSONB NOT NULL,
    categorized_items JSONB,
    optimized_route JSONB,
    estimated_cost DECIMAL(10,2),
    actual_cost DECIMAL(10,2),
    store_name VARCHAR(255),
    shopping_date DATE,
    is_completed BOOLEAN DEFAULT false,
    is_ai_optimized BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_shopping_lists_user (user_id),
    INDEX idx_shopping_lists_meal_plan (meal_plan_id)
);

-- Nutritional Goals table
CREATE TABLE IF NOT EXISTS nutrition.nutritional_goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    name VARCHAR(255) NOT NULL,
    daily_calories INTEGER,
    daily_protein DECIMAL(10,2),
    daily_carbs DECIMAL(10,2),
    daily_fat DECIMAL(10,2),
    daily_fiber DECIMAL(10,2),
    daily_sugar_limit DECIMAL(10,2),
    daily_sodium_limit DECIMAL(10,2),
    vitamin_targets JSONB,
    mineral_targets JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_nutritional_goals_user (user_id)
);

-- Analytics table
CREATE TABLE IF NOT EXISTS meal_planning.meal_plan_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meal_plan_id UUID REFERENCES meal_planning.meal_plans(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id),
    avg_daily_calories DECIMAL(10,2),
    avg_daily_protein DECIMAL(10,2),
    avg_daily_carbs DECIMAL(10,2),
    avg_daily_fat DECIMAL(10,2),
    total_unique_recipes INTEGER,
    total_ingredients INTEGER,
    estimated_total_cost DECIMAL(10,2),
    actual_total_cost DECIMAL(10,2),
    adherence_score DECIMAL(3,2),
    user_rating INTEGER,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_analytics_meal_plan (meal_plan_id),
    INDEX idx_analytics_user (user_id)
);

-- Audit Log table
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    changes JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_user (user_id),
    INDEX idx_audit_entity (entity_type, entity_id),
    INDEX idx_audit_created (created_at)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_recipes_created ON recipes.recipes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meal_plans_active ON meal_planning.meal_plans(is_active, user_id);
CREATE INDEX IF NOT EXISTS idx_shopping_lists_pending ON shopping.shopping_lists(is_completed, user_id);

-- Create triggers for updated_at
DELIMITER $$

CREATE TRIGGER update_users_timestamp 
BEFORE UPDATE ON auth.users 
FOR EACH ROW 
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END$$

CREATE TRIGGER update_recipes_timestamp 
BEFORE UPDATE ON recipes.recipes 
FOR EACH ROW 
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END$$

CREATE TRIGGER update_meal_plans_timestamp 
BEFORE UPDATE ON meal_planning.meal_plans 
FOR EACH ROW 
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END$$

DELIMITER ;

-- Insert default data
INSERT INTO auth.users (email, username, full_name, is_verified, is_premium, subscription_tier)
VALUES 
    ('demo@chefoodai.com', 'demo', 'Demo User', true, false, 'free'),
    ('premium@chefoodai.com', 'premium_demo', 'Premium Demo User', true, true, 'premium')
ON CONFLICT DO NOTHING;

-- Grant permissions (PostgreSQL specific, adjust for MySQL)
-- GRANT ALL PRIVILEGES ON DATABASE chefoodai TO chefoodai_user;
-- GRANT ALL ON SCHEMA auth TO chefoodai_user;
-- GRANT ALL ON SCHEMA core TO chefoodai_user;
-- GRANT ALL ON SCHEMA recipes TO chefoodai_user;
-- GRANT ALL ON SCHEMA meal_planning TO chefoodai_user;
-- GRANT ALL ON SCHEMA shopping TO chefoodai_user;
-- GRANT ALL ON SCHEMA nutrition TO chefoodai_user;
-- GRANT ALL ON SCHEMA audit TO chefoodai_user;