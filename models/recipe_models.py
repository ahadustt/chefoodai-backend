"""
ChefoodAI Recipe Models
Database models for recipes and ingredients
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base


class Recipe(Base):
    """Recipe model for storing recipe information"""
    __tablename__ = "recipes"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    cuisine = Column(String(100))
    difficulty = Column(String(50))
    prep_time = Column(Integer)  # in minutes
    cook_time = Column(Integer)  # in minutes
    servings = Column(Integer, default=4)
    calories_per_serving = Column(Integer)
    
    # JSON fields for complex data
    instructions = Column(JSON)  # List of instruction steps
    nutrition_info = Column(JSON)  # Detailed nutrition breakdown
    tags = Column(JSON)  # List of tags
    dietary_restrictions = Column(JSON)  # List of dietary restrictions
    
    # Image and AI generation
    image_url = Column(String(500))
    ai_generated = Column(Boolean, default=False)
    ai_model_used = Column(String(100))
    
    # User relationship
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="recipes")
    
    # Ratings and popularity
    rating = Column(Float, default=0.0)
    total_ratings = Column(Integer, default=0)
    times_cooked = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    ingredients = relationship("RecipeIngredient", back_populates="recipe", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert recipe to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "cuisine": self.cuisine,
            "difficulty": self.difficulty,
            "prep_time": self.prep_time,
            "cook_time": self.cook_time,
            "servings": self.servings,
            "calories_per_serving": self.calories_per_serving,
            "instructions": self.instructions,
            "nutrition_info": self.nutrition_info,
            "tags": self.tags,
            "dietary_restrictions": self.dietary_restrictions,
            "image_url": self.image_url,
            "rating": self.rating,
            "ingredients": [ing.to_dict() for ing in self.ingredients],
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class RecipeIngredient(Base):
    """Recipe ingredient model for storing ingredient details"""
    __tablename__ = "recipe_ingredients"
    
    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"))
    name = Column(String(255), nullable=False)
    quantity = Column(Float)
    unit = Column(String(50))
    notes = Column(String(255))
    category = Column(String(100))  # e.g., "protein", "vegetable", "spice"
    
    # Relationships
    recipe = relationship("Recipe", back_populates="ingredients")
    
    def to_dict(self):
        """Convert ingredient to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "notes": self.notes,
            "category": self.category
        }


