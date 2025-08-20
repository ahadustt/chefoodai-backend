"""
ChefoodAI Prompt Engineering
Advanced prompt templates optimized for Gemini 2.0 Flash Thinking
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class CuisineType(Enum):
    """Supported cuisine types"""
    ITALIAN = "italian"
    MEXICAN = "mexican"
    ASIAN = "asian"
    MEDITERRANEAN = "mediterranean"
    AMERICAN = "american"
    INDIAN = "indian"
    FRENCH = "french"
    MIDDLE_EASTERN = "middle_eastern"
    FUSION = "fusion"


class DifficultyLevel(Enum):
    """Recipe difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class RecipeContext:
    """Context for recipe generation"""
    cuisine_type: Optional[CuisineType] = None
    difficulty: Optional[DifficultyLevel] = None
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    servings: Optional[int] = None
    budget_level: Optional[str] = None  # low, medium, high
    equipment_available: Optional[List[str]] = None


class PromptTemplates:
    """
    Advanced prompt templates for different AI operations
    Optimized for Gemini 2.0 Flash Thinking reasoning capabilities
    """
    
    def __init__(self):
        self.base_system_prompt = """
        You are ChefoodAI, a world-class AI personal chef assistant with advanced culinary knowledge and reasoning capabilities. 
        You have expertise in:
        - International cuisines and cooking techniques
        - Nutritional science and dietary requirements
        - Food safety and preparation methods
        - Ingredient substitutions and flavor pairing
        - Meal planning and portion control
        - Cost optimization and seasonal cooking
        
        Always provide detailed, accurate, and personalized responses. Use your thinking process to reason through complex culinary challenges.
        """
    
    async def build_recipe_prompt(
        self,
        base_request: str,
        dietary_restrictions: List[str] = None,
        preferences: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Build comprehensive recipe generation prompt with advanced reasoning
        """
        prompt_parts = [
            self.base_system_prompt,
            "\n<thinking>",
            "Let me think through this recipe request step by step:",
            "1. Analyze the user's request and preferences",
            "2. Consider dietary restrictions and nutritional goals", 
            "3. Select appropriate ingredients and techniques",
            "4. Optimize for taste, nutrition, and practicality",
            "5. Ensure food safety and proper cooking methods",
            "</thinking>\n",
            
            f"**Recipe Request**: {base_request}\n"
        ]
        
        # Add dietary restrictions
        if dietary_restrictions:
            restrictions_text = ", ".join(dietary_restrictions)
            prompt_parts.extend([
                f"**Dietary Restrictions**: {restrictions_text}",
                "CRITICAL: Ensure the recipe completely avoids all restricted ingredients and follows safe substitution practices.\n"
            ])
        
        # Add preferences
        if preferences:
            prompt_parts.append("**User Preferences**:")
            for key, value in preferences.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        # Add context
        if context:
            if context.get("cooking_skill"):
                prompt_parts.append(f"**Cooking Skill Level**: {context['cooking_skill']}")
            if context.get("available_time"):
                prompt_parts.append(f"**Available Time**: {context['available_time']} minutes")
            if context.get("servings"):
                prompt_parts.append(f"**Number of Servings**: {context['servings']}")
            if context.get("budget"):
                prompt_parts.append(f"**Budget Level**: {context['budget']}")
            if context.get("equipment"):
                equipment_list = ", ".join(context['equipment'])
                prompt_parts.append(f"**Available Equipment**: {equipment_list}")
            prompt_parts.append("")
        
        # Add output format requirements
        prompt_parts.extend([
            "**Output Requirements**:",
            "Provide a complete recipe in the following JSON format:",
            "```json",
            "{",
            '  "name": "Recipe Name",',
            '  "description": "Brief appetizing description",',
            '  "cuisine_type": "cuisine category",',
            '  "difficulty": "beginner/intermediate/advanced",',
            '  "prep_time_minutes": 15,',
            '  "cook_time_minutes": 30,', 
            '  "total_time_minutes": 45,',
            '  "servings": 4,',
            '  "ingredients": [',
            '    {',
            '      "name": "ingredient name",',
            '      "amount": "1 cup",',
            '      "notes": "preparation notes if needed"',
            '    }',
            '  ],',
            '  "instructions": [',
            '    {',
            '      "step": 1,',
            '      "instruction": "Detailed step-by-step instruction",',
            '      "time_minutes": 5,',
            '      "temperature": "if applicable"',
            '    }',
            '  ],',
            '  "nutrition_per_serving": {',
            '    "calories": 350,',
            '    "protein_g": 25,',
            '    "carbs_g": 30,',
            '    "fat_g": 15,',
            '    "fiber_g": 5',
            '  },',
            '  "tips": [',
            '    "Helpful cooking tips and variations"',
            '  ],',
            '  "dietary_tags": ["vegetarian", "gluten-free", etc.],',
            '  "difficulty_notes": "Why this difficulty level",',
            '  "ingredient_substitutions": {',
            '    "original_ingredient": "substitute_ingredient with notes"',
            '  }',
            "}",
            "```",
            "",
            "Ensure all measurements are precise, instructions are clear, and the recipe is delicious and achievable."
        ])
        
        return "\n".join(prompt_parts)
    
    async def build_meal_plan_prompt(
        self,
        days: int,
        dietary_restrictions: List[str] = None,
        preferences: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Build comprehensive meal planning prompt with nutritional optimization
        """
        prompt_parts = [
            self.base_system_prompt,
            "\n<thinking>",
            "Let me create a comprehensive meal plan by considering:",
            "1. Nutritional balance across all meals and days",
            "2. Variety in ingredients, cuisines, and cooking methods",
            "3. Dietary restrictions and health goals",
            "4. Practical considerations like prep time and shopping",
            "5. Cost efficiency and seasonal ingredient availability",
            "6. Family preferences and portion sizes",
            "</thinking>\n",
            
            f"Create a detailed {days}-day meal plan that is nutritionally balanced, varied, and delicious.\n"
        ]
        
        # Add dietary restrictions
        if dietary_restrictions:
            restrictions_text = ", ".join(dietary_restrictions)
            prompt_parts.extend([
                f"**Dietary Restrictions**: {restrictions_text}",
                "CRITICAL: All meals must comply with these restrictions.\n"
            ])
        
        # Add preferences and context
        if preferences:
            prompt_parts.append("**Preferences**:")
            for key, value in preferences.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        if context:
            if context.get("family_size"):
                prompt_parts.append(f"**Family Size**: {context['family_size']} people")
            if context.get("budget_per_week"):
                prompt_parts.append(f"**Weekly Budget**: ${context['budget_per_week']}")
            if context.get("health_goals"):
                goals = ", ".join(context['health_goals'])
                prompt_parts.append(f"**Health Goals**: {goals}")
            if context.get("cooking_time_available"):
                prompt_parts.append(f"**Available Cooking Time**: {context['cooking_time_available']} minutes per meal")
            prompt_parts.append("")
        
        # Add output format
        prompt_parts.extend([
            "**Output Format** (JSON):",
            "```json",
            "{",
            '  "meal_plan": {',
            '    "overview": {',
            '      "total_days": 7,',
            '      "estimated_cost": "$120",',
            '      "prep_time_per_day": "45 minutes",',
            '      "nutritional_highlights": ["high protein", "balanced macros"]',
            '    },',
            '    "daily_plans": [',
            '      {',
            '        "day": 1,',
            '        "date": "2024-01-01",',
            '        "meals": {',
            '          "breakfast": {',
            '            "name": "Meal Name",',
            '            "prep_time": 15,',
            '            "ingredients": ["ingredient list"],',
            '            "nutrition": {"calories": 400, "protein": 20}',
            '          },',
            '          "lunch": {...},',
            '          "dinner": {...},',
            '          "snacks": [...]',
            '        },',
            '        "daily_nutrition": {',
            '          "calories": 2000,',
            '          "protein_g": 150,',
            '          "carbs_g": 200,',
            '          "fat_g": 80',
            '        }',
            '      }',
            '    ]',
            '  },',
            '  "shopping_list": {',
            '    "organized_by_category": {',
            '      "produce": ["apples", "spinach"],',
            '      "proteins": ["chicken breast", "eggs"],',
            '      "pantry": ["olive oil", "spices"]',
            '    },',
            '    "estimated_total": "$120"',
            '  },',
            '  "prep_schedule": [',
            '    {',
            '      "day": "Sunday",',
            '      "tasks": ["prep vegetables", "marinate proteins"],',
            '      "time_required": "2 hours"',
            '    }',
            '  ],',
            '  "nutritional_analysis": {',
            '    "daily_averages": {"calories": 2000, "protein": 150},',
            '    "weekly_variety_score": 0.85,',
            '    "dietary_compliance": "100%"',
            '  }',
            "}",
            "```"
        ])
        
        return "\n".join(prompt_parts)
    
    async def build_image_analysis_prompt(
        self,
        context: Dict[str, Any] = None,
        dietary_restrictions: List[str] = None
    ) -> str:
        """
        Build prompt for multimodal food image analysis
        """
        prompt_parts = [
            self.base_system_prompt,
            "\n<thinking>",
            "I need to analyze this food image by:",
            "1. Identifying all visible ingredients and dishes",
            "2. Estimating nutritional content and portion sizes",
            "3. Assessing cooking methods and techniques used",
            "4. Suggesting improvements or variations",
            "5. Checking against any dietary restrictions",
            "</thinking>\n",
            
            "Analyze this food image in detail and provide comprehensive insights.\n"
        ]
        
        if dietary_restrictions:
            restrictions_text = ", ".join(dietary_restrictions)
            prompt_parts.extend([
                f"**Check for Dietary Compliance**: {restrictions_text}",
                "Identify any ingredients that may violate these restrictions.\n"
            ])
        
        if context and context.get("analysis_focus"):
            prompt_parts.append(f"**Focus Area**: {context['analysis_focus']}\n")
        
        prompt_parts.extend([
            "**Provide Analysis in JSON Format**:",
            "```json",
            "{",
            '  "dish_identification": {',
            '    "primary_dish": "dish name",',
            '    "cuisine_type": "cuisine category",',
            '    "confidence_score": 0.95',
            '  },',
            '  "ingredients_detected": [',
            '    {',
            '      "name": "ingredient name",',
            '      "confidence": 0.90,',
            '      "estimated_amount": "1 cup",',
            '      "preparation_method": "diced/grilled/etc"',
            '    }',
            '  ],',
            '  "nutritional_estimate": {',
            '    "calories_per_serving": 450,',
            '    "protein_g": 25,',
            '    "carbs_g": 35,',
            '    "fat_g": 20,',
            '    "portion_size": "medium"',
            '  },',
            '  "cooking_analysis": {',
            '    "cooking_methods": ["grilled", "sautéed"],',
            '    "doneness_level": "medium",',
            '    "presentation_quality": "excellent"',
            '  },',
            '  "dietary_flags": {',
            '    "vegetarian": true,',
            '    "gluten_free": false,',
            '    "restrictions_violated": []',
            '  },',
            '  "suggestions": {',
            '    "improvements": ["suggestion 1", "suggestion 2"],',
            '    "variations": ["variation 1", "variation 2"],',
            '    "wine_pairing": "wine recommendation"',
            '  },',
            '  "recipe_generation": {',
            '    "can_recreate": true,',
            '    "difficulty_level": "intermediate",',
            '    "estimated_time": "45 minutes"',
            '  }',
            "}",
            "```"
        ])
        
        return "\n".join(prompt_parts)
    
    async def build_cooking_guidance_prompt(
        self,
        recipe: Dict[str, Any],
        current_step: int,
        user_question: str
    ) -> str:
        """
        Build prompt for real-time cooking guidance
        """
        recipe_name = recipe.get("name", "your recipe")
        total_steps = len(recipe.get("instructions", []))
        
        prompt_parts = [
            "You are providing real-time cooking guidance as an expert chef assistant.",
            f"The user is cooking: {recipe_name}",
            f"They are on step {current_step} of {total_steps}.",
            f"User's question: {user_question}",
            "",
            "Provide helpful, encouraging, and practical guidance. Be concise but thorough.",
            "If there are any safety concerns, mention them immediately.",
            "Adapt your advice to their current cooking stage and skill level."
        ]
        
        if current_step <= len(recipe.get("instructions", [])):
            current_instruction = recipe["instructions"][current_step - 1]
            prompt_parts.extend([
                f"Current step they should be on: {current_instruction.get('instruction', 'N/A')}",
                ""
            ])
        
        return "\n".join(prompt_parts)
    
    # Response parsing methods
    
    async def parse_recipe_response(self, content: str) -> Dict[str, Any]:
        """Parse natural language recipe response into structured format"""
        try:
            # Extract JSON if present
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Otherwise parse natural language
            return await self._parse_natural_language_recipe(content)
            
        except Exception as e:
            logger.error(f"Recipe parsing failed: {str(e)}")
            return {"raw_content": content, "parsed": False}
    
    async def parse_image_analysis(self, content: str) -> Dict[str, Any]:
        """Parse image analysis response"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            return await self._parse_natural_language_analysis(content)
            
        except Exception as e:
            logger.error(f"Image analysis parsing failed: {str(e)}")
            return {"raw_content": content, "parsed": False}
    
    async def parse_meal_plan(self, content: str) -> Dict[str, Any]:
        """Parse meal plan response"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            return await self._parse_natural_language_meal_plan(content)
            
        except Exception as e:
            logger.error(f"Meal plan parsing failed: {str(e)}")
            return {"raw_content": content, "parsed": False}
    
    # Private parsing helpers
    
    async def _parse_natural_language_recipe(self, content: str) -> Dict[str, Any]:
        """Parse natural language recipe into structured format"""
        # Implement natural language parsing logic
        # This is a simplified version - in production would use more sophisticated NLP
        
        lines = content.split('\n')
        recipe = {
            "name": "Generated Recipe",
            "description": "",
            "ingredients": [],
            "instructions": [],
            "parsed": True
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if line.lower().startswith(('ingredients:', 'ingredient')):
                current_section = 'ingredients'
                continue
            elif line.lower().startswith(('instructions:', 'steps:', 'directions:')):
                current_section = 'instructions'
                continue
            elif line.lower().startswith(('recipe:', 'dish:')):
                recipe["name"] = line.split(':', 1)[1].strip()
                continue
            
            # Parse content
            if current_section == 'ingredients' and line.startswith(('- ', '* ', '• ')):
                recipe["ingredients"].append(line[2:].strip())
            elif current_section == 'instructions' and (line.startswith(('1.', '2.', '3.')) or line.startswith(('- ', '* '))):
                recipe["instructions"].append(line.strip())
        
        return recipe
    
    async def _parse_natural_language_analysis(self, content: str) -> Dict[str, Any]:
        """Parse natural language image analysis"""
        return {
            "analysis": content,
            "confidence": 0.7,
            "parsed": True
        }
    
    async def _parse_natural_language_meal_plan(self, content: str) -> Dict[str, Any]:
        """Parse natural language meal plan"""
        return {
            "meal_plan": content,
            "parsed": True
        }


class PromptOptimizer:
    """
    Optimize prompts for better performance and cost efficiency
    """
    
    def __init__(self):
        self.optimization_strategies = {
            "token_reduction": True,
            "context_pruning": True,
            "template_caching": True
        }
    
    async def optimize_prompt(self, prompt: str, max_tokens: int = 8192) -> str:
        """Optimize prompt for token efficiency"""
        # Remove excessive whitespace
        optimized = re.sub(r'\n\s*\n', '\n\n', prompt)
        optimized = re.sub(r'[ \t]+', ' ', optimized)
        
        # Truncate if too long (rough token estimation: 1 token ≈ 4 chars)
        estimated_tokens = len(optimized) // 4
        if estimated_tokens > max_tokens * 0.8:  # Leave room for response
            # Truncate while preserving structure
            optimized = optimized[:max_tokens * 3]  # Rough char limit
            logger.warning(f"Prompt truncated to fit token limit")
        
        return optimized
    
    async def get_optimal_model_for_request(self, request_type: str, content_length: int) -> str:
        """Recommend optimal model based on request characteristics"""
        if request_type == "recipe_generation" and content_length < 1000:
            return "gemini-1.5-flash"  # Faster and cheaper for simple recipes
        elif request_type == "meal_planning":
            return "gemini-2.0-flash-thinking"  # Complex reasoning needed
        elif request_type == "image_analysis":
            return "gemini-2.0-flash-thinking"  # Multimodal capabilities
        else:
            return "gemini-1.5-pro"  # Balanced option