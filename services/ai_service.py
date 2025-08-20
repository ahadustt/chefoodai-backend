"""
ChefoodAI Advanced AI Service
Integration with Google Vertex AI Gemini 2.0 Flash Thinking for premium features
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import structlog
from PIL import Image
import io
import base64

import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason

# Import safety settings with fallback for version compatibility
try:
    from vertexai.generative_models import HarmCategory, HarmBlockThreshold, SafetySettings
    SAFETY_AVAILABLE = True
except ImportError:
    # Fallback for older Vertex AI versions
    HarmCategory = None
    HarmBlockThreshold = None
    SafetySettings = None
    SAFETY_AVAILABLE = False
from google.cloud import aiplatform

from core.config import settings
from core.redis import cache
from middleware.logging import log_business_event
from services.prompt_engineering import PromptTemplates, PromptOptimizer
from services.cost_optimization import CostOptimizer
from services.safety_filters import SafetyFilters

logger = structlog.get_logger()


class AIModelType(Enum):
    """Available AI models with different capabilities"""
    GEMINI_2_FLASH_THINKING = "gemini-2.0-flash-thinking"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class AIRequestType(Enum):
    """Types of AI requests for different pricing and caching"""
    RECIPE_GENERATION = "recipe_generation"
    MEAL_PLANNING = "meal_planning"
    NUTRITION_ANALYSIS = "nutrition_analysis"
    IMAGE_ANALYSIS = "image_analysis"
    COOKING_GUIDANCE = "cooking_guidance"
    INGREDIENT_SUBSTITUTION = "ingredient_substitution"
    NAME_GENERATION = "name_generation"


@dataclass
class AIRequest:
    """Structured AI request with context and preferences"""
    request_type: AIRequestType
    user_id: str
    prompt: str
    context: Dict[str, Any]
    dietary_restrictions: List[str] = None
    preferences: Dict[str, Any] = None
    image_data: bytes = None
    video_data: bytes = None
    max_tokens: int = 4096
    temperature: float = 0.7
    model_type: AIModelType = AIModelType.GEMINI_2_FLASH_THINKING


@dataclass
class AIResponse:
    """Structured AI response with metadata"""
    content: str
    request_id: str
    model_used: str
    tokens_used: int
    processing_time: float
    cached: bool = False
    confidence_score: float = 0.0
    safety_ratings: Dict[str, str] = None
    thinking_process: str = None  # For Gemini 2.0 Flash Thinking


class ChefoodAIService:
    """
    Premium AI service for ChefoodAI using Gemini 2.0 Flash Thinking
    Provides advanced reasoning, multimodal capabilities, and cost optimization
    """
    
    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_AI_LOCATION
        )
        
        # Initialize models
        self.models = {
            AIModelType.GEMINI_2_FLASH_THINKING: GenerativeModel(
                "gemini-2.0-flash-thinking",
                system_instruction="You are ChefoodAI, a premium AI personal chef assistant with advanced reasoning capabilities. Provide detailed, accurate, and personalized cooking advice."
            ),
            AIModelType.GEMINI_1_5_PRO: GenerativeModel(
                "gemini-1.5-pro",
                system_instruction="You are ChefoodAI, an expert culinary assistant."
            ),
            AIModelType.GEMINI_1_5_FLASH: GenerativeModel(
                "gemini-1.5-flash",
                system_instruction="You are ChefoodAI, a helpful cooking assistant."
            )
        }
        
        # Safety settings with version compatibility
        if SAFETY_AVAILABLE:
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        else:
            self.safety_settings = {}
        
        # Initialize utility services
        self.prompt_templates = PromptTemplates()
        self.prompt_optimizer = PromptOptimizer()
        self.cost_optimizer = CostOptimizer()
        self.safety_filters = SafetyFilters()
        
        logger.info("ChefoodAI Service initialized with Gemini 2.0 Flash Thinking")
    
    async def generate_recipe(
        self, 
        request: AIRequest
    ) -> AIResponse:
        """
        Generate AI recipe with advanced reasoning and personalization
        Uses Gemini 2.0 Flash Thinking for complex recipe optimization
        """
        start_time = time.time()
        request_id = f"recipe_{request.user_id}_{int(start_time)}"
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await cache.get_json(cache_key, "ai_recipes")
            
            if cached_response:
                logger.info(f"Recipe cache hit for user {request.user_id}")
                return AIResponse(
                    content=cached_response["content"],
                    request_id=request_id,
                    model_used=cached_response["model_used"],
                    tokens_used=0,
                    processing_time=time.time() - start_time,
                    cached=True,
                    confidence_score=cached_response.get("confidence_score", 0.8)
                )
            
            # Validate and optimize request
            optimized_request = await self.cost_optimizer.optimize_request(request)
            
            # Build comprehensive prompt
            recipe_prompt = await self.prompt_templates.build_recipe_prompt(
                base_request=optimized_request.prompt,
                dietary_restrictions=optimized_request.dietary_restrictions,
                preferences=optimized_request.preferences,
                context=optimized_request.context
            )
            
            # Generate with thinking model
            model = self.models[optimized_request.model_type]
            
            generation_config = {
                "max_output_tokens": optimized_request.max_tokens,
                "temperature": optimized_request.temperature,
                "top_p": 0.9,
                "top_k": 40
            }
            
            # Generate response
            response = await self._generate_with_retry(
                model=model,
                prompt=recipe_prompt,
                config=generation_config,
                request_id=request_id
            )
            
            # Extract thinking process if available
            thinking_process = None
            content = response.text
            
            if "thinking>" in content:
                # Extract thinking tags for Gemini 2.0 Flash Thinking
                thinking_start = content.find("<thinking>")
                thinking_end = content.find("</thinking>")
                if thinking_start != -1 and thinking_end != -1:
                    thinking_process = content[thinking_start+10:thinking_end]
                    content = content[thinking_end+11:].strip()
            
            # Safety check
            safety_check = await self.safety_filters.check_recipe_safety(content)
            if not safety_check.is_safe:
                raise ValueError(f"Recipe failed safety check: {safety_check.reason}")
            
            # Parse and structure recipe
            structured_recipe = await self._structure_recipe_response(content)
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                structured_recipe, 
                request.dietary_restrictions
            )
            
            # Create response
            ai_response = AIResponse(
                content=json.dumps(structured_recipe, indent=2),
                request_id=request_id,
                model_used=optimized_request.model_type.value,
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                processing_time=time.time() - start_time,
                cached=False,
                confidence_score=confidence_score,
                thinking_process=thinking_process,
                safety_ratings=self._extract_safety_ratings(response)
            )
            
            # Cache response
            await self._cache_response(cache_key, ai_response, ttl=3600)
            
            # Log business event
            log_business_event("recipe_generated", {
                "user_id": request.user_id,
                "model_used": optimized_request.model_type.value,
                "tokens_used": ai_response.tokens_used,
                "processing_time": ai_response.processing_time,
                "confidence_score": confidence_score
            })
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Recipe generation failed: {str(e)}", user_id=request.user_id)
            # Return fallback recipe if available
            return await self._get_fallback_recipe(request, request_id)
    
    async def analyze_food_image(
        self,
        request: AIRequest
    ) -> AIResponse:
        """
        Analyze food images using multimodal AI capabilities
        Identifies ingredients, dishes, nutritional content, and cooking methods
        """
        start_time = time.time()
        request_id = f"image_{request.user_id}_{int(start_time)}"
        
        try:
            if not request.image_data:
                raise ValueError("No image data provided")
            
            # Process image
            image_part = await self._process_image_data(request.image_data)
            
            # Build multimodal prompt
            analysis_prompt = await self.prompt_templates.build_image_analysis_prompt(
                context=request.context,
                dietary_restrictions=request.dietary_restrictions
            )
            
            # Use Gemini 2.0 Flash Thinking for advanced visual reasoning
            model = self.models[AIModelType.GEMINI_2_FLASH_THINKING]
            
            # Generate analysis
            response = await self._generate_with_retry(
                model=model,
                prompt=[analysis_prompt, image_part],
                config={
                    "max_output_tokens": 2048,
                    "temperature": 0.3  # Lower temperature for accuracy
                },
                request_id=request_id
            )
            
            # Structure analysis result
            analysis_result = await self._structure_image_analysis(response.text)
            
            ai_response = AIResponse(
                content=json.dumps(analysis_result, indent=2),
                request_id=request_id,
                model_used=AIModelType.GEMINI_2_FLASH_THINKING.value,
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                processing_time=time.time() - start_time,
                confidence_score=analysis_result.get("confidence", 0.8)
            )
            
            log_business_event("image_analyzed", {
                "user_id": request.user_id,
                "processing_time": ai_response.processing_time,
                "confidence_score": ai_response.confidence_score
            })
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}", user_id=request.user_id)
            raise
    
    async def generate_meal_plan(
        self,
        request: AIRequest
    ) -> AIResponse:
        """
        Generate comprehensive meal plans with advanced nutritional optimization
        Uses complex reasoning for family coordination and dietary restrictions
        """
        start_time = time.time()
        request_id = f"meal_plan_{request.user_id}_{int(start_time)}"
        
        try:
            # Check for cached meal plan
            cache_key = self._generate_cache_key(request)
            cached_response = await cache.get_json(cache_key, "ai_meal_plans")
            
            if cached_response:
                logger.info(f"Meal plan cache hit for user {request.user_id}")
                return AIResponse(
                    content=cached_response["content"],
                    request_id=request_id,
                    model_used=cached_response["model_used"],
                    tokens_used=0,
                    processing_time=time.time() - start_time,
                    cached=True
                )
            
            # Build comprehensive meal planning prompt
            meal_plan_prompt = await self.prompt_templates.build_meal_plan_prompt(
                days=request.context.get("days", 7),
                dietary_restrictions=request.dietary_restrictions,
                preferences=request.preferences,
                context=request.context
            )
            
            # Use Gemini 2.0 Flash Thinking for complex meal optimization
            model = self.models[AIModelType.GEMINI_2_FLASH_THINKING]
            
            response = await self._generate_with_retry(
                model=model,
                prompt=meal_plan_prompt,
                config={
                    "max_output_tokens": 8192,  # Large context for comprehensive plans
                    "temperature": 0.7
                },
                request_id=request_id
            )
            
            # Structure meal plan
            structured_plan = await self._structure_meal_plan(response.text)
            
            # Validate nutritional balance
            nutrition_score = await self._validate_meal_plan_nutrition(structured_plan)
            
            ai_response = AIResponse(
                content=json.dumps(structured_plan, indent=2),
                request_id=request_id,
                model_used=AIModelType.GEMINI_2_FLASH_THINKING.value,
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                processing_time=time.time() - start_time,
                confidence_score=nutrition_score
            )
            
            # Cache meal plan
            await self._cache_response(cache_key, ai_response, ttl=7200)  # 2 hours
            
            log_business_event("meal_plan_generated", {
                "user_id": request.user_id,
                "days": request.context.get("days", 7),
                "nutrition_score": nutrition_score,
                "processing_time": ai_response.processing_time
            })
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Meal plan generation failed: {str(e)}", user_id=request.user_id)
            raise
    
    async def generate_meal_plan_name(
        self,
        request: AIRequest
    ) -> AIResponse:
        """
        Generate personalized, exciting meal plan names based on user preferences
        Uses AI to create unique names that reflect dietary goals and preferences
        """
        start_time = time.time()
        request_id = f"name_gen_{request.user_id}_{int(start_time)}"
        
        try:
            # Build personalized name generation prompt
            name_prompt = await self._build_name_generation_prompt(
                duration_days=request.context.get("duration_days", 7),
                family_size=request.context.get("family_size", 4),
                goals=request.context.get("goals", []),
                dietary_restrictions=request.dietary_restrictions or [],
                cuisine_preferences=request.preferences.get("cuisine_preferences", []) if request.preferences else [],
                skill_level=request.preferences.get("skill_level", "intermediate") if request.preferences else "intermediate"
            )
            
            # Use faster model for quick name generation
            model = self.models[AIModelType.GEMINI_1_5_FLASH]
            
            response = await self._generate_with_retry(
                model=model,
                prompt=name_prompt,
                config={
                    "max_output_tokens": 256,  # Short response for names
                    "temperature": 0.9  # Higher creativity for unique names
                },
                request_id=request_id
            )
            
            # Clean and validate the generated name
            name = response.text.strip().strip('"').strip("'")
            if len(name) > 50:  # Truncate if too long
                name = name[:47] + "..."
            
            ai_response = AIResponse(
                content=name,
                request_id=request_id,
                model_used=AIModelType.GEMINI_1_5_FLASH.value,
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                processing_time=time.time() - start_time,
                confidence_score=1.0
            )
            
            log_business_event("meal_plan_name_generated", {
                "user_id": request.user_id,
                "name_length": len(name),
                "processing_time": ai_response.processing_time
            })
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Meal plan name generation failed: {str(e)}", user_id=request.user_id)
            # Fallback to simple name if AI fails
            fallback_name = f"My {request.context.get('duration_days', 7)}-Day Plan"
            return AIResponse(
                content=fallback_name,
                request_id=request_id,
                model_used="fallback",
                tokens_used=0,
                processing_time=time.time() - start_time,
                confidence_score=0.5
            )
    
    async def stream_cooking_guidance(
        self,
        request: AIRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream real-time cooking guidance with step-by-step instructions
        Perfect for interactive cooking sessions
        """
        try:
            guidance_prompt = await self.prompt_templates.build_cooking_guidance_prompt(
                recipe=request.context.get("recipe"),
                current_step=request.context.get("current_step", 1),
                user_question=request.prompt
            )
            
            model = self.models[AIModelType.GEMINI_2_FLASH_THINKING]
            
            # Stream response for real-time guidance
            response_stream = model.generate_content(
                guidance_prompt,
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.5,
                    "candidate_count": 1
                },
                safety_settings=self.safety_settings,
                stream=True
            )
            
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Cooking guidance streaming failed: {str(e)}")
            yield f"I'm sorry, I encountered an error providing cooking guidance: {str(e)}"
    
    # Helper methods
    
    def _generate_cache_key(self, request: AIRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "type": request.request_type.value,
            "prompt": request.prompt[:100],  # First 100 chars
            "dietary_restrictions": sorted(request.dietary_restrictions or []),
            "preferences": request.preferences,
            "context": request.context
        }
        import hashlib
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def _generate_with_retry(
        self, 
        model: GenerativeModel, 
        prompt: Any, 
        config: Dict, 
        request_id: str,
        max_retries: int = 3
    ):
        """Generate with retry logic and fallback"""
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=config,
                    safety_settings=self.safety_settings
                )
                
                # Check for safety blocks
                if response.candidates[0].finish_reason == FinishReason.SAFETY:
                    raise ValueError("Content blocked by safety filters")
                
                return response
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Try fallback model
                    if model != self.models[AIModelType.GEMINI_1_5_FLASH]:
                        logger.info("Trying fallback model")
                        return await self._generate_with_retry(
                            self.models[AIModelType.GEMINI_1_5_FLASH],
                            prompt,
                            config,
                            request_id,
                            1
                        )
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _process_image_data(self, image_data: bytes) -> Part:
        """Process image data for multimodal input"""
        try:
            # Validate and optimize image
            image = Image.open(io.BytesIO(image_data))
            
            # Resize if too large (cost optimization)
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            processed_data = img_byte_arr.getvalue()
            
            return Part.from_data(processed_data, mime_type="image/jpeg")
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError("Invalid image data")
    
    async def _structure_recipe_response(self, content: str) -> Dict[str, Any]:
        """Structure raw AI response into recipe format"""
        try:
            # Try to parse as JSON first
            if content.strip().startswith("{"):
                return json.loads(content)
            
            # Otherwise, parse natural language response
            return await self.prompt_templates.parse_recipe_response(content)
            
        except Exception as e:
            logger.error(f"Failed to structure recipe: {str(e)}")
            return {"recipe": content, "parsed": False}
    
    async def _structure_image_analysis(self, content: str) -> Dict[str, Any]:
        """Structure image analysis response"""
        try:
            return await self.prompt_templates.parse_image_analysis(content)
        except Exception as e:
            logger.error(f"Failed to structure image analysis: {str(e)}")
            return {"analysis": content, "parsed": False}
    
    async def _structure_meal_plan(self, content: str) -> Dict[str, Any]:
        """Structure meal plan response"""
        try:
            return await self.prompt_templates.parse_meal_plan(content)
        except Exception as e:
            logger.error(f"Failed to structure meal plan: {str(e)}")
            return {"meal_plan": content, "parsed": False}
    
    async def _calculate_confidence_score(
        self, 
        recipe: Dict[str, Any], 
        dietary_restrictions: List[str]
    ) -> float:
        """Calculate confidence score for recipe quality"""
        score = 0.5  # Base score
        
        # Check recipe completeness
        if "ingredients" in recipe and len(recipe["ingredients"]) > 0:
            score += 0.2
        if "instructions" in recipe and len(recipe["instructions"]) > 0:
            score += 0.2
        if "nutrition" in recipe:
            score += 0.1
        
        # Check dietary restriction compliance
        if dietary_restrictions:
            # This would check against ingredient database
            # Placeholder logic
            score += 0.1
        
        return min(score, 1.0)
    
    async def _validate_meal_plan_nutrition(self, meal_plan: Dict[str, Any]) -> float:
        """Validate nutritional balance of meal plan"""
        # Placeholder nutritional validation
        # In real implementation, would calculate macro/micro balance
        return 0.85
    
    def _extract_safety_ratings(self, response) -> Dict[str, str]:
        """Extract safety ratings from response"""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'safety_ratings'):
                return {
                    rating.category.name: rating.probability.name 
                    for rating in candidate.safety_ratings
                }
        return {}
    
    async def _cache_response(self, cache_key: str, response: AIResponse, ttl: int):
        """Cache AI response"""
        cache_data = {
            "content": response.content,
            "model_used": response.model_used,
            "confidence_score": response.confidence_score,
            "cached_at": time.time()
        }
        await cache.set_json(cache_key, cache_data, ttl, "ai_responses")
    
    async def _build_name_generation_prompt(
        self,
        duration_days: int,
        family_size: int,
        goals: List[str],
        dietary_restrictions: List[str],
        cuisine_preferences: List[str],
        skill_level: str
    ) -> str:
        """Build a prompt for generating personalized meal plan names"""
        
        # Create context from user preferences
        duration_text = f"{duration_days} day" if duration_days == 1 else f"{duration_days} days"
        family_text = f"family of {family_size}" if family_size > 1 else "individual"
        
        goals_text = ", ".join(goals) if goals else "balanced nutrition"
        restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "no restrictions"
        cuisines_text = ", ".join(cuisine_preferences) if cuisine_preferences else "various cuisines"
        
        prompt = f"""Generate a creative, personalized, and exciting meal plan name based on these preferences:

Duration: {duration_text}
For: {family_text}  
Health Goals: {goals_text}
Dietary Preferences: {restrictions_text}
Preferred Cuisines: {cuisines_text}
Cooking Level: {skill_level}

Requirements:
- Create ONE unique, catchy name (2-6 words)
- Make it personal and motivating
- Reflect their goals and preferences
- Avoid generic words like "plan" or "menu"
- Make it sound exciting and achievable
- Maximum 50 characters

Examples of good names:
- "Mediterranean Wellness Journey"
- "Keto Family Feast Adventure" 
- "Plant-Powered Energy Week"
- "Quick & Healthy Victory"
- "Italian Comfort Revival"

Generate just the name, nothing else:"""

        return prompt
    
    async def _get_fallback_recipe(self, request: AIRequest, request_id: str) -> AIResponse:
        """Get fallback recipe when AI generation fails"""
        fallback_content = {
            "name": "Simple Pasta Recipe",
            "description": "A basic pasta recipe when AI is unavailable",
            "ingredients": [
                "400g pasta",
                "2 tbsp olive oil",
                "2 cloves garlic",
                "Salt and pepper to taste"
            ],
            "instructions": [
                "Boil water and cook pasta according to package instructions",
                "Heat olive oil and sauté minced garlic",
                "Combine pasta with garlic oil",
                "Season with salt and pepper"
            ],
            "fallback": True
        }
        
        return AIResponse(
            content=json.dumps(fallback_content, indent=2),
            request_id=request_id,
            model_used="fallback",
            tokens_used=0,
            processing_time=0.1,
            confidence_score=0.3
        )


# Global AI service instance
ai_service = ChefoodAIService()