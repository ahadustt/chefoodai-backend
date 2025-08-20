"""
ChefoodAI AI Service Endpoints
Premium Gemini 2.0 Flash Thinking integration for advanced AI features
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import structlog

from services import (
    ai_service, AIRequest, AIResponse, AIRequestType, AIModelType,
    cost_optimizer, safety_filters
)
from middleware.logging import log_business_event
from core.database import get_db

logger = structlog.get_logger()
router = APIRouter()


# Request/Response Models
class RecipeGenerationRequest(BaseModel):
    """Request model for recipe generation"""
    prompt: str = Field(..., description="Recipe request description")
    dietary_restrictions: Optional[List[str]] = Field(default=[], description="Dietary restrictions and allergies")
    preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences")
    cuisine_type: Optional[str] = Field(default=None, description="Preferred cuisine type")
    difficulty: Optional[str] = Field(default="intermediate", description="Recipe difficulty level")
    servings: Optional[int] = Field(default=4, description="Number of servings")
    prep_time_max: Optional[int] = Field(default=60, description="Max prep time in minutes")
    cooking_skill: Optional[str] = Field(default="intermediate", description="User's cooking skill level")
    available_equipment: Optional[List[str]] = Field(default=[], description="Available kitchen equipment")
    budget_level: Optional[str] = Field(default="medium", description="Budget level: low, medium, high")


class MealPlanRequest(BaseModel):
    """Request model for meal plan generation"""
    days: int = Field(default=7, description="Number of days for meal plan")
    dietary_restrictions: Optional[List[str]] = Field(default=[], description="Dietary restrictions")
    health_goals: Optional[List[str]] = Field(default=[], description="Health and fitness goals")
    family_size: Optional[int] = Field(default=2, description="Number of people")
    budget_per_week: Optional[float] = Field(default=150.0, description="Weekly grocery budget")
    cooking_time_available: Optional[int] = Field(default=45, description="Available cooking time per meal")
    preferences: Optional[Dict[str, Any]] = Field(default={}, description="Additional preferences")


class ImageAnalysisResponse(BaseModel):
    """Response model for image analysis"""
    dish_identification: Dict[str, Any]
    ingredients_detected: List[Dict[str, Any]]
    nutritional_estimate: Dict[str, Any]
    cooking_analysis: Dict[str, Any]
    dietary_flags: Dict[str, Any]
    suggestions: Dict[str, Any]
    recipe_generation: Dict[str, Any]


class CookingGuidanceRequest(BaseModel):
    """Request model for cooking guidance"""
    recipe_id: Optional[str] = Field(default=None, description="Recipe being cooked")
    current_step: int = Field(default=1, description="Current cooking step")
    question: str = Field(..., description="User's cooking question")
    recipe_data: Optional[Dict[str, Any]] = Field(default=None, description="Recipe data if not using ID")


class MealPlanNameRequest(BaseModel):
    """Request model for AI-generated meal plan names"""
    duration_days: int = Field(..., description="Number of days for the meal plan")
    family_size: int = Field(..., description="Number of people")
    goals: List[str] = Field(default=[], description="Health and nutrition goals")
    dietary_restrictions: List[str] = Field(default=[], description="Dietary restrictions")
    cuisine_preferences: List[str] = Field(default=[], description="Preferred cuisine types")
    skill_level: str = Field(default="intermediate", description="Cooking skill level")


@router.post("/generate-recipe", response_model=Dict[str, Any])
async def generate_recipe(
    request: RecipeGenerationRequest,
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Generate AI recipe with Gemini 2.0 Flash Thinking
    Uses advanced reasoning for personalized recipe creation
    """
    try:
        # Validate user quota
        usage = await cost_optimizer.get_usage_analytics(user_id)
        if usage["daily"]["remaining"] <= 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Daily AI quota exceeded",
                    "quota_reset": "24 hours",
                    "upgrade_url": "/pricing"
                }
            )
        
        # Build AI request
        context = {
            "cuisine_type": request.cuisine_type,
            "difficulty": request.difficulty,
            "servings": request.servings,
            "prep_time_max": request.prep_time_max,
            "cooking_skill": request.cooking_skill,
            "available_equipment": request.available_equipment,
            "budget_level": request.budget_level
        }
        
        ai_request = AIRequest(
            request_type=AIRequestType.RECIPE_GENERATION,
            user_id=user_id,
            prompt=request.prompt,
            context=context,
            dietary_restrictions=request.dietary_restrictions,
            preferences=request.preferences,
            model_type=AIModelType.GEMINI_2_FLASH_THINKING
        )
        
        # Generate recipe
        logger.info(f"Generating recipe for user {user_id}")
        response = await ai_service.generate_recipe(ai_request)
        
        # Track usage
        await cost_optimizer.track_usage(
            user_id=user_id,
            model=response.model_used,
            input_tokens=response.tokens_used // 2,  # Rough estimate
            output_tokens=response.tokens_used // 2,
            cached=response.cached
        )
        
        # Parse response content
        try:
            import json
            recipe_data = json.loads(response.content)
        except json.JSONDecodeError:
            recipe_data = {"recipe": response.content, "parsed": False}
        
        return {
            "success": True,
            "recipe": recipe_data,
            "metadata": {
                "request_id": response.request_id,
                "model_used": response.model_used,
                "processing_time": response.processing_time,
                "confidence_score": response.confidence_score,
                "cached": response.cached,
                "thinking_process": response.thinking_process
            },
            "usage": await cost_optimizer.get_usage_analytics(user_id)
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"Recipe generation failed: {str(e)}", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Recipe generation failed", "message": str(e)}
        )


@router.post("/analyze-image", response_model=Dict[str, Any])
async def analyze_food_image(
    file: UploadFile = File(...),
    dietary_restrictions: Optional[str] = None,
    analysis_focus: Optional[str] = None,
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Analyze food images using multimodal AI capabilities
    Identifies ingredients, dishes, nutritional content, and cooking methods
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Only image files are supported"}
            )
        
        # Check file size (10MB limit)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"error": "Image file too large (max 10MB)"}
            )
        
        # Parse dietary restrictions
        restrictions = []
        if dietary_restrictions:
            restrictions = [r.strip() for r in dietary_restrictions.split(",")]
        
        # Build AI request
        context = {}
        if analysis_focus:
            context["analysis_focus"] = analysis_focus
        
        ai_request = AIRequest(
            request_type=AIRequestType.IMAGE_ANALYSIS,
            user_id=user_id,
            prompt="Analyze this food image in detail",
            context=context,
            dietary_restrictions=restrictions,
            image_data=file_content,
            model_type=AIModelType.GEMINI_2_FLASH_THINKING
        )
        
        # Analyze image
        logger.info(f"Analyzing image for user {user_id}")
        response = await ai_service.analyze_food_image(ai_request)
        
        # Track usage
        await cost_optimizer.track_usage(
            user_id=user_id,
            model=response.model_used,
            input_tokens=response.tokens_used // 2,
            output_tokens=response.tokens_used // 2,
            cached=response.cached
        )
        
        # Parse response
        try:
            import json
            analysis_data = json.loads(response.content)
        except json.JSONDecodeError:
            analysis_data = {"analysis": response.content, "parsed": False}
        
        return {
            "success": True,
            "analysis": analysis_data,
            "metadata": {
                "request_id": response.request_id,
                "model_used": response.model_used,
                "processing_time": response.processing_time,
                "confidence_score": response.confidence_score,
                "file_size": len(file_content),
                "file_type": file.content_type
            },
            "usage": await cost_optimizer.get_usage_analytics(user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Image analysis failed", "message": str(e)}
        )


@router.post("/generate-meal-plan", response_model=Dict[str, Any])
async def generate_meal_plan(
    request: MealPlanRequest,
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Generate comprehensive meal plans with advanced nutritional optimization
    Uses complex reasoning for family coordination and dietary restrictions
    """
    try:
        # Check if user has premium features
        # TODO: Implement proper user tier checking
        if request.days > 7:
            # Require premium for extended meal plans
            pass
        
        # Build AI request
        context = {
            "days": request.days,
            "family_size": request.family_size,
            "budget_per_week": request.budget_per_week,
            "cooking_time_available": request.cooking_time_available,
            "health_goals": request.health_goals
        }
        
        ai_request = AIRequest(
            request_type=AIRequestType.MEAL_PLANNING,
            user_id=user_id,
            prompt=f"Create a {request.days}-day meal plan",
            context=context,
            dietary_restrictions=request.dietary_restrictions,
            preferences=request.preferences,
            model_type=AIModelType.GEMINI_2_FLASH_THINKING
        )
        
        # Generate meal plan
        logger.info(f"Generating {request.days}-day meal plan for user {user_id}")
        response = await ai_service.generate_meal_plan(ai_request)
        
        # Track usage
        await cost_optimizer.track_usage(
            user_id=user_id,
            model=response.model_used,
            input_tokens=response.tokens_used // 2,
            output_tokens=response.tokens_used // 2,
            cached=response.cached
        )
        
        # Parse response
        try:
            import json
            meal_plan_data = json.loads(response.content)
        except json.JSONDecodeError:
            meal_plan_data = {"meal_plan": response.content, "parsed": False}
        
        return {
            "success": True,
            "meal_plan": meal_plan_data,
            "metadata": {
                "request_id": response.request_id,
                "model_used": response.model_used,
                "processing_time": response.processing_time,
                "confidence_score": response.confidence_score,
                "cached": response.cached,
                "days": request.days
            },
            "usage": await cost_optimizer.get_usage_analytics(user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meal plan generation failed: {str(e)}", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Meal plan generation failed", "message": str(e)}
        )


@router.post("/cooking-guidance")
async def stream_cooking_guidance(
    request: CookingGuidanceRequest,
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Stream real-time cooking guidance with step-by-step instructions
    Perfect for interactive cooking sessions
    """
    try:
        # Build context
        context = {
            "recipe": request.recipe_data,
            "current_step": request.current_step
        }
        
        ai_request = AIRequest(
            request_type=AIRequestType.COOKING_GUIDANCE,
            user_id=user_id,
            prompt=request.question,
            context=context,
            model_type=AIModelType.GEMINI_2_FLASH_THINKING
        )
        
        # Stream guidance
        async def guidance_generator():
            async for chunk in ai_service.stream_cooking_guidance(ai_request):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            guidance_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Cooking guidance failed: {str(e)}", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Cooking guidance failed", "message": str(e)}
        )


@router.get("/usage-analytics")
async def get_ai_usage_analytics(
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Get user's AI usage analytics and cost optimization report
    """
    try:
        usage = await cost_optimizer.get_usage_analytics(user_id)
        cost_prediction = await cost_optimizer.predict_monthly_cost(user_id)
        savings_report = await cost_optimizer.get_cost_savings_report(user_id)
        
        return {
            "usage": usage,
            "cost_prediction": cost_prediction,
            "savings_report": savings_report,
            "optimization_tips": [
                "Use specific prompts to get better results with fewer retries",
                "Cache frequently used recipes and meal plans",
                "Consider upgrading to Premium for unlimited AI access"
            ]
        }
        
    except Exception as e:
        logger.error(f"Usage analytics failed: {str(e)}", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get usage analytics", "message": str(e)}
        )


@router.post("/generate-image")
async def generate_image(
    title: str,
    description: Optional[str] = None,
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Generate high-quality food images using DALL-E 3 or Vertex AI Imagen
    Similar to how regular recipes are generated
    """
    try:
        import os
        import httpx
        import uuid
        from datetime import datetime
        
        logger.info(f"Generating image for: {title}")
        
        # Try OpenAI DALL-E 3 first if API key is configured
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if OPENAI_API_KEY:
            try:
                # Create optimized prompt for DALL-E 3
                prompt = (f"Professional food photography of {title}. "
                         f"{description if description else ''} "
                         f"High-quality, appetizing, restaurant presentation, "
                         f"perfect lighting, award-winning culinary photography, "
                         f"ultra-realistic, 8k resolution, food magazine style.")
                
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "dall-e-3",
                    "prompt": prompt[:1000],  # DALL-E 3 has a 1000 char limit
                    "size": "1024x1024",
                    "quality": "standard",
                    "n": 1
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("data") and len(result["data"]) > 0:
                            image_url = result["data"][0]["url"]
                            logger.info(f"✨ DALL-E 3 image generated successfully for: {title}")
                            
                            return {
                                "success": True,
                                "image_url": image_url,
                                "generator": "dall-e-3",
                                "metadata": {
                                    "title": title,
                                    "description": description,
                                    "generated_at": datetime.utcnow().isoformat(),
                                    "user_id": user_id
                                }
                            }
            except Exception as e:
                logger.warning(f"DALL-E 3 generation failed, trying Vertex AI: {e}")
        
        # Fallback to Vertex AI Imagen
        try:
            from ai.vertex_integration import VertexAIService
            vertex_service = VertexAIService()
            
            image_url = await vertex_service.generate_recipe_image(
                recipe_title=title,
                description=description or ""
            )
            
            if image_url:
                logger.info(f"✨ Vertex AI Imagen generated successfully for: {title}")
                return {
                    "success": True,
                    "image_url": image_url,
                    "generator": "vertex-ai-imagen",
                    "metadata": {
                        "title": title,
                        "description": description,
                        "generated_at": datetime.utcnow().isoformat(),
                        "user_id": user_id
                    }
                }
        except Exception as e:
            logger.warning(f"Vertex AI Imagen failed: {e}")
        
        # Final fallback to Unsplash
        import urllib.parse
        clean_title = urllib.parse.quote(title.replace(' ', '+'))
        fallback_url = f"https://source.unsplash.com/800x600/?{clean_title},food,meal"
        
        logger.info(f"Using Unsplash fallback for: {title}")
        return {
            "success": True,
            "image_url": fallback_url,
            "generator": "unsplash-fallback",
            "metadata": {
                "title": title,
                "description": description,
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
        }
        
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Image generation failed", "message": str(e)}
        )


@router.post("/validate-recipe-safety")
async def validate_recipe_safety(
    recipe_content: str,
    dietary_restrictions: Optional[List[str]] = None,
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Validate recipe safety and dietary compliance
    """
    try:
        # Check recipe safety
        safety_result = await safety_filters.check_recipe_safety(recipe_content)
        
        # Check dietary compliance if restrictions provided
        dietary_result = None
        if dietary_restrictions:
            dietary_result = await safety_filters.check_dietary_compliance(
                recipe_content, dietary_restrictions
            )
        
        return {
            "safety_check": {
                "is_safe": safety_result.is_safe,
                "level": safety_result.level.value,
                "reason": safety_result.reason,
                "violations": safety_result.violations,
                "confidence": safety_result.confidence
            },
            "dietary_compliance": {
                "is_compliant": dietary_result.is_safe if dietary_result else True,
                "level": dietary_result.level.value if dietary_result else "safe",
                "violations": dietary_result.violations if dietary_result else []
            } if dietary_result else None
        }
        
    except Exception as e:
        logger.error(f"Recipe safety validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Recipe safety validation failed", "message": str(e)}
        )


@router.post("/generate-meal-plan-name", response_model=Dict[str, Any])
async def generate_meal_plan_name(
    request: MealPlanNameRequest,
    user_id: str = "demo_user",  # TODO: Get from JWT token
    db=Depends(get_db)
):
    """
    Generate personalized, exciting meal plan names based on user preferences
    Uses AI to create unique names that reflect dietary goals and preferences
    """
    try:
        # Build AI request for name generation
        context = {
            "duration_days": request.duration_days,
            "family_size": request.family_size,
            "goals": request.goals,
        }
        
        preferences = {
            "cuisine_preferences": request.cuisine_preferences,
            "skill_level": request.skill_level
        }
        
        ai_request = AIRequest(
            request_type=AIRequestType.NAME_GENERATION,
            user_id=user_id,
            prompt="Generate a personalized meal plan name",
            context=context,
            dietary_restrictions=request.dietary_restrictions,
            preferences=preferences,
            model_type=AIModelType.GEMINI_1_5_FLASH  # Use fast model for quick response
        )
        
        # Generate name
        logger.info(f"Generating meal plan name for user {user_id}")
        response = await ai_service.generate_meal_plan_name(ai_request)
        
        # Track usage (very low cost for name generation)
        await cost_optimizer.track_usage(
            user_id=user_id,
            model=response.model_used,
            input_tokens=response.tokens_used // 4,  # Minimal tokens for name generation
            output_tokens=response.tokens_used // 4,
            cached=response.cached
        )
        
        return {
            "success": True,
            "name": response.content,
            "processing_time": response.processing_time,
            "model_used": response.model_used,
            "fallback_used": response.model_used == "fallback"
        }
        
    except Exception as e:
        logger.error(f"Meal plan name generation failed: {str(e)}")
        # Return a simple fallback name
        fallback_name = f"My {request.duration_days}-Day Journey"
        return {
            "success": True,
            "name": fallback_name,
            "processing_time": 0.1,
            "model_used": "fallback",
            "fallback_used": True
        }


@router.get("/health")
async def ai_service_health():
    """Check AI service health and model availability"""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "models_available": [
                "gemini-2.0-flash-thinking",
                "gemini-1.5-pro", 
                "gemini-1.5-flash"
            ],
            "features": {
                "recipe_generation": True,
                "image_analysis": True,
                "meal_planning": True,
                "cooking_guidance": True,
                "safety_validation": True
            },
            "cost_optimization": True,
            "cache_enabled": True
        }
        
    except Exception as e:
        logger.error(f"AI service health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "AI service unhealthy", "message": str(e)}
        )