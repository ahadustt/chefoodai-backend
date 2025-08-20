"""
FastAPI routes for AI-powered recipe and meal planning features
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.responses import StreamingResponse
import json
import asyncio
from datetime import datetime
import uuid

from ..ai.vertex_integration import (
    VertexAIService,
    RecipeRequest,
    MealPlanRequest,
    ModelVersion,
    InferenceMode
)
from ..models import User
from ..dependencies import get_current_user, get_vertex_ai_service
from ..cache import cache_manager
from ..metrics import track_ai_usage

router = APIRouter(prefix="/api/v1/ai", tags=["AI"])

# A/B test configuration
ACTIVE_EXPERIMENTS = {
    "recipe_prompt": {
        "variants": ["control", "detailed_context", "conversational", "minimal"],
        "traffic_split": [0.25, 0.25, 0.25, 0.25]
    },
    "model_selection": {
        "variants": [ModelVersion.GEMINI_1_5_PRO, ModelVersion.GEMINI_1_5_FLASH],
        "traffic_split": [0.7, 0.3]
    }
}

def get_experiment_variant(experiment_name: str, user_id: str) -> str:
    """Determine experiment variant for user"""
    # Simple hash-based assignment for consistent user experience
    hash_value = hash(f"{experiment_name}:{user_id}") % 100
    
    experiment = ACTIVE_EXPERIMENTS.get(experiment_name)
    if not experiment:
        return "control"
    
    cumulative = 0
    for i, split in enumerate(experiment["traffic_split"]):
        cumulative += split * 100
        if hash_value < cumulative:
            return experiment["variants"][i]
    
    return experiment["variants"][0]

@router.post("/recipes/generate", response_model=Dict[str, Any])
async def generate_recipe(
    request: RecipeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    ai_service: VertexAIService = Depends(get_vertex_ai_service)
):
    """
    Generate a recipe based on available ingredients and preferences
    
    Features:
    - Smart caching for repeated requests
    - A/B testing for prompt optimization
    - Fallback to faster model on failure
    - Usage tracking for billing
    """
    
    try:
        # Determine experiment variants
        prompt_variant = get_experiment_variant("recipe_prompt", str(current_user.id))
        model_variant = get_experiment_variant("model_selection", str(current_user.id))
        
        # Generate recipe with selected variants
        recipe = await ai_service.generate_recipe(
            request=request,
            model_version=model_variant,
            use_cache=True,
            experiment_id=prompt_variant if prompt_variant != "control" else None
        )
        
        if not recipe:
            raise HTTPException(status_code=500, detail="Failed to generate recipe")
        
        # Track usage in background
        background_tasks.add_task(
            track_ai_usage,
            user_id=current_user.id,
            feature="recipe_generation",
            model=model_variant.value,
            tokens_used=len(json.dumps(recipe))  # Approximate
        )
        
        # Add metadata
        recipe["metadata"] = {
            "generated_at": datetime.utcnow().isoformat(),
            "model_version": model_variant.value,
            "experiment_variant": prompt_variant,
            "user_preferences": request.dietary_preferences
        }
        
        return recipe
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipes/generate-batch", response_model=List[Dict[str, Any]])
async def generate_recipes_batch(
    requests: List[RecipeRequest],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    ai_service: VertexAIService = Depends(get_vertex_ai_service)
):
    """
    Batch generate multiple recipes for efficiency
    
    Optimizations:
    - Processes up to 10 recipes in a single API call
    - Uses faster model for batch processing
    - Automatic fallback to individual generation on failure
    """
    
    if len(requests) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 recipes per batch request"
        )
    
    try:
        recipes = await ai_service.batch_generate_recipes(requests)
        
        # Track batch usage
        background_tasks.add_task(
            track_ai_usage,
            user_id=current_user.id,
            feature="batch_recipe_generation",
            model=ModelVersion.GEMINI_1_5_FLASH.value,
            tokens_used=len(json.dumps(recipes))
        )
        
        return recipes
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipes/stream", response_class=StreamingResponse)
async def generate_recipe_stream(
    request: RecipeRequest,
    current_user: User = Depends(get_current_user),
    ai_service: VertexAIService = Depends(get_vertex_ai_service)
):
    """
    Stream recipe generation for real-time UI updates
    
    Benefits:
    - Faster perceived response time
    - Progressive rendering in UI
    - Early cancellation support
    """
    
    async def recipe_stream():
        try:
            # For streaming, we'd use Vertex AI's streaming capabilities
            # This is a simplified example
            recipe_parts = [
                {"type": "title", "content": "Generating recipe..."},
                {"type": "ingredients", "content": []},
                {"type": "instructions", "content": []},
                {"type": "complete", "content": {}}
            ]
            
            for part in recipe_parts:
                yield f"data: {json.dumps(part)}\n\n"
                await asyncio.sleep(0.5)  # Simulate streaming delay
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        recipe_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/meal-plans/generate", response_model=Dict[str, Any])
async def generate_meal_plan(
    request: MealPlanRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    ai_service: VertexAIService = Depends(get_vertex_ai_service)
):
    """
    Generate a complete meal plan with shopping list
    
    Features:
    - Context window optimization for long outputs
    - Intelligent ingredient reuse
    - Budget and nutrition optimization
    - Batch cooking suggestions
    """
    
    try:
        # Check user's subscription for meal planning feature
        if not current_user.has_premium_features:
            raise HTTPException(
                status_code=403,
                detail="Meal planning requires premium subscription"
            )
        
        meal_plan = await ai_service.generate_meal_plan(request)
        
        if not meal_plan:
            raise HTTPException(status_code=500, detail="Failed to generate meal plan")
        
        # Track usage
        background_tasks.add_task(
            track_ai_usage,
            user_id=current_user.id,
            feature="meal_plan_generation",
            model=ModelVersion.GEMINI_1_5_PRO.value,
            tokens_used=len(json.dumps(meal_plan))
        )
        
        # Add user-specific customizations
        meal_plan["metadata"] = {
            "generated_at": datetime.utcnow().isoformat(),
            "user_id": str(current_user.id),
            "preferences": request.dietary_preferences,
            "plan_id": str(uuid.uuid4())
        }
        
        return meal_plan
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipes/analyze-image", response_model=Dict[str, Any])
async def analyze_recipe_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    ai_service: VertexAIService = Depends(get_vertex_ai_service)
):
    """
    Analyze uploaded food image using multi-modal AI
    
    Capabilities:
    - Dish identification
    - Ingredient detection
    - Cooking method analysis
    - Nutritional estimation
    - Recipe suggestions
    """
    
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WebP images are supported"
        )
    
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail="Image size must be less than 10MB"
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Analyze image
        analysis = await ai_service.analyze_recipe_image(image_data)
        
        if not analysis:
            raise HTTPException(status_code=500, detail="Failed to analyze image")
        
        # Track usage
        background_tasks.add_task(
            track_ai_usage,
            user_id=current_user.id,
            feature="image_analysis",
            model=ModelVersion.GEMINI_1_5_PRO.value,
            tokens_used=1000  # Approximate for image
        )
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/usage/stats", response_model=Dict[str, Any])
async def get_ai_usage_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get user's AI usage statistics for the current billing period
    
    Returns:
    - Total API calls
    - Tokens consumed
    - Feature breakdown
    - Remaining quota
    """
    
    # This would fetch from your usage tracking system
    usage_stats = {
        "period_start": "2025-01-01",
        "period_end": "2025-01-31",
        "total_requests": 156,
        "total_tokens": 45000,
        "features": {
            "recipe_generation": 120,
            "meal_planning": 20,
            "image_analysis": 16
        },
        "quota": {
            "requests_limit": 1000,
            "requests_remaining": 844,
            "tokens_limit": 100000,
            "tokens_remaining": 55000
        },
        "estimated_cost": 4.50
    }
    
    return usage_stats

@router.post("/feedback", response_model=Dict[str, str])
async def submit_ai_feedback(
    recipe_id: str,
    rating: int,
    feedback: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Submit feedback on AI-generated content for continuous improvement
    
    Used for:
    - Model fine-tuning data collection
    - A/B test evaluation
    - Quality monitoring
    """
    
    if rating < 1 or rating > 5:
        raise HTTPException(
            status_code=400,
            detail="Rating must be between 1 and 5"
        )
    
    # Store feedback for analysis and model improvement
    feedback_entry = {
        "recipe_id": recipe_id,
        "user_id": str(current_user.id),
        "rating": rating,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # In production, this would be stored in BigQuery or similar
    # for analysis and model fine-tuning
    
    return {"message": "Feedback received. Thank you for helping us improve!"}

@router.get("/experiments/active", response_model=Dict[str, Any])
async def get_active_experiments(
    current_user: User = Depends(get_current_user)
):
    """
    Get user's active A/B test assignments
    
    Useful for:
    - Debugging
    - User transparency
    - Experiment analysis
    """
    
    user_experiments = {}
    
    for experiment_name in ACTIVE_EXPERIMENTS:
        variant = get_experiment_variant(experiment_name, str(current_user.id))
        user_experiments[experiment_name] = variant
    
    return {
        "user_id": str(current_user.id),
        "experiments": user_experiments,
        "timestamp": datetime.utcnow().isoformat()
    }