"""
ChefoodAI Safety Filters
Advanced safety checking for AI-generated content
"""

import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class SafetyLevel(Enum):
    """Safety check levels"""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


@dataclass
class SafetyResult:
    """Safety check result"""
    is_safe: bool
    level: SafetyLevel
    reason: str
    violations: List[str]
    confidence: float


class SafetyFilters:
    """
    Comprehensive safety filters for AI-generated culinary content
    Ensures food safety, allergy awareness, and content appropriateness
    """
    
    def __init__(self):
        # Dangerous food combinations and practices
        self.dangerous_combinations = {
            "raw_meat_cross_contamination": [
                "raw chicken", "raw beef", "raw pork", "raw fish"
            ],
            "dangerous_temperatures": [
                "room temperature", "warm storage", "lukewarm"
            ],
            "toxic_ingredients": [
                "raw kidney beans", "raw elderberries", "green potatoes",
                "raw rhubarb leaves", "bitter almonds", "raw cassava"
            ]
        }
        
        # Food safety keywords that require careful handling
        self.safety_keywords = {
            "high_risk": [
                "raw eggs", "unpasteurized", "raw milk", "rare meat",
                "sushi grade", "tartare", "carpaccio"
            ],
            "allergen_warnings": [
                "nuts", "peanuts", "shellfish", "dairy", "eggs",
                "wheat", "soy", "sesame", "fish"
            ],
            "temperature_critical": [
                "poultry", "ground meat", "seafood", "eggs", "dairy"
            ]
        }
        
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            re.compile(r'\b(poison|toxic|deadly|kill|harm)\b', re.IGNORECASE),
            re.compile(r'\b(alcohol.*child|drunk|intoxicat)\b', re.IGNORECASE),
            re.compile(r'\b(illegal|drug|narcotic)\b', re.IGNORECASE)
        ]
        
        # Safe cooking temperature guidelines (°F)
        self.safe_temperatures = {
            "poultry": 165,
            "ground_meat": 160,
            "beef_steaks": 145,
            "pork": 145,
            "fish": 145,
            "eggs": 160,
            "leftovers": 165
        }
        
        # Allergen detection patterns
        self.allergen_patterns = {
            "nuts": re.compile(r'\b(almonds?|walnuts?|pecans?|cashews?|pistachios?|hazelnuts?|brazil nuts?|macadamia)\b', re.IGNORECASE),
            "peanuts": re.compile(r'\b(peanuts?|peanut butter|groundnuts?)\b', re.IGNORECASE),
            "dairy": re.compile(r'\b(milk|cheese|butter|cream|yogurt|whey|casein|lactose)\b', re.IGNORECASE),
            "eggs": re.compile(r'\b(eggs?|egg whites?|egg yolks?|mayonnaise|meringue)\b', re.IGNORECASE),
            "wheat": re.compile(r'\b(wheat|flour|bread|pasta|gluten|seitan)\b', re.IGNORECASE),
            "soy": re.compile(r'\b(soy|tofu|tempeh|miso|soy sauce|edamame)\b', re.IGNORECASE),
            "fish": re.compile(r'\b(fish|salmon|tuna|cod|halibut|anchovy|sardine)\b', re.IGNORECASE),
            "shellfish": re.compile(r'\b(shrimp|crab|lobster|clams?|mussels?|oysters?|scallops?)\b', re.IGNORECASE),
            "sesame": re.compile(r'\b(sesame|tahini)\b', re.IGNORECASE)
        }
    
    async def check_recipe_safety(self, recipe_content: str) -> SafetyResult:
        """
        Comprehensive safety check for recipe content
        """
        violations = []
        confidence = 1.0
        
        try:
            # Parse recipe if it's JSON
            if recipe_content.strip().startswith('{'):
                recipe_data = json.loads(recipe_content)
                content_to_check = json.dumps(recipe_data, indent=2)
            else:
                content_to_check = recipe_content
                recipe_data = {}
            
            # Check for inappropriate content
            inappropriate = await self._check_inappropriate_content(content_to_check)
            if inappropriate:
                violations.extend(inappropriate)
            
            # Check for dangerous food practices
            dangerous_practices = await self._check_dangerous_practices(content_to_check, recipe_data)
            if dangerous_practices:
                violations.extend(dangerous_practices)
            
            # Check temperature safety
            temp_issues = await self._check_temperature_safety(content_to_check, recipe_data)
            if temp_issues:
                violations.extend(temp_issues)
            
            # Check for cross-contamination risks
            contamination_risks = await self._check_contamination_risks(content_to_check, recipe_data)
            if contamination_risks:
                violations.extend(contamination_risks)
            
            # Determine safety level
            if not violations:
                return SafetyResult(
                    is_safe=True,
                    level=SafetyLevel.SAFE,
                    reason="Recipe passed all safety checks",
                    violations=[],
                    confidence=confidence
                )
            
            # Categorize violations
            critical_violations = [v for v in violations if "CRITICAL" in v]
            warning_violations = [v for v in violations if "WARNING" in v]
            
            if critical_violations:
                return SafetyResult(
                    is_safe=False,
                    level=SafetyLevel.UNSAFE,
                    reason="Recipe contains critical safety violations",
                    violations=violations,
                    confidence=confidence
                )
            elif warning_violations:
                return SafetyResult(
                    is_safe=True,
                    level=SafetyLevel.WARNING,
                    reason="Recipe has safety warnings that should be addressed",
                    violations=violations,
                    confidence=confidence * 0.8
                )
            else:
                return SafetyResult(
                    is_safe=True,
                    level=SafetyLevel.SAFE,
                    reason="Recipe is safe with minor considerations",
                    violations=violations,
                    confidence=confidence * 0.9
                )
                
        except Exception as e:
            logger.error(f"Safety check failed: {str(e)}")
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.BLOCKED,
                reason=f"Safety check error: {str(e)}",
                violations=["CRITICAL: Safety validation failed"],
                confidence=0.0
            )
    
    async def check_dietary_compliance(
        self, 
        recipe_content: str, 
        dietary_restrictions: List[str]
    ) -> SafetyResult:
        """
        Check if recipe complies with dietary restrictions
        """
        violations = []
        
        if not dietary_restrictions:
            return SafetyResult(
                is_safe=True,
                level=SafetyLevel.SAFE,
                reason="No dietary restrictions to check",
                violations=[],
                confidence=1.0
            )
        
        try:
            content_lower = recipe_content.lower()
            
            for restriction in dietary_restrictions:
                restriction_lower = restriction.lower()
                
                # Check common dietary restrictions
                if "vegetarian" in restriction_lower:
                    meat_found = await self._check_for_meat(content_lower)
                    if meat_found:
                        violations.append(f"WARNING: Contains meat products ({', '.join(meat_found)}) - violates vegetarian diet")
                
                elif "vegan" in restriction_lower:
                    animal_products = await self._check_for_animal_products(content_lower)
                    if animal_products:
                        violations.append(f"WARNING: Contains animal products ({', '.join(animal_products)}) - violates vegan diet")
                
                elif "gluten" in restriction_lower:
                    gluten_sources = await self._check_for_gluten(content_lower)
                    if gluten_sources:
                        violations.append(f"WARNING: Contains gluten sources ({', '.join(gluten_sources)}) - violates gluten-free diet")
                
                elif "dairy" in restriction_lower:
                    dairy_products = await self._check_for_dairy(content_lower)
                    if dairy_products:
                        violations.append(f"WARNING: Contains dairy products ({', '.join(dairy_products)}) - violates dairy-free diet")
                
                # Check allergen compliance
                for allergen, pattern in self.allergen_patterns.items():
                    if allergen in restriction_lower:
                        matches = pattern.findall(content_lower)
                        if matches:
                            violations.append(f"CRITICAL: Contains allergen {allergen} ({', '.join(set(matches))}) - ALLERGY RISK")
            
            # Determine result
            critical_violations = [v for v in violations if "CRITICAL" in v]
            
            if critical_violations:
                return SafetyResult(
                    is_safe=False,
                    level=SafetyLevel.UNSAFE,
                    reason="Recipe violates critical dietary restrictions (allergy risk)",
                    violations=violations,
                    confidence=0.95
                )
            elif violations:
                return SafetyResult(
                    is_safe=True,
                    level=SafetyLevel.WARNING,
                    reason="Recipe may not comply with some dietary preferences",
                    violations=violations,
                    confidence=0.8
                )
            else:
                return SafetyResult(
                    is_safe=True,
                    level=SafetyLevel.SAFE,
                    reason="Recipe complies with all dietary restrictions",
                    violations=[],
                    confidence=0.9
                )
                
        except Exception as e:
            logger.error(f"Dietary compliance check failed: {str(e)}")
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.BLOCKED,
                reason=f"Dietary compliance check error: {str(e)}",
                violations=["CRITICAL: Dietary compliance validation failed"],
                confidence=0.0
            )
    
    async def validate_ingredient_safety(self, ingredients: List[str]) -> SafetyResult:
        """
        Validate safety of individual ingredients
        """
        violations = []
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            
            # Check for toxic ingredients
            for toxic in self.dangerous_combinations["toxic_ingredients"]:
                if toxic in ingredient_lower:
                    violations.append(f"CRITICAL: Potentially toxic ingredient detected: {ingredient}")
            
            # Check for high-risk ingredients
            for high_risk in self.safety_keywords["high_risk"]:
                if high_risk in ingredient_lower:
                    violations.append(f"WARNING: High-risk ingredient requires careful handling: {ingredient}")
            
            # Check for allergens
            allergens_found = []
            for allergen, pattern in self.allergen_patterns.items():
                if pattern.search(ingredient_lower):
                    allergens_found.append(allergen)
            
            if allergens_found:
                violations.append(f"INFO: Allergen warning for {ingredient}: contains {', '.join(allergens_found)}")
        
        # Determine safety level
        critical_violations = [v for v in violations if "CRITICAL" in v]
        warning_violations = [v for v in violations if "WARNING" in v]
        
        if critical_violations:
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.UNSAFE,
                reason="Unsafe ingredients detected",
                violations=violations,
                confidence=0.95
            )
        elif warning_violations:
            return SafetyResult(
                is_safe=True,
                level=SafetyLevel.WARNING,
                reason="Ingredients require careful handling",
                violations=violations,
                confidence=0.85
            )
        else:
            return SafetyResult(
                is_safe=True,
                level=SafetyLevel.SAFE,
                reason="All ingredients are safe",
                violations=violations,
                confidence=0.9
            )
    
    # Private helper methods
    
    async def _check_inappropriate_content(self, content: str) -> List[str]:
        """Check for inappropriate content patterns"""
        violations = []
        
        for pattern in self.inappropriate_patterns:
            matches = pattern.findall(content)
            if matches:
                violations.append(f"CRITICAL: Inappropriate content detected: {', '.join(set(matches))}")
        
        return violations
    
    async def _check_dangerous_practices(self, content: str, recipe_data: Dict) -> List[str]:
        """Check for dangerous food preparation practices"""
        violations = []
        content_lower = content.lower()
        
        # Check for dangerous temperature practices
        dangerous_temps = ["room temperature storage", "leave out", "keep warm"]
        for temp_practice in dangerous_temps:
            if temp_practice in content_lower:
                violations.append(f"WARNING: Potentially unsafe temperature practice: {temp_practice}")
        
        # Check for raw meat handling issues
        if any(meat in content_lower for meat in self.dangerous_combinations["raw_meat_cross_contamination"]):
            if "wash hands" not in content_lower and "clean" not in content_lower:
                violations.append("WARNING: Raw meat handling detected - ensure proper hygiene practices")
        
        return violations
    
    async def _check_temperature_safety(self, content: str, recipe_data: Dict) -> List[str]:
        """Check cooking temperature safety"""
        violations = []
        
        # Extract temperature information
        temp_pattern = re.compile(r'(\d+)°?\s*[fF]', re.IGNORECASE)
        temperatures = temp_pattern.findall(content)
        
        if temperatures:
            for temp_str in temperatures:
                temp = int(temp_str)
                
                # Check against safe temperatures
                content_lower = content.lower()
                for food_type, safe_temp in self.safe_temperatures.items():
                    if food_type.replace("_", " ") in content_lower and temp < safe_temp:
                        violations.append(f"WARNING: Temperature {temp}°F may be unsafe for {food_type} (recommended: {safe_temp}°F)")
        
        return violations
    
    async def _check_contamination_risks(self, content: str, recipe_data: Dict) -> List[str]:
        """Check for cross-contamination risks"""
        violations = []
        content_lower = content.lower()
        
        # Check for raw meat and fresh produce mixing
        has_raw_meat = any(meat in content_lower for meat in ["raw chicken", "raw beef", "raw pork"])
        has_fresh_produce = any(produce in content_lower for produce in ["lettuce", "tomato", "cucumber", "herbs"])
        
        if has_raw_meat and has_fresh_produce:
            if "separate" not in content_lower and "wash hands" not in content_lower:
                violations.append("WARNING: Raw meat and fresh produce - ensure proper separation and hygiene")
        
        return violations
    
    async def _check_for_meat(self, content: str) -> List[str]:
        """Check for meat products"""
        meat_products = [
            "beef", "pork", "chicken", "turkey", "lamb", "veal", 
            "bacon", "ham", "sausage", "fish", "salmon", "tuna"
        ]
        found = [meat for meat in meat_products if meat in content]
        return found
    
    async def _check_for_animal_products(self, content: str) -> List[str]:
        """Check for animal products (vegan check)"""
        animal_products = [
            "meat", "beef", "pork", "chicken", "fish", "milk", "cheese", 
            "butter", "eggs", "honey", "gelatin", "lard"
        ]
        found = [product for product in animal_products if product in content]
        return found
    
    async def _check_for_gluten(self, content: str) -> List[str]:
        """Check for gluten sources"""
        gluten_sources = [
            "wheat", "flour", "bread", "pasta", "barley", "rye", 
            "oats", "soy sauce", "beer"
        ]
        found = [source for source in gluten_sources if source in content]
        return found
    
    async def _check_for_dairy(self, content: str) -> List[str]:
        """Check for dairy products"""
        dairy_products = [
            "milk", "cheese", "butter", "cream", "yogurt", 
            "whey", "casein", "lactose"
        ]
        found = [dairy for dairy in dairy_products if dairy in content]
        return found


# Global safety filters instance
safety_filters = SafetyFilters()