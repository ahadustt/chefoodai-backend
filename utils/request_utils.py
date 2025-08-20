"""
ChefoodAI Request Utilities
Helper functions for extracting request information
"""

from fastapi import Request
from typing import Optional, Dict, Any
import re
import json
from user_agents import parse

def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request headers
    
    Handles various proxy configurations and cloud load balancers
    """
    # Check common proxy headers in order of preference
    headers_to_check = [
        "cf-connecting-ip",  # Cloudflare
        "x-forwarded-for",   # Standard proxy header
        "x-real-ip",         # Nginx proxy
        "x-client-ip",       # Apache mod_proxy
        "x-cluster-client-ip",  # Cluster/load balancer
        "forwarded-for",     # Less common
        "forwarded",         # RFC 7239
    ]
    
    for header in headers_to_check:
        ip = request.headers.get(header)
        if ip:
            # X-Forwarded-For can contain multiple IPs, take the first (original client)
            if "," in ip:
                ip = ip.split(",")[0].strip()
            
            # Basic IP validation
            if _is_valid_ip(ip):
                return ip
    
    # Fallback to direct connection IP
    if hasattr(request.client, 'host') and request.client.host:
        return request.client.host
    
    return "127.0.0.1"  # Default fallback

def get_user_agent(request: Request) -> str:
    """Extract user agent from request headers"""
    return request.headers.get("user-agent", "Unknown")

def parse_user_agent(user_agent: str) -> Dict[str, Any]:
    """
    Parse user agent string into structured information
    
    Returns browser, OS, device information
    """
    try:
        parsed = parse(user_agent)
        
        return {
            "browser": {
                "family": parsed.browser.family,
                "version": parsed.browser.version_string
            },
            "os": {
                "family": parsed.os.family,
                "version": parsed.os.version_string
            },
            "device": {
                "family": parsed.device.family,
                "brand": parsed.device.brand,
                "model": parsed.device.model
            },
            "is_mobile": parsed.is_mobile,
            "is_tablet": parsed.is_tablet,
            "is_touch_capable": parsed.is_touch_capable,
            "is_pc": parsed.is_pc,
            "is_bot": parsed.is_bot,
            "raw": user_agent
        }
    except Exception as e:
        return {
            "browser": {"family": "Unknown", "version": ""},
            "os": {"family": "Unknown", "version": ""},
            "device": {"family": "Unknown", "brand": "", "model": ""},
            "is_mobile": False,
            "is_tablet": False,
            "is_touch_capable": False,
            "is_pc": True,
            "is_bot": False,
            "raw": user_agent,
            "parse_error": str(e)
        }

def get_request_fingerprint(request: Request) -> str:
    """
    Generate a fingerprint for the request based on headers and client info
    
    Useful for detecting duplicate requests or tracking sessions
    """
    import hashlib
    
    # Collect relevant headers
    fingerprint_data = {
        "ip": get_client_ip(request),
        "user_agent": get_user_agent(request),
        "accept": request.headers.get("accept", ""),
        "accept_language": request.headers.get("accept-language", ""),
        "accept_encoding": request.headers.get("accept-encoding", ""),
        "host": request.headers.get("host", ""),
    }
    
    # Create hash
    fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]

def is_secure_request(request: Request) -> bool:
    """Check if request is made over HTTPS"""
    # Check the request scheme
    if request.url.scheme == "https":
        return True
    
    # Check proxy headers for original scheme
    forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
    if forwarded_proto == "https":
        return True
    
    forwarded = request.headers.get("forwarded", "").lower()
    if "proto=https" in forwarded:
        return True
    
    return False

def get_request_location(request: Request) -> Dict[str, Optional[str]]:
    """
    Extract location information from request headers
    
    Returns country, region, city if available from proxy/CDN headers
    """
    location = {
        "country": None,
        "region": None,
        "city": None,
        "timezone": None,
        "continent": None
    }
    
    # Cloudflare headers
    location["country"] = request.headers.get("cf-ipcountry")
    location["region"] = request.headers.get("cf-region")
    location["city"] = request.headers.get("cf-ipcity")
    location["timezone"] = request.headers.get("cf-timezone")
    
    # AWS CloudFront headers
    if not location["country"]:
        location["country"] = request.headers.get("cloudfront-viewer-country")
    
    # Google Cloud Load Balancer
    if not location["country"]:
        location["country"] = request.headers.get("x-country-code")
    
    # Other common headers
    if not location["country"]:
        location["country"] = request.headers.get("x-geoip-country")
    
    return location

def get_request_metadata(request: Request) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from request
    
    Returns all available request information
    """
    metadata = {
        "timestamp": request.state.timestamp if hasattr(request.state, 'timestamp') else None,
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "ip_address": get_client_ip(request),
        "user_agent_raw": get_user_agent(request),
        "user_agent_parsed": parse_user_agent(get_user_agent(request)),
        "fingerprint": get_request_fingerprint(request),
        "is_secure": is_secure_request(request),
        "location": get_request_location(request),
        "headers": dict(request.headers),
        "content_type": request.headers.get("content-type"),
        "content_length": request.headers.get("content-length"),
        "referer": request.headers.get("referer"),
        "origin": request.headers.get("origin"),
    }
    
    return metadata

def is_bot_request(request: Request) -> bool:
    """
    Detect if request is from a bot/crawler
    
    Uses user agent analysis and common bot patterns
    """
    user_agent = get_user_agent(request).lower()
    
    # Common bot patterns
    bot_patterns = [
        r'bot', r'crawler', r'spider', r'scraper',
        r'curl', r'wget', r'python-requests',
        r'googlebot', r'bingbot', r'slurp',
        r'facebookexternalhit', r'twitterbot',
        r'linkedinbot', r'whatsapp', r'telegram',
        r'discord', r'postman', r'insomnia'
    ]
    
    for pattern in bot_patterns:
        if re.search(pattern, user_agent):
            return True
    
    # Check parsed user agent
    parsed = parse_user_agent(get_user_agent(request))
    if parsed.get("is_bot", False):
        return True
    
    # Check for missing common headers
    required_headers = ["accept", "accept-language"]
    missing_headers = sum(1 for header in required_headers if not request.headers.get(header))
    
    if missing_headers >= len(required_headers) // 2:
        return True
    
    return False

def is_suspicious_request(request: Request) -> Dict[str, Any]:
    """
    Analyze request for suspicious patterns
    
    Returns suspicion score and reasons
    """
    suspicion_score = 0
    reasons = []
    
    # Check if it's a bot
    if is_bot_request(request):
        suspicion_score += 30
        reasons.append("Bot/crawler detected")
    
    # Check for missing user agent
    user_agent = get_user_agent(request)
    if not user_agent or user_agent == "Unknown":
        suspicion_score += 20
        reasons.append("Missing or unknown user agent")
    
    # Check for suspicious user agents
    suspicious_ua_patterns = [
        r'curl', r'wget', r'python', r'script',
        r'hack', r'scan', r'exploit', r'test'
    ]
    
    for pattern in suspicious_ua_patterns:
        if re.search(pattern, user_agent.lower()):
            suspicion_score += 25
            reasons.append(f"Suspicious user agent pattern: {pattern}")
            break
    
    # Check for unusual request patterns
    path = request.url.path.lower()
    suspicious_paths = [
        r'\.env', r'config', r'admin', r'phpmyadmin',
        r'wp-admin', r'\.git', r'backup', r'sql'
    ]
    
    for pattern in suspicious_paths:
        if re.search(pattern, path):
            suspicion_score += 40
            reasons.append(f"Suspicious path pattern: {pattern}")
            break
    
    # Check for rate limiting indicators
    headers = request.headers
    if headers.get("x-rate-limit-remaining") == "0":
        suspicion_score += 15
        reasons.append("Rate limit exhausted")
    
    # Check for proxy chains (multiple forwarded headers)
    forwarded_headers = [
        "x-forwarded-for", "x-real-ip", "x-client-ip",
        "cf-connecting-ip", "x-cluster-client-ip"
    ]
    
    forwarded_count = sum(1 for header in forwarded_headers if headers.get(header))
    if forwarded_count > 2:
        suspicion_score += 10
        reasons.append("Multiple proxy headers detected")
    
    return {
        "suspicion_score": suspicion_score,
        "is_suspicious": suspicion_score >= 50,
        "risk_level": _get_risk_level(suspicion_score),
        "reasons": reasons
    }

def extract_request_context(request: Request) -> Dict[str, Any]:
    """
    Extract all relevant context from request for logging/monitoring
    """
    return {
        "request_id": getattr(request.state, 'request_id', None),
        "method": request.method,
        "url": str(request.url),
        "ip": get_client_ip(request),
        "user_agent": parse_user_agent(get_user_agent(request)),
        "location": get_request_location(request),
        "fingerprint": get_request_fingerprint(request),
        "is_secure": is_secure_request(request),
        "is_bot": is_bot_request(request),
        "suspicion": is_suspicious_request(request),
        "timestamp": getattr(request.state, 'start_time', None)
    }

def _is_valid_ip(ip: str) -> bool:
    """Validate IP address format"""
    import ipaddress
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def _get_risk_level(score: int) -> str:
    """Convert suspicion score to risk level"""
    if score >= 80:
        return "critical"
    elif score >= 60:
        return "high"
    elif score >= 40:
        return "medium"
    elif score >= 20:
        return "low"
    else:
        return "minimal"