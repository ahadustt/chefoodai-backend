"""
ChefoodAI Security Utilities
Security helpers for password validation, token generation, and protection
"""

import secrets
import string
import re
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurityUtils:
    def __init__(self):
        self.password_min_length = 8
        self.password_max_length = 128
        self.token_length = 32
        
        # Password strength patterns
        self.password_patterns = {
            'lowercase': re.compile(r'[a-z]'),
            'uppercase': re.compile(r'[A-Z]'),
            'digit': re.compile(r'\d'),
            'special': re.compile(r'[!@#$%^&*(),.?":{}|<>]'),
            'whitespace': re.compile(r'\s'),
            'common_patterns': [
                re.compile(r'(.)\1{2,}'),  # Repeated characters
                re.compile(r'123456|654321|abcdef|qwerty', re.IGNORECASE),  # Common sequences
                re.compile(r'password|admin|user|login', re.IGNORECASE),  # Common words
            ]
        }
        
        # Common weak passwords
        self.weak_passwords = {
            'password', 'password123', '123456', '123456789', 'qwerty',
            'abc123', 'password1', 'admin', 'letmein', 'welcome',
            'monkey', '1234567890', 'dragon', 'master', 'hello',
            'freedom', 'whatever', 'qazwsx', 'trustno1', 'batman'
        }
    
    def validate_password_strength(
        self, 
        password: str, 
        user_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive password strength validation
        
        Args:
            password: Password to validate
            user_info: Optional user information to check against
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': True,
            'score': 0,
            'strength': 'weak',
            'errors': [],
            'suggestions': [],
            'requirements_met': {}
        }
        
        # Basic length check
        if len(password) < self.password_min_length:
            result['errors'].append(f'Password must be at least {self.password_min_length} characters long')
            result['is_valid'] = False
        
        if len(password) > self.password_max_length:
            result['errors'].append(f'Password must be no more than {self.password_max_length} characters long')
            result['is_valid'] = False
        
        # Character type requirements
        requirements = {
            'has_lowercase': bool(self.password_patterns['lowercase'].search(password)),
            'has_uppercase': bool(self.password_patterns['uppercase'].search(password)),
            'has_digit': bool(self.password_patterns['digit'].search(password)),
            'has_special': bool(self.password_patterns['special'].search(password)),
            'no_whitespace': not bool(self.password_patterns['whitespace'].search(password))
        }
        
        result['requirements_met'] = requirements
        
        # Score calculation
        base_score = min(len(password) * 2, 20)  # Length bonus (max 20)
        
        for req, met in requirements.items():
            if met:
                if req == 'has_lowercase':
                    base_score += 5
                elif req == 'has_uppercase':
                    base_score += 5
                elif req == 'has_digit':
                    base_score += 10
                elif req == 'has_special':
                    base_score += 15
                elif req == 'no_whitespace':
                    base_score += 5
        
        # Penalty for common patterns
        for pattern in self.password_patterns['common_patterns']:
            if pattern.search(password):
                base_score -= 10
                result['suggestions'].append('Avoid repeated characters or common sequences')
                break
        
        # Check against weak passwords
        if password.lower() in self.weak_passwords:
            base_score -= 20
            result['errors'].append('This is a commonly used password')
            result['is_valid'] = False
        
        # Check against user information
        if user_info:
            user_data = ' '.join(str(v).lower() for v in user_info.values() if v)
            if any(info.lower() in password.lower() for info in user_info.values() if info and len(info) > 2):
                base_score -= 15
                result['suggestions'].append('Avoid using personal information in password')
        
        # Minimum requirements check
        required_types = ['has_lowercase', 'has_uppercase', 'has_digit']
        if not all(requirements[req] for req in required_types):
            result['errors'].append('Password must contain uppercase, lowercase, and number')
            result['is_valid'] = False
        
        # Finalize score and strength
        result['score'] = max(0, min(100, base_score))
        
        if result['score'] >= 80:
            result['strength'] = 'very_strong'
        elif result['score'] >= 60:
            result['strength'] = 'strong'
        elif result['score'] >= 40:
            result['strength'] = 'medium'
        else:
            result['strength'] = 'weak'
        
        # Add suggestions based on missing requirements
        if not requirements['has_special']:
            result['suggestions'].append('Add special characters (!@#$%^&*) for stronger security')
        
        if len(password) < 12:
            result['suggestions'].append('Consider using a longer password (12+ characters)')
        
        return result
    
    def generate_secure_token(self, length: int = None) -> str:
        """Generate cryptographically secure random token"""
        if length is None:
            length = self.token_length
        
        return secrets.token_urlsafe(length)
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a secure random password"""
        if length < 8:
            length = 8
        
        # Ensure we have at least one of each required character type
        password_chars = []
        
        # Add required character types
        password_chars.append(secrets.choice(string.ascii_lowercase))
        password_chars.append(secrets.choice(string.ascii_uppercase))
        password_chars.append(secrets.choice(string.digits))
        password_chars.append(secrets.choice('!@#$%^&*(),.?":{}|<>'))
        
        # Fill remaining length with random characters
        all_chars = string.ascii_letters + string.digits + '!@#$%^&*(),.?":{}|<>'
        for _ in range(length - 4):
            password_chars.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password_chars)
        
        return ''.join(password_chars)
    
    def hash_password(self, password: str, rounds: int = 12) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against bcrypt hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except (ValueError, TypeError):
            return False
    
    def generate_otp(self, length: int = 6) -> str:
        """Generate numeric OTP"""
        return ''.join(secrets.choice(string.digits) for _ in range(length))
    
    def generate_backup_codes(self, count: int = 10, length: int = 8) -> List[str]:
        """Generate backup codes for 2FA"""
        codes = []
        for _ in range(count):
            # Generate alphanumeric codes
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) 
                          for _ in range(length))
            # Add dash in middle for readability
            formatted_code = f"{code[:4]}-{code[4:]}"
            codes.append(formatted_code)
        
        return codes
    
    def create_hmac_signature(
        self, 
        message: str, 
        secret: str, 
        algorithm: str = 'sha256'
    ) -> str:
        """Create HMAC signature"""
        hash_algo = getattr(hashes, algorithm.upper())() if hasattr(hashes, algorithm.upper()) else hashes.SHA256()
        
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            getattr(hashlib, algorithm)
        ).hexdigest()
        
        return signature
    
    def verify_hmac_signature(
        self, 
        message: str, 
        signature: str, 
        secret: str, 
        algorithm: str = 'sha256'
    ) -> bool:
        """Verify HMAC signature"""
        expected_signature = self.create_hmac_signature(message, secret, algorithm)
        return hmac.compare_digest(signature, expected_signature)
    
    def encrypt_data(self, data: str, key: str) -> str:
        """Encrypt data using Fernet (AES 128)"""
        # Derive key from password
        password = key.encode()
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Encrypt data
        f = Fernet(derived_key)
        encrypted_data = f.encrypt(data.encode())
        
        # Combine salt and encrypted data
        return base64.urlsafe_b64encode(salt + encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str, key: str) -> str:
        """Decrypt data using Fernet"""
        try:
            # Decode and split salt and data
            combined = base64.urlsafe_b64decode(encrypted_data.encode())
            salt = combined[:16]
            encrypted_bytes = combined[16:]
            
            # Derive key
            password = key.encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Decrypt
            f = Fernet(derived_key)
            decrypted_data = f.decrypt(encrypted_bytes)
            
            return decrypted_data.decode()
        except Exception:
            raise ValueError("Failed to decrypt data")
    
    def sanitize_input(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if not input_str:
            return ""
        
        # Truncate if too long
        sanitized = input_str[:max_length]
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\r']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Strip whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    def is_safe_url(self, url: str, allowed_hosts: Optional[List[str]] = None) -> bool:
        """Check if URL is safe for redirects"""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            
            # Must be relative or from allowed hosts
            if not parsed.netloc:  # Relative URL
                return True
            
            if allowed_hosts and parsed.netloc in allowed_hosts:
                return True
            
            return False
            
        except Exception:
            return False
    
    def generate_csrf_token(self, session_id: str, secret: str) -> str:
        """Generate CSRF token"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        message = f"{session_id}:{timestamp}"
        signature = self.create_hmac_signature(message, secret)
        
        token_data = f"{timestamp}:{signature}"
        return base64.urlsafe_b64encode(token_data.encode()).decode()
    
    def verify_csrf_token(
        self, 
        token: str, 
        session_id: str, 
        secret: str, 
        max_age: int = 3600
    ) -> bool:
        """Verify CSRF token"""
        try:
            # Decode token
            token_data = base64.urlsafe_b64decode(token.encode()).decode()
            timestamp_str, signature = token_data.split(':', 1)
            
            # Check age
            timestamp = int(timestamp_str)
            if datetime.utcnow().timestamp() - timestamp > max_age:
                return False
            
            # Verify signature
            message = f"{session_id}:{timestamp_str}"
            return self.verify_hmac_signature(message, signature, secret)
            
        except Exception:
            return False
    
    def check_password_breach(self, password: str) -> bool:
        """
        Check if password has been breached using k-anonymity
        (This would integrate with HaveIBeenPwned API in production)
        """
        # Generate SHA-1 hash
        sha1_hash = hashlib.sha1(password.encode()).hexdigest().upper()
        hash_prefix = sha1_hash[:5]
        hash_suffix = sha1_hash[5:]
        
        # In production, this would query the HaveIBeenPwned API
        # For now, return False (not breached) as a placeholder
        return False
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), location=()'
        }
    
    def validate_email_security(self, email: str) -> Dict[str, Any]:
        """Validate email for security concerns"""
        result = {
            'is_valid': True,
            'is_disposable': False,
            'is_suspicious': False,
            'warnings': []
        }
        
        # Basic email format validation
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email_pattern.match(email):
            result['is_valid'] = False
            result['warnings'].append('Invalid email format')
            return result
        
        domain = email.split('@')[1].lower()
        
        # Check for common disposable email domains
        disposable_domains = {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'yopmail.com', 'temp-mail.org'
        }
        
        if domain in disposable_domains:
            result['is_disposable'] = True
            result['warnings'].append('Disposable email detected')
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\d{10,}',  # Long sequences of numbers
            r'[a-z]{20,}',  # Very long random strings
            r'test|spam|fake|dummy',  # Suspicious keywords
        ]
        
        local_part = email.split('@')[0].lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, local_part):
                result['is_suspicious'] = True
                result['warnings'].append('Suspicious email pattern detected')
                break
        
        return result

# Create singleton instance
security_utils = SecurityUtils()