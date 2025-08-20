"""
ChefoodAI Email Service
Comprehensive email service for authentication, notifications, and marketing
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List, Dict, Any, Union
import jinja2
from pathlib import Path
import aiosmtplib
from datetime import datetime
import logging

from core.config import get_settings
from models.users import User

settings = get_settings()
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.FROM_EMAIL
        self.from_name = settings.FROM_NAME or "ChefoodAI"
        
        # Initialize Jinja2 template environment
        template_dir = Path(__file__).parent.parent / "templates" / "emails"
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Email templates
        self.templates = {
            'welcome': 'welcome.html',
            'email_verification': 'email_verification.html',
            'password_reset': 'password_reset.html',
            'password_changed': 'password_changed.html',
            'login_notification': 'login_notification.html',
            'recipe_recommendation': 'recipe_recommendation.html',
            'meal_plan_reminder': 'meal_plan_reminder.html',
            'security_alert': 'security_alert.html',
            'subscription_update': 'subscription_update.html',
            'cooking_tips': 'cooking_tips.html'
        }
    
    async def send_email(
        self,
        to_email: Union[str, List[str]],
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        reply_to: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
    ) -> bool:
        """
        Send email using async SMTP
        
        Args:
            to_email: Recipient email(s)
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text content (optional)
            attachments: List of attachments
            reply_to: Reply-to address
            cc: CC recipients
            bcc: BCC recipients
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = f"{self.from_name} <{self.from_email}>"
            
            # Handle recipient emails
            if isinstance(to_email, str):
                message['To'] = to_email
                recipients = [to_email]
            else:
                message['To'] = ', '.join(to_email)
                recipients = to_email
            
            if reply_to:
                message['Reply-To'] = reply_to
            
            if cc:
                message['Cc'] = ', '.join(cc)
                recipients.extend(cc)
            
            # Add message ID and date
            message['Message-ID'] = aiosmtplib.email.message.make_msgid()
            message['Date'] = aiosmtplib.email.utils.formatdate(localtime=True)
            
            # Add text content
            if text_content:
                text_part = MIMEText(text_content, 'plain', 'utf-8')
                message.attach(text_part)
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html', 'utf-8')
            message.attach(html_part)
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    self._add_attachment(message, attachment)
            
            # Add BCC recipients
            if bcc:
                recipients.extend(bcc)
            
            # Send email
            await aiosmtplib.send(
                message,
                hostname=self.smtp_server,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                use_tls=True,
                recipients=recipients
            )
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
    
    def _add_attachment(self, message: MIMEMultipart, attachment: Dict[str, Any]):
        """Add attachment to email message"""
        try:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment['content'])
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {attachment["filename"]}'
            )
            message.attach(part)
        except Exception as e:
            logger.error(f"Failed to add attachment {attachment.get('filename', 'unknown')}: {e}")
    
    def _render_template(
        self, 
        template_name: str, 
        context: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Render email template
        
        Returns:
            Tuple of (html_content, text_content)
        """
        try:
            template = self.template_env.get_template(template_name)
            
            # Add common context variables
            context.update({
                'app_name': 'ChefoodAI',
                'current_year': datetime.now().year,
                'support_email': 'support@chefoodai.com',
                'website_url': settings.FRONTEND_URL,
                'unsubscribe_url': f"{settings.FRONTEND_URL}/unsubscribe"
            })
            
            html_content = template.render(**context)
            
            # Generate text content from HTML (simplified)
            text_content = self._html_to_text(html_content)
            
            return html_content, text_content
            
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            return "", ""
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text (simplified)"""
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        
        # Replace common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def send_welcome_email(self, user: User) -> bool:
        """Send welcome email to new user"""
        try:
            context = {
                'user_name': user.first_name,
                'user_email': user.email,
                'plan': user.plan.title(),
                'login_url': f"{settings.FRONTEND_URL}/login",
                'dashboard_url': f"{settings.FRONTEND_URL}/dashboard"
            }
            
            html_content, text_content = self._render_template(
                self.templates['welcome'], context
            )
            
            subject = f"Welcome to ChefoodAI, {user.first_name}! 🍽️"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send welcome email to {user.email}: {e}")
            return False
    
    async def send_email_verification(self, user: User, token: str = None) -> bool:
        """Send email verification"""
        try:
            # Generate verification token if not provided
            if not token:
                from utils.security import security_utils
                token = security_utils.generate_secure_token()
            
            verification_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
            
            context = {
                'user_name': user.first_name,
                'verification_url': verification_url,
                'token_expiry': '24 hours'
            }
            
            html_content, text_content = self._render_template(
                self.templates['email_verification'], context
            )
            
            subject = "Please verify your ChefoodAI email address"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send verification email to {user.email}: {e}")
            return False
    
    async def send_password_reset_email(self, user: User, reset_token: str) -> bool:
        """Send password reset email"""
        try:
            reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
            
            context = {
                'user_name': user.first_name,
                'reset_url': reset_url,
                'token_expiry': '1 hour',
                'support_email': 'support@chefoodai.com'
            }
            
            html_content, text_content = self._render_template(
                self.templates['password_reset'], context
            )
            
            subject = "Reset your ChefoodAI password"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send password reset email to {user.email}: {e}")
            return False
    
    async def send_password_change_notification(self, user: User) -> bool:
        """Send password change notification"""
        try:
            context = {
                'user_name': user.first_name,
                'change_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
                'security_url': f"{settings.FRONTEND_URL}/security",
                'support_email': 'support@chefoodai.com'
            }
            
            html_content, text_content = self._render_template(
                self.templates['password_changed'], context
            )
            
            subject = "Your ChefoodAI password has been changed"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send password change notification to {user.email}: {e}")
            return False
    
    async def send_login_notification(
        self, 
        user: User, 
        ip_address: str, 
        user_agent: str,
        location: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send login notification for new device/location"""
        try:
            from utils.request_utils import parse_user_agent
            
            parsed_ua = parse_user_agent(user_agent)
            device_info = f"{parsed_ua['browser']['family']} on {parsed_ua['os']['family']}"
            
            location_str = "Unknown location"
            if location and location.get('city') and location.get('country'):
                location_str = f"{location['city']}, {location['country']}"
            elif location and location.get('country'):
                location_str = location['country']
            
            context = {
                'user_name': user.first_name,
                'login_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
                'ip_address': ip_address,
                'device_info': device_info,
                'location': location_str,
                'security_url': f"{settings.FRONTEND_URL}/security",
                'support_email': 'support@chefoodai.com'
            }
            
            html_content, text_content = self._render_template(
                self.templates['login_notification'], context
            )
            
            subject = "New login to your ChefoodAI account"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send login notification to {user.email}: {e}")
            return False
    
    async def send_password_reset_confirmation(self, user: User) -> bool:
        """Send password reset confirmation"""
        try:
            context = {
                'user_name': user.first_name,
                'reset_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
                'login_url': f"{settings.FRONTEND_URL}/login",
                'support_email': 'support@chefoodai.com'
            }
            
            html_content, text_content = self._render_template(
                'password_reset_confirmation.html', context
            )
            
            subject = "Your ChefoodAI password has been reset"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send password reset confirmation to {user.email}: {e}")
            return False
    
    async def send_security_alert(
        self, 
        user: User, 
        alert_type: str, 
        details: Dict[str, Any]
    ) -> bool:
        """Send security alert"""
        try:
            context = {
                'user_name': user.first_name,
                'alert_type': alert_type,
                'alert_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
                'details': details,
                'security_url': f"{settings.FRONTEND_URL}/security",
                'support_email': 'support@chefoodai.com'
            }
            
            html_content, text_content = self._render_template(
                self.templates['security_alert'], context
            )
            
            subject = f"ChefoodAI Security Alert: {alert_type.replace('_', ' ').title()}"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send security alert to {user.email}: {e}")
            return False
    
    async def send_recipe_recommendation(
        self, 
        user: User, 
        recipes: List[Dict[str, Any]]
    ) -> bool:
        """Send personalized recipe recommendations"""
        try:
            if not user.notification_settings.get('recipe_recommendations', True):
                return True  # User opted out
            
            context = {
                'user_name': user.first_name,
                'recipes': recipes,
                'dashboard_url': f"{settings.FRONTEND_URL}/dashboard",
                'preferences_url': f"{settings.FRONTEND_URL}/preferences"
            }
            
            html_content, text_content = self._render_template(
                self.templates['recipe_recommendation'], context
            )
            
            subject = f"New recipe recommendations for you, {user.first_name}! 👨‍🍳"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send recipe recommendations to {user.email}: {e}")
            return False
    
    async def send_meal_plan_reminder(
        self, 
        user: User, 
        meal_plan: Dict[str, Any]
    ) -> bool:
        """Send meal plan reminder"""
        try:
            if not user.notification_settings.get('meal_plan_reminders', True):
                return True  # User opted out
            
            context = {
                'user_name': user.first_name,
                'meal_plan': meal_plan,
                'dashboard_url': f"{settings.FRONTEND_URL}/dashboard",
                'meal_plans_url': f"{settings.FRONTEND_URL}/meal-plans"
            }
            
            html_content, text_content = self._render_template(
                self.templates['meal_plan_reminder'], context
            )
            
            subject = "Don't forget about your meal plan today! 📅"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send meal plan reminder to {user.email}: {e}")
            return False
    
    async def send_subscription_update(
        self, 
        user: User, 
        old_plan: str, 
        new_plan: str
    ) -> bool:
        """Send subscription update notification"""
        try:
            context = {
                'user_name': user.first_name,
                'old_plan': old_plan.title(),
                'new_plan': new_plan.title(),
                'effective_date': datetime.utcnow().strftime('%Y-%m-%d'),
                'billing_url': f"{settings.FRONTEND_URL}/billing",
                'features_url': f"{settings.FRONTEND_URL}/features"
            }
            
            html_content, text_content = self._render_template(
                self.templates['subscription_update'], context
            )
            
            subject = f"Your ChefoodAI plan has been updated to {new_plan.title()}"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send subscription update to {user.email}: {e}")
            return False
    
    async def send_bulk_email(
        self,
        recipients: List[str],
        subject: str,
        template_name: str,
        context: Dict[str, Any],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Send bulk email with batching
        
        Returns:
            Dictionary with success/failure counts
        """
        results = {
            'sent': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            html_content, text_content = self._render_template(template_name, context)
            
            # Process in batches
            for i in range(0, len(recipients), batch_size):
                batch = recipients[i:i + batch_size]
                
                # Send to batch
                success = await self.send_email(
                    to_email=batch,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                
                if success:
                    results['sent'] += len(batch)
                else:
                    results['failed'] += len(batch)
                    results['errors'].append(f"Batch {i//batch_size + 1} failed")
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Bulk email error: {e}")
        
        return results
    
    async def send_cooking_tips(self, user: User, tips: List[str]) -> bool:
        """Send weekly cooking tips"""
        try:
            if not user.notification_settings.get('marketing', False):
                return True  # User opted out
            
            context = {
                'user_name': user.first_name,
                'tips': tips,
                'dashboard_url': f"{settings.FRONTEND_URL}/dashboard"
            }
            
            html_content, text_content = self._render_template(
                self.templates['cooking_tips'], context
            )
            
            subject = "Weekly cooking tips from ChefoodAI 👨‍🍳"
            
            return await self.send_email(
                to_email=user.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send cooking tips to {user.email}: {e}")
            return False

# Create singleton instance
email_service = EmailService()