#alert_manager.py

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import cv2
from twilio.rest import Client
import numpy as np

class AlertManager:
    """Handle different types of alerts when fire is detected"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger('FireDetection.AlertManager')
        if self.config['alerts']['twilio']['enabled']:
            self.twilio_client = Client(
                self.config['alerts']['twilio']['account_sid'],
                self.config['alerts']['twilio']['auth_token']
            )
        
    def send_email_alert(self, frame: np.ndarray, confidence: float):
        """Send email alert with the fire detection image"""
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f'FIRE DETECTED! Confidence: {confidence:.2%}'
            msg['From'] = self.config['alerts']['email']['sender_email']
            
            # Add text
            text = MIMEText(f'''FIRE DETECTED in your monitored area!
            Time: {datetime.now()}
            Confidence: {confidence:.2%}
            
            Please take immediate action and verify the situation.
            ''')
            msg.attach(text)
            
            # Add image
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
            image = MIMEImage(img_bytes)
            msg.attach(image)
            
            # Send email
            with smtplib.SMTP(self.config['alerts']['email']['smtp_server'], 
                            self.config['alerts']['email']['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.config['alerts']['email']['sender_email'],
                    self.config['alerts']['email']['sender_password']
                )
                for recipient in self.config['alerts']['email']['recipient_emails']:
                    server.send_message(msg, to_addrs=recipient)
                    
            self.logger.info("Email alert sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")

    def make_emergency_call(self, confidence: float):
        """Make emergency call using Twilio"""
        if not self.config['alerts']['twilio']['enabled']:
            return
            
        try:
            message = f'''This is an automated emergency alert from your fire detection system. 
            Fire has been detected with {confidence:.2%} confidence. 
            Please verify and take immediate action.'''
            
            # Make calls to all configured numbers
            for to_number in self.config['alerts']['twilio']['to_numbers']:
                call = self.twilio_client.calls.create(
                    twiml=f'<Response><Say>{message}</Say></Response>',
                    to=to_number,
                    from_=self.config['alerts']['twilio']['from_number']
                )
                self.logger.info(f"Emergency call initiated to {to_number}, SID: {call.sid}")
            
        except Exception as e:
            self.logger.error(f"Failed to make emergency call: {str(e)}")