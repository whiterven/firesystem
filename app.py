import cv2
import numpy as np
import smtplib
import requests
import logging
import json
import threading
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple
import queue
import os

class FireDetectionConfig:
    """Configuration management for the fire detection system"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            self._create_default_config()
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'camera': {
                'device_id': 0,
                'resolution': {'width': 640, 'height': 480},
                'fps': 30
            },
            'detection': {
                'fire_threshold': 0.6,
                'min_fire_area': 100,
                'color_ranges': {
                    'lower': [0, 50, 50],
                    'upper': [20, 255, 255]
                }
            },
            'alerts': {
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': '',
                    'sender_password': '',
                    'recipient_emails': []
                },
                'emergency_services': {
                    'enabled': False,
                    'api_key': '',
                    'emergency_number': '911'
                },
                'cooldown_period': 300  # 5 minutes between alerts
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'fire_detection.log'
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f)

class FireDetector:
    """Core fire detection logic using computer vision"""
    def __init__(self, config: FireDetectionConfig):
        self.config = config.config
        self.logger = self._setup_logger()
        self.frame_queue = queue.Queue(maxsize=30)
        self.alert_queue = queue.Queue()
        self.last_alert_time = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('FireDetection')
        logger.setLevel(self.config['logging']['level'])
        handler = logging.FileHandler(self.config['logging']['file_path'])
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def detect_fire(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """
        Detect fire in the given frame using multiple detection methods
        Returns: (fire_detected, processed_frame, confidence)
        """
        # Convert to HSV color space for better fire detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for fire-like colors
        lower = np.array(self.config['detection']['lower'])
        upper = np.array(self.config['detection']['upper'])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of potential fire regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False
        confidence = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config['detection']['min_fire_area']:
                # Calculate confidence based on area and color intensity
                roi = frame[contour]
                color_confidence = np.mean(roi[:, :, 2]) / 255.0  # Red channel intensity
                size_confidence = min(1.0, area / 10000)
                confidence = max(confidence, (color_confidence + size_confidence) / 2)
                
                if confidence > self.config['detection']['fire_threshold']:
                    fire_detected = True
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                    
        return fire_detected, frame, confidence

class AlertManager:
    """Handle different types of alerts when fire is detected"""
    def __init__(self, config: FireDetectionConfig):
        self.config = config.config
        self.logger = logging.getLogger('FireDetection.AlertManager')
        
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

    def contact_emergency_services(self, confidence: float):
        """Contact emergency services through API or automated call"""
        if not self.config['alerts']['emergency_services']['enabled']:
            return
            
        try:
            # This is a placeholder for emergency services API integration
            # In a real implementation, you would integrate with your local
            # emergency services' API or use a service like Twilio for automated calls
            
            payload = {
                'api_key': self.config['alerts']['emergency_services']['api_key'],
                'emergency_number': self.config['alerts']['emergency_services']['emergency_number'],
                'message': f'Fire detected with {confidence:.2%} confidence at location: [ADD LOCATION]',
                'timestamp': datetime.now().isoformat()
            }
            
            # Placeholder API endpoint
            # response = requests.post('https://emergency-services-api.example.com/alert', json=payload)
            # response.raise_for_status()
            
            self.logger.info("Emergency services contacted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to contact emergency services: {str(e)}")

class FireDetectionSystem:
    """Main system class that coordinates all components"""
    def __init__(self):
        self.config = FireDetectionConfig()
        self.detector = FireDetector(self.config)
        self.alert_manager = AlertManager(self.config)
        self.camera = None
        self.is_running = False
        self.processing_thread = None
        self.alert_thread = None
        
    def initialize_camera(self):
        """Initialize the webcam"""
        try:
            self.camera = cv2.VideoCapture(self.config.config['camera']['device_id'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 
                          self.config.config['camera']['resolution']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 
                          self.config.config['camera']['resolution']['height'])
            self.camera.set(cv2.CAP_PROP_FPS, 
                          self.config.config['camera']['fps'])
            
            if not self.camera.isOpened():
                raise Exception("Failed to open camera")
                
        except Exception as e:
            self.detector.logger.error(f"Camera initialization failed: {str(e)}")
            raise

    def process_frames(self):
        """Process frames from the camera in a separate thread"""
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                    
                fire_detected, processed_frame, confidence = self.detector.detect_fire(frame)
                
                if fire_detected:
                    current_time = time.time()
                    if (current_time - self.detector.last_alert_time > 
                            self.config.config['alerts']['cooldown_period']):
                        self.detector.alert_queue.put((processed_frame, confidence))
                        self.detector.last_alert_time = current_time
                
                self.detector.frame_queue.put(processed_frame)
                
            except Exception as e:
                self.detector.logger.error(f"Error processing frame: {str(e)}")
                time.sleep(1)

    def handle_alerts(self):
        """Handle alerts in a separate thread"""
        while self.is_running:
            try:
                frame, confidence = self.detector.alert_queue.get(timeout=1)
                
                # Send email alert
                if self.config.config['alerts']['email']['enabled']:
                    self.alert_manager.send_email_alert(frame, confidence)
                
                # Contact emergency services
                if self.config.config['alerts']['emergency_services']['enabled']:
                    self.alert_manager.contact_emergency_services(confidence)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.detector.logger.error(f"Error handling alert: {str(e)}")

    def start(self):
        """Start the fire detection system"""
        try:
            self.initialize_camera()
            self.is_running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_frames)
            self.processing_thread.start()
            
            # Start alert thread
            self.alert_thread = threading.Thread(target=self.handle_alerts)
            self.alert_thread.start()
            
            # Main loop for displaying frames
            while self.is_running:
                try:
                    frame = self.detector.frame_queue.get(timeout=1)
                    cv2.imshow('Fire Detection System', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()
                        break
                        
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            self.stop()
        finally:
            cv2.destroyAllWindows()

    def stop(self):
        """Stop the fire detection system"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        if self.alert_thread:
            self.alert_thread.join()
        if self.camera:
            self.camera.release()
        self.detector.logger.info("Fire detection system stopped")

if __name__ == "__main__":
    system = FireDetectionSystem()
    system.start()