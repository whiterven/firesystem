#detector.py

import cv2
import numpy as np
import logging
from typing import Tuple
import queue
import time
from datetime import datetime

class FireDetector:
    """Core fire detection logic using computer vision"""
    def __init__(self, config: dict):
        self.config = config
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
        lower = np.array(self.config['detection']['color_ranges']['lower'])
        upper = np.array(self.config['detection']['color_ranges']['upper'])
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
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                color_confidence = np.mean(roi[:, :, 2]) / 255.0  # Red channel intensity
                size_confidence = min(1.0, area / 10000)
                confidence = max(confidence, (color_confidence + size_confidence) / 2)
                
                if confidence > self.config['detection']['fire_threshold']:
                    fire_detected = True
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Fire! ({confidence:.2%})", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
        return fire_detected, frame, confidence