#main.py

import cv2
import yaml
import threading
import queue
import time
from detector import FireDetector
from alert_manager import AlertManager

class FireDetectionSystem:
    """Main system class that coordinates all components"""
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.detector = FireDetector(self.config)
        self.alert_manager = AlertManager(self.config)
        self.camera = None
        self.is_running = False
        self.processing_thread = None
        self.alert_thread = None
        
    def initialize_camera(self):
        """Initialize the webcam"""
        try:
            self.camera = cv2.VideoCapture(self.config['camera']['device_id'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 
                          self.config['camera']['resolution']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 
                          self.config['camera']['resolution']['height'])
            self.camera.set(cv2.CAP_PROP_FPS, 
                          self.config['camera']['fps'])
            
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
                            self.config['alerts']['cooldown_period']):
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
                if self.config['alerts']['email']['enabled']:
                    self.alert_manager.send_email_alert(frame, confidence)
                
                # Make emergency call
                if self.config['alerts']['twilio']['enabled']:
                    self.alert_manager.make_emergency_call(confidence)
                    
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