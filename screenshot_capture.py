import cv2
import numpy as np
from PIL import ImageGrab
import time
import logging
import os

class ScreenCapture:
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger('ScreenCapture')
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create file handler if it doesn't exist
        if not self.logger.handlers:
            file_handler = logging.FileHandler('logs/hearthstone_debug.log')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.last_screenshot = None
        self.last_timestamp = None
        
        self.logger.info("ScreenCapture initialized")
    
    def capture_color(self):
        """Capture screenshot in color (BGR format for OpenCV)"""
        self.logger.debug("Capturing color screenshot...")
        screenshot = ImageGrab.grab()
        bgr_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self.last_screenshot = bgr_image
        self.last_timestamp = time.time()
        
        self.logger.debug(f"Color screenshot captured: {bgr_image.shape}")
        return bgr_image
    
    def capture_grayscale(self):
        """Capture screenshot in grayscale"""
        self.logger.debug("Capturing grayscale screenshot...")
        screenshot = ImageGrab.grab()
        gray_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        self.last_screenshot = gray_image
        self.last_timestamp = time.time()
        
        self.logger.debug(f"Grayscale screenshot captured: {gray_image.shape}")
        return gray_image
    
    def capture_region(self, x, y, width, height, grayscale=True):
        """Capture specific region of screen"""
        bbox = (x, y, x + width, y + height)
        self.logger.debug(f"Capturing region: {bbox}, grayscale={grayscale}")
        
        screenshot = ImageGrab.grab(bbox)
        
        if grayscale:
            result = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        else:
            result = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        self.logger.debug(f"Region screenshot captured: {result.shape}")
        return result
    
    def capture_instant(self, grayscale=True):
        """Instant capture without delay"""
        if grayscale:
            return self.capture_grayscale()
        else:
            return self.capture_color()
    
    def save_screenshot(self, image, filename):
        """Save screenshot to file"""
        cv2.imwrite(filename, image)
        self.logger.debug(f"Screenshot saved as '{filename}'")
    
    def get_screen_dimensions(self):
        """Get screen dimensions"""
        screenshot = ImageGrab.grab()
        dimensions = screenshot.size  # Returns (width, height)
        self.logger.debug(f"Screen dimensions: {dimensions}")
        return dimensions

def quick_capture(grayscale=True):
    """Quick function for one-off captures"""
    capturer = ScreenCapture()
    return capturer.capture_instant(grayscale)

if __name__ == "__main__":
    # Set up logging for main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('ScreenCaptureMain')
    
    # Test the screenshot capture
    capturer = ScreenCapture()
    
    logger.info("Testing screenshot capture module...")
    logger.info(f"Screen dimensions: {capturer.get_screen_dimensions()}")
    
    # Test grayscale capture
    gray_img = capturer.capture_instant(grayscale=True)
    capturer.save_screenshot(gray_img, 'test_grayscale.png')
    
    logger.info(f"Image shape: {gray_img.shape}")
    logger.info("Screenshot capture module working!")