import cv2
import numpy as np
from PIL import ImageGrab
import time

class ScreenCapture:
    def __init__(self):
        self.last_screenshot = None
        self.last_timestamp = None
    
    def capture_color(self):
        """Capture screenshot in color (BGR format for OpenCV)"""
        screenshot = ImageGrab.grab()
        bgr_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self.last_screenshot = bgr_image
        self.last_timestamp = time.time()
        return bgr_image
    
    def capture_grayscale(self):
        """Capture screenshot in grayscale"""
        screenshot = ImageGrab.grab()
        gray_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        self.last_screenshot = gray_image
        self.last_timestamp = time.time()
        return gray_image
    
    def capture_region(self, x, y, width, height, grayscale=True):
        """Capture specific region of screen"""
        bbox = (x, y, x + width, y + height)
        screenshot = ImageGrab.grab(bbox)
        
        if grayscale:
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        else:
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    def capture_instant(self, grayscale=True):
        """Instant capture without delay"""
        if grayscale:
            return self.capture_grayscale()
        else:
            return self.capture_color()
    
    def save_screenshot(self, image, filename):
        """Save screenshot to file"""
        cv2.imwrite(filename, image)
        print(f"Screenshot saved as '{filename}'")
    
    def get_screen_dimensions(self):
        """Get screen dimensions"""
        screenshot = ImageGrab.grab()
        return screenshot.size  # Returns (width, height)

def quick_capture(grayscale=True):
    """Quick function for one-off captures"""
    capturer = ScreenCapture()
    return capturer.capture_instant(grayscale)

if __name__ == "__main__":
    # Test the screenshot capture
    capturer = ScreenCapture()
    
    print("Screen dimensions:", capturer.get_screen_dimensions())
    
    # Test grayscale capture
    gray_img = capturer.capture_instant(grayscale=True)
    capturer.save_screenshot(gray_img, 'test_grayscale.png')
    
    print(f"Image shape: {gray_img.shape}")
    print("Screenshot capture module working!")