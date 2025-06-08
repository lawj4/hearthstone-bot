import cv2
import numpy as np

class HearthstoneRegions:
    """Class to extract specific regions from Hearthstone screenshots"""
    
    def __init__(self, screen_width=None, screen_height=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
    
    def get_hand_region(self, image, color=True):
        """Extract the hand area from bottom of screen
        Args:
            image: Input image (can be color or grayscale)
            color: If True, returns color region; if False, returns grayscale
        """
        height, width = image.shape[:2]
        # Hand is typically bottom 20% of screen
        
        hand_region = image[int(height * 0.85):height, int(width * 0.27):int(width * 0.65)]
        
        # Convert to grayscale if requested and image is color
        if not color and len(image.shape) == 3:
            hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        
        return hand_region
    
    def get_mana_crystals_region(self, image, color=False):
        """Extract the mana crystals area (bottom middle right)
        Args:
            image: Input image (can be color or grayscale)
            color: If True, returns color region; if False, returns grayscale
        """
        height, width = image.shape[:2]
        # Mana crystals are bottom-left corner
        mana_region = image[int(height * 0.9):int(height * 0.95), int(width * 0.65):int(width * 0.7)]
        
        # Convert to grayscale if requested and image is color
        if not color and len(image.shape) == 3:
            mana_region = cv2.cvtColor(mana_region, cv2.COLOR_BGR2GRAY)
        
        return mana_region
    
    def get_board_region(self, image, color=True):
        """Extract the game board area (center)"""
        height, width = image.shape[:2]
        # Board is roughly middle 40% vertically, center 60% horizontally
        board_region = image[int(height * 0.3):int(height * 0.7), 
                           int(width * 0.2):int(width * 0.8)]
        return board_region
    
    def get_enemy_hand_region(self, image, color=True):
        """Extract enemy hand area (top of screen)"""
        height, width = image.shape[:2]
        # Enemy hand is top 15% of screen
        enemy_hand = image[:int(height * 0.15), :]
        return enemy_hand
    
    def get_hero_portraits_region(self, image, color=True):
        """Extract both hero portraits"""
        height, width = image.shape[:2]
        # Heroes are on right side, vertically centered
        heroes_region = image[int(height * 0.2):int(height * 0.8), 
                            int(width * 0.85):]
        return heroes_region
    
    def find_card_regions_in_hand(self, hand_region):
        """Find individual card bounding boxes in hand region
        Works with both color and grayscale images
        """
        # Convert to grayscale for edge detection if needed
        if len(hand_region.shape) == 3:
            gray_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_hand = hand_region
        
        # Edge detection
        edges = cv2.Canny(gray_hand, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        card_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio (cards are taller than wide)
                if h > w * 1.2 and w > 50 and h > 80:
                    card_boxes.append((x, y, w, h))
        
        # Sort by x-position (left to right)
        card_boxes.sort(key=lambda box: box[0])
        return card_boxes
    
    def extract_card_subregions(self, card_region):
        """Extract important parts of a single card"""
        height, width = card_region.shape[:2]
        
        regions = {
            'mana_cost': card_region[:int(height * 0.25), :int(width * 0.25)],
            'card_art': card_region[int(height * 0.15):int(height * 0.7), 
                                  int(width * 0.1):int(width * 0.9)],
            'attack': card_region[int(height * 0.75):, :int(width * 0.3)],
            'health': card_region[int(height * 0.75):, int(width * 0.7):],
            'name_area': card_region[int(height * 0.7):int(height * 0.85), :]
        }
        
        return regions
    
    def save_regions_as_images(self, image, prefix='region', hand_in_color=True):
        """Save different regions as PNG files
        Args:
            image: Input image (should be color BGR format)
            prefix: Filename prefix
            hand_in_color: If True, saves hand region in color; if False, in grayscale
        """
        import os
        
        # Get regions with appropriate color settings
        regions = {
            'hand': self.get_hand_region(image, color=hand_in_color),
            'mana_crystals': self.get_mana_crystals_region(image, color=False),  # Keep mana in grayscale for OCR
            'board': self.get_board_region(image, color=True),
            'enemy_hand': self.get_enemy_hand_region(image, color=True),
            'hero_portraits': self.get_hero_portraits_region(image, color=True)
        }
        
        # Save original
        cv2.imwrite(f'{prefix}_full_screenshot.png', image)
        print(f"Saved: {prefix}_full_screenshot.png")
        
        # Save each region
        for name, region in regions.items():
            filename = f'{prefix}_{name}.png'
            cv2.imwrite(filename, region)
            color_info = "color" if len(region.shape) == 3 else "grayscale"
            print(f"Saved: {filename} ({color_info})")
        
        return regions

if __name__ == "__main__":
    # Test with a sample image
    from screenshot_capture import quick_capture, ScreenCapture
    
    print("Testing region extraction with color support...")
    
    # Capture a test image in COLOR (not grayscale)
    capturer = ScreenCapture()
    image = capturer.capture_color()  # Get color image
    
    # Test region extraction
    regions = HearthstoneRegions()
    
    # Save all regions as PNG files - hand in color, mana crystals in grayscale
    regions.save_regions_as_images(image, 'test', hand_in_color=True)
    
    # Test hand card detection with color image
    hand_region = regions.get_hand_region(image, color=True)  # Get color hand
    print(f"Hand region shape: {hand_region.shape} ({'color' if len(hand_region.shape) == 3 else 'grayscale'})")
    
    card_boxes = regions.find_card_regions_in_hand(hand_region)
    print(f"Found {len(card_boxes)} cards in hand")
    
    for i, (x, y, w, h) in enumerate(card_boxes):
        print(f"Card {i+1}: position=({x},{y}), size=({w}x{h})")