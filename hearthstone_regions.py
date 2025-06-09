import cv2
import numpy as np
import logging
import os

class HearthstoneRegions:
    """Class to extract specific regions from Hearthstone screenshots"""
    
    def __init__(self, screen_width=None, screen_height=None):
        # Set up logging
        self.logger = logging.getLogger('HearthstoneRegions')
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
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.logger.info("HearthstoneRegions initialized")
    
    def get_hand_region(self, image, color=True):
        """Extract the hand area from bottom of screen
        Args:
            image: Input image (can be color or grayscale)
            color: If True, returns color region; if False, returns grayscale
        """
        height, width = image.shape[:2]
        # Hand is typically bottom 20% of screen
        
        hand_region = image[int(height * 0.85):height, int(width * 0.27):int(width * 0.65)]
        
        self.logger.debug(f"Extracted hand region: {hand_region.shape}, color={color}")
        
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
        mana_region = image[int(height * 0.92):int(height * 0.94), int(width * 0.65):int(width * 0.7)]
        
        self.logger.debug(f"Extracted mana crystals region: {mana_region.shape}, color={color}")
        
        # Convert to grayscale if requested and image is color
        if not color and len(image.shape) == 3:
            mana_region = cv2.cvtColor(mana_region, cv2.COLOR_BGR2GRAY)
        
        return mana_region
    
    def get_ally_board_region(self, image, color=True):
        """Extract the ally board area (lower center)"""
        height, width = image.shape[:2]
        # Ally board is roughly lower middle section
        ally_board_region = image[int(height * 0.59):int(height * 0.63), 
                           int(width * 0.15):int(width * 0.85)]
        
        self.logger.debug(f"Extracted ally board region: {ally_board_region.shape}, color={color}")
        return ally_board_region
    
    def get_enemy_board_region(self, image, color=True):
        """Extract the enemy board area (upper center)"""
        height, width = image.shape[:2]
        # Enemy board is roughly upper middle section, above ally board
        enemy_board_region = image[int(height * 0.37):int(height * 0.41), 
                            int(width * 0.15):int(width * 0.85)]
        
        self.logger.debug(f"Extracted enemy board region: {enemy_board_region.shape}, color={color}")
        return enemy_board_region
    
    def get_enemy_health_region(self, image, color=True):
        """Extract enemy health area (top of screen)"""
        height, width = image.shape[:2]
        # Enemy health is top center of screen
        enemy_health = image[int(height * 0.255):int(height * 0.295), int(width * 0.53):int(width * 0.555)]
        
        self.logger.debug(f"Extracted enemy health region: {enemy_health.shape}, color={color}")
        return enemy_health
    
    def get_ally_health_region(self, image, color=True):
        """Extract ally health area (bottom center)"""
        height, width = image.shape[:2]
        # Ally health is bottom center, similar positioning to enemy but lower
        ally_health = image[int(height * 0.82):int(height * 0.86), int(width * 0.53):int(width * 0.555)]
        
        self.logger.debug(f"Extracted ally health region: {ally_health.shape}, color={color}")
        return ally_health
    
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
        
        self.logger.debug(f"Found {len(card_boxes)} card regions in hand")
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
        
        self.logger.debug(f"Extracted {len(regions)} card subregions")
        return regions
    
    def save_regions_as_images(self, image, prefix='region', hand_in_color=True):
        """Save different regions as PNG files
        Args:
            image: Input image (should be color BGR format)
            prefix: Filename prefix
            hand_in_color: If True, saves hand region in color; if False, in grayscale
        """
        self.logger.info(f"Saving regions with prefix '{prefix}'")
        
        # Get regions with appropriate color settings
        regions = {
            'hand': self.get_hand_region(image, color=hand_in_color),
            'mana_crystals': self.get_mana_crystals_region(image, color=False),  # Keep mana in grayscale for OCR
            'ally_board': self.get_ally_board_region(image, color=True),
            'enemy_board': self.get_enemy_board_region(image, color=True),
            'enemy_health': self.get_enemy_health_region(image, color=True),
            'ally_health': self.get_ally_health_region(image, color=True)
        }
        
        # Save original
        os.makedirs("images", exist_ok=True)

        # Save full screenshot
        full_path = os.path.join("images", f"{prefix}_full_screenshot.png")
        cv2.imwrite(full_path, image)
        self.logger.debug(f"Saved: {full_path}")

        # Save each region
        for name, region in regions.items():
            region_path = os.path.join("images", f"{prefix}_{name}.png")
            cv2.imwrite(region_path, region)
            color_info = "color" if len(region.shape) == 3 else "grayscale"
            self.logger.debug(f"Saved: {region_path} ({color_info})")
        
        self.logger.info(f"Successfully saved {len(regions) + 1} region files")
        return regions

if __name__ == "__main__":
    # Set up logging for main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('HearthstoneRegionsMain')
    
    # Test with a sample image
    from screenshot_capture import quick_capture, ScreenCapture
    
    logger.info("Testing region extraction with color support...")
    
    # Capture a test image in COLOR (not grayscale)
    capturer = ScreenCapture()
    image = capturer.capture_color()  # Get color image
    
    # Test region extraction
    regions = HearthstoneRegions()
    
    # Save all regions as PNG files - hand in color, mana crystals in grayscale
    regions.save_regions_as_images(image, 'preprocess', hand_in_color=True)
    
    # Test hand card detection with color image
    hand_region = regions.get_hand_region(image, color=True)  # Get color hand
    logger.info(f"Hand region shape: {hand_region.shape} ({'color' if len(hand_region.shape) == 3 else 'grayscale'})")
    
    card_boxes = regions.find_card_regions_in_hand(hand_region)
    logger.info(f"Found {len(card_boxes)} cards in hand")
    
    for i, (x, y, w, h) in enumerate(card_boxes):
        logger.info(f"Card {i+1}: position=({x},{y}), size=({w}x{h})")