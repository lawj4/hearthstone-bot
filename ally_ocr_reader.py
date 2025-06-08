import cv2
import numpy as np
import os

class AllyBoardReader:
    """Class specifically for reading cards from test_ally_board.png using color mask detection"""
    
    def __init__(self):
        # Define target colors in BGR format (OpenCV uses BGR, not RGB)
        self.target_colors = {
            'red': np.array([35, 52, 234]),    # RGB(234,52,35) -> BGR(35,52,234)
            'green': np.array([76, 251, 117]), # RGB(117,251,76) -> BGR(76,251,117)
            'white': np.array([255, 255, 255]) # RGB(255,255,255) -> BGR(255,255,255)
        }
        # Color tolerance for matching
        self.color_tolerance = 30  # Adjust this value to be more/less strict with color matching
        self.white_threshold = 240  # Threshold for white detection (240-255)
        
        # Height restrictions for valid crystals
        self.height_restrictions = {
            'min_top_border': 10,    # Minimum pixels from top of image to crystal top
            'min_bottom_border': 10, # Minimum pixels from crystal bottom to bottom of image
            'min_height': 8,         # Minimum height of crystal itself
            'max_height': 100        # Maximum height of crystal itself
        }
    
    def load_ally_board_image(self):
        """Load test_ally_board.png file"""
        if not os.path.exists('test_ally_board.png'):
            raise FileNotFoundError("test_ally_board.png not found! Please run hearthstone_regions.py first.")
        
        ally_board_image = cv2.imread('test_ally_board.png', cv2.IMREAD_COLOR)
        if ally_board_image is None:
            raise ValueError("Could not load test_ally_board.png")
        
        return ally_board_image
    
    def detect_color_regions(self, ally_board_image):
        """Create color mask for specific red, green, and white colors
        Args:
            ally_board_image: Color image of ally board region
        Returns:
            combined_mask: Binary mask of red, green, and white regions
            red_mask: Binary mask of red regions only
            green_mask: Binary mask of green regions only
            white_mask: Binary mask of white regions only
        """
        # Create masks for each target color
        red_mask = self.create_color_mask(ally_board_image, self.target_colors['red'], 'red')
        green_mask = self.create_color_mask(ally_board_image, self.target_colors['green'], 'green')
        white_mask = self.create_white_mask(ally_board_image)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, white_mask)
        
        return combined_mask, red_mask, green_mask, white_mask
    
    def create_color_mask(self, image, target_color, color_name):
        """Create mask for a specific color with tolerance
        Args:
            image: Input BGR image
            target_color: Target color in BGR format
            color_name: Name of the color for debugging
        Returns:
            mask: Binary mask where target color pixels are white
        """
        # Define lower and upper bounds for the color
        lower_bound = np.clip(target_color - self.color_tolerance, 0, 255)
        upper_bound = np.clip(target_color + self.color_tolerance, 0, 255)
        
        print(f"  {color_name.capitalize()} color detection:")
        print(f"    Target BGR: {target_color}")
        print(f"    Lower bound: {lower_bound}")
        print(f"    Upper bound: {upper_bound}")
        
        # Create mask for this color range
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # Count pixels found
        color_pixels = np.sum(mask == 255)
        print(f"    Found {color_pixels} {color_name} pixels")
        
        return mask
    
    def create_white_mask(self, image):
        """Create mask for white/bright areas using threshold method
        Args:
            image: Input BGR image
        Returns:
            mask: Binary mask where white pixels are white
        """
        # Convert to grayscale for white detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for white pixels
        _, white_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        # Count pixels found
        white_pixels = np.sum(white_mask == 255)
        print(f"  White color detection:")
        print(f"    Threshold: {self.white_threshold}")
        print(f"    Found {white_pixels} white pixels")
        
        return white_mask
    
    def extract_individual_crystals(self, combined_mask):
        """Extract individual crystal contours from color mask with height restrictions"""
        # Get image dimensions for border calculations
        image_height, image_width = combined_mask.shape
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crystal_boxes = []
        valid_contours = []
        rejected_crystals = []
        
        print(f"  Image dimensions: {image_width}x{image_height}")
        print(f"  Height restrictions:")
        print(f"    Min top border: {self.height_restrictions['min_top_border']}px")
        print(f"    Min bottom border: {self.height_restrictions['min_bottom_border']}px")
        print(f"    Min crystal height: {self.height_restrictions['min_height']}px")
        print(f"    Max crystal height: {self.height_restrictions['max_height']}px")
        print()
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle for each crystal
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calculate border distances
            top_border = y  # Distance from top of image to top of crystal
            bottom_border = image_height - (y + h)  # Distance from bottom of crystal to bottom of image
            
            # Check all restrictions
            valid = True
            rejection_reasons = []
            
            # Basic size filters
            if area <= 50:
                valid = False
                rejection_reasons.append(f"area too small ({area})")
            if w <= 5:
                valid = False
                rejection_reasons.append(f"width too small ({w})")
            if h <= 5:
                valid = False
                rejection_reasons.append(f"height too small ({h})")
            
            # Height restrictions
            if top_border < self.height_restrictions['min_top_border']:
                valid = False
                rejection_reasons.append(f"top border too small ({top_border}px < {self.height_restrictions['min_top_border']}px)")
            
            if bottom_border < self.height_restrictions['min_bottom_border']:
                valid = False
                rejection_reasons.append(f"bottom border too small ({bottom_border}px < {self.height_restrictions['min_bottom_border']}px)")
            
            if h < self.height_restrictions['min_height']:
                valid = False
                rejection_reasons.append(f"crystal height too small ({h}px < {self.height_restrictions['min_height']}px)")
            
            if h > self.height_restrictions['max_height']:
                valid = False
                rejection_reasons.append(f"crystal height too large ({h}px > {self.height_restrictions['max_height']}px)")
            
            # Store result
            if valid:
                crystal_boxes.append((x, y, w, h))
                valid_contours.append(contour)
                print(f"  ✓ Crystal {len(crystal_boxes)}: pos=({x},{y}), size=({w}x{h}), area={area}, top_border={top_border}px, bottom_border={bottom_border}px")
            else:
                rejected_crystals.append({
                    'position': (x, y, w, h),
                    'area': area,
                    'top_border': top_border,
                    'bottom_border': bottom_border,
                    'reasons': rejection_reasons
                })
                print(f"  ✗ Rejected: pos=({x},{y}), size=({w}x{h}) - {', '.join(rejection_reasons)}")
        
        print(f"\nSummary: {len(crystal_boxes)} valid crystals, {len(rejected_crystals)} rejected")
        
        # Sort crystals from left to right based on x-coordinate
        if crystal_boxes:
            print(f"\nSorting crystals from left to right...")
            
            # Create list of (crystal_box, contour) pairs for sorting
            crystal_contour_pairs = list(zip(crystal_boxes, valid_contours))
            
            # Sort by x-coordinate (leftmost first)
            crystal_contour_pairs.sort(key=lambda pair: pair[0][0])  # pair[0][0] is the x-coordinate
            
            # Separate back into sorted lists
            crystal_boxes = [pair[0] for pair in crystal_contour_pairs]
            valid_contours = [pair[1] for pair in crystal_contour_pairs]
            
            # Print sorted order
            for i, (x, y, w, h) in enumerate(crystal_boxes):
                print(f"  Crystal {i+1}: x={x} (leftmost to rightmost)")
        
        return crystal_boxes, valid_contours
    
    def analyze_color_mask(self):
        """Analyze color mask from test_ally_board.png and save results"""
        print("Loading test_ally_board.png...")
        
        # Load ally board image
        ally_board_image = self.load_ally_board_image()
        print(f"Ally board image shape: {ally_board_image.shape}")
        
        # Create color masks
        print(f"\nDetecting colors with tolerance ±{self.color_tolerance}:")
        combined_mask, red_mask, green_mask, white_mask = self.detect_color_regions(ally_board_image)
        
        # Calculate statistics
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        color_pixels = np.sum(combined_mask == 255)
        color_percentage = (color_pixels / total_pixels) * 100
        
        print(f"\nColor mask statistics:")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Color pixels found: {color_pixels}")
        print(f"  Color percentage: {color_percentage:.2f}%")
        
        # Extract individual crystals using contours
        crystal_boxes, valid_contours = self.extract_individual_crystals(combined_mask)
        print(f"Found {len(crystal_boxes)} crystals using contour detection")
        
        # Create visualization with contours and bounding boxes
        visualization = ally_board_image.copy()
        
        # Draw contours in yellow (visible against both red and green)
        cv2.drawContours(visualization, valid_contours, -1, (0, 255, 255), 2)
        
        # Draw bounding boxes in blue
        for i, (x, y, w, h) in enumerate(crystal_boxes):
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Add crystal number
            cv2.putText(visualization, str(i+1), (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save individual color masks
        cv2.imwrite('ally_board_red_mask.png', red_mask)
        cv2.imwrite('ally_board_green_mask.png', green_mask)
        cv2.imwrite('ally_board_white_mask.png', white_mask)
        cv2.imwrite('ally_board_combined_mask.png', combined_mask)
        print("Saved: ally_board_red_mask.png")
        print("Saved: ally_board_green_mask.png") 
        print("Saved: ally_board_white_mask.png")
        print("Saved: ally_board_combined_mask.png")
        
        # Save visualization with contours
        cv2.imwrite('ally_board_crystal_contours.png', visualization)
        print("Saved: ally_board_crystal_contours.png")
        
        # Create comprehensive comparison: original | red | green | white | combined | contours
        comparison = np.zeros((ally_board_image.shape[0], ally_board_image.shape[1] * 6, 3), dtype=np.uint8)
        comparison[:, :ally_board_image.shape[1]] = ally_board_image
        comparison[:, ally_board_image.shape[1]:ally_board_image.shape[1]*2] = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        comparison[:, ally_board_image.shape[1]*2:ally_board_image.shape[1]*3] = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        comparison[:, ally_board_image.shape[1]*3:ally_board_image.shape[1]*4] = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        comparison[:, ally_board_image.shape[1]*4:ally_board_image.shape[1]*5] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        comparison[:, ally_board_image.shape[1]*5:] = visualization
        
        cv2.imwrite('ally_board_color_analysis.png', comparison)
        print("Saved: ally_board_color_analysis.png")
        
        return combined_mask, crystal_boxes
    
    def analyze_white_mask(self):
        """Compatibility method - calls analyze_color_mask for backward compatibility"""
        return self.analyze_color_mask()

def main():
    """Test color mask detection with contour analysis on ally board"""
    print("Hearthstone Ally Board Reader - Color Mask Detection")
    print("=" * 55)
    print("Target colors:")
    print("  Red: RGB(234,52,35) -> BGR(35,52,234)")
    print("  Green: RGB(117,251,76) -> BGR(76,251,117)")
    print("  White: RGB(255,255,255) -> BGR(255,255,255)")
    print()
    
    # Check if test_ally_board.png exists
    if not os.path.exists('test_ally_board.png'):
        print("Error: test_ally_board.png not found!")
        print("Please run hearthstone_regions.py first to generate this file.")
        return
    
    try:
        # Create ally board reader
        reader = AllyBoardReader()
        
        # Analyze color mask and extract crystals
        combined_mask, crystal_boxes = reader.analyze_color_mask()
        
        print(f"\n✓ Successfully detected {len(crystal_boxes)} crystals!")
        print(f"✓ Check ally_board_color_analysis.png for full visualization")
        
        # Summary
        if len(crystal_boxes) >= 1:
            print(f"🎯 Found {len(crystal_boxes)} colored regions on ally board")
        else:
            print(f"⚠️  No colored regions found - might need to adjust color tolerance")
            print(f"   Consider increasing color_tolerance (currently {reader.color_tolerance})")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nGenerated files:")
    print("  - ally_board_red_mask.png: Red color mask only")
    print("  - ally_board_green_mask.png: Green color mask only") 
    print("  - ally_board_white_mask.png: White color mask only")
    print("  - ally_board_combined_mask.png: Combined red + green + white mask")
    print("  - ally_board_crystal_contours.png: Contours + bounding boxes")
    print("  - ally_board_color_analysis.png: Original | Red | Green | White | Combined | Contours")
    print("\nTo adjust sensitivity:")
    print("  - Increase color_tolerance (currently 30) to detect more color variations")
    print("  - Decrease color_tolerance to be more strict with exact color matching")
    print("  - Adjust white_threshold (currently 240) for white detection sensitivity")
    print("\nTo adjust height restrictions:")
    print("  - min_top_border: Minimum pixels from top edge to crystal")
    print("  - min_bottom_border: Minimum pixels from crystal to bottom edge") 
    print("  - min_height/max_height: Crystal size limits")

if __name__ == "__main__":
    main()