import cv2
import numpy as np
import os

class HandReader:
    """Class specifically for reading cards from test_hand.png using white mask detection"""
    
    def __init__(self):
        # Height restrictions for valid crystals
        self.height_restrictions = {
            'min_height': 20,         # Minimum height of crystal itself
            'max_height': 40         # Maximum height of crystal itself
        }
    
    def load_hand_image(self):
        """Load test_hand.png file"""
        file_name = "images/preprocess_hand.png"
        if not os.path.exists(file_name):
            raise FileNotFoundError("test_hand.png not found! Please run hearthstone_regions.py first.")
        
        hand_image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        if hand_image is None:
            raise ValueError("Could not load test_hand.png")
        
        return hand_image
    
    def detect_white_crystals(self, hand_image):
        """Create white mask for crystal detection
        Args:
            hand_image: Color image of hand region
        Returns:
            white_mask: Binary mask of white regions
        """
        # Convert to grayscale for white detection
        gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for white/bright areas
        # Threshold for white pixels (adjust this value: 200-255 range)
        _, white_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        
        return white_mask
    
    def extract_individual_crystals(self, white_mask):
        """Extract individual crystal contours from white mask with height restrictions"""
        # Get image dimensions
        image_height, image_width = white_mask.shape
        
        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crystal_boxes = []
        valid_contours = []
        rejected_crystals = []
        
        print(f"  Image dimensions: {image_width}x{image_height}")
        print(f"  Height restrictions:")
        print(f"    Min crystal height: {self.height_restrictions['min_height']}px")
        print(f"    Max crystal height: {self.height_restrictions['max_height']}px")
        print()
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle for each crystal
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
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
                print(f"  ✓ Crystal {len(crystal_boxes)}: pos=({x},{y}), size=({w}x{h}), area={area}")
            else:
                rejected_crystals.append({
                    'position': (x, y, w, h),
                    'area': area,
                    'reasons': rejection_reasons
                })
                print(f"  ✗ Rejected: pos=({x},{y}), size=({w}x{h}) - {', '.join(rejection_reasons)}")
        
        print(f"\nSummary: {len(crystal_boxes)} valid crystals, {len(rejected_crystals)} rejected")
        return crystal_boxes, valid_contours
    
    def analyze_white_mask(self):
        """Analyze white mask from test_hand.png and save results"""
        print("Loading test_hand.png...")
        
        # Load hand image
        hand_image = self.load_hand_image()
        print(f"Hand image shape: {hand_image.shape}")
        
        # Create white mask
        white_mask = self.detect_white_crystals(hand_image)
        
        # Calculate statistics
        total_pixels = white_mask.shape[0] * white_mask.shape[1]
        white_pixels = np.sum(white_mask == 255)
        white_percentage = (white_pixels / total_pixels) * 100
        
        print(f"White mask statistics:")
        print(f"  Total pixels: {total_pixels}")
        print(f"  White pixels: {white_pixels}")
        print(f"  White percentage: {white_percentage:.2f}%")
        
        # Extract individual crystals using contours
        crystal_boxes, valid_contours = self.extract_individual_crystals(white_mask)
        print(f"Found {len(crystal_boxes)} crystals using contour detection")
        
        # Create visualization with contours and bounding boxes
        visualization = hand_image.copy()
        
        # Draw contours in green
        cv2.drawContours(visualization, valid_contours, -1, (0, 255, 0), 2)
        
        # Draw bounding boxes in red
        for i, (x, y, w, h) in enumerate(crystal_boxes):
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Add crystal number
            cv2.putText(visualization, str(i+1), (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save white mask
        cv2.imwrite('white_mask.png', white_mask)
        print("Saved: white_mask.png")
        
        # Save visualization with contours
        cv2.imwrite('crystal_contours.png', visualization)
        print("Saved: crystal_contours.png")
        
        # Create side-by-side comparison: original | mask | contours
        comparison = np.zeros((hand_image.shape[0], hand_image.shape[1] * 3, 3), dtype=np.uint8)
        comparison[:, :hand_image.shape[1]] = hand_image
        comparison[:, hand_image.shape[1]:hand_image.shape[1]*2] = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        comparison[:, hand_image.shape[1]*2:] = visualization
        
        cv2.imwrite('crystal_analysis.png', comparison)
        print("Saved: crystal_analysis.png")
        
        return white_mask, crystal_boxes

def main():
    """Test white mask detection with contour analysis"""
    print("Hearthstone Hand Reader - White Mask Detection")
    print("=" * 50)
    
    # Check if test_hand.png exists
    file_name = "images/preprocess_hand.png"
    if not os.path.exists(file_name):
        print(f"Error: {file_name} not found!")
        print("Please run hearthstone_regions.py first to generate this file.")
        return
    
    try:
        # Create hand reader
        reader = HandReader()
        
        # Analyze white mask and extract crystals
        white_mask, crystal_boxes = reader.analyze_white_mask()
        
        print(f"\n✓ Successfully detected {len(crystal_boxes)} crystals!")
        print(f"✓ Check crystal_analysis.png for full visualization")
        
        # Summary
        if len(crystal_boxes) == 9:
            print("🎯 Perfect! Found exactly 9 crystals as expected")
        elif len(crystal_boxes) > 9:
            print(f"⚠️  Found {len(crystal_boxes)} crystals - might be detecting noise")
            print("   Consider increasing size filters or threshold value")
        else:
            print(f"⚠️  Only found {len(crystal_boxes)} crystals - might be missing some")
            print("   Consider decreasing threshold value (currently 250)")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nGenerated files:")
    print("  - white_mask.png: Pure binary mask of white areas")
    print("  - crystal_contours.png: Contours + bounding boxes")
    print("  - crystal_analysis.png: Original | Mask | Contours side-by-side")
    print("\nTo adjust sensitivity:")
    print("  - Lower threshold (< 250) to detect more white areas")
    print("  - Raise threshold (> 250) to detect only brightest areas")
    print("\nTo adjust height restrictions:")
    print(f"  - min_height: Minimum crystal height (currently {reader.height_restrictions['min_height']}px)")
    print(f"  - max_height: Maximum crystal height (currently {reader.height_restrictions['max_height']}px)")

if __name__ == "__main__":
    main()