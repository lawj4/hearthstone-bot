import cv2
import numpy as np
import os

class HandReader:
    """Class specifically for reading cards from test_hand.png using white mask detection"""
    
    def __init__(self):
        pass
    
    def load_hand_image(self):
        """Load test_hand.png file"""
        if not os.path.exists('test_hand.png'):
            raise FileNotFoundError("test_hand.png not found! Please run hearthstone_regions.py first.")
        
        hand_image = cv2.imread('test_hand.png', cv2.IMREAD_COLOR)
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
        """Extract individual crystal contours from white mask"""
        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crystal_boxes = []
        valid_contours = []
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle for each crystal
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter out noise (adjust these values based on your crystal size)
            if area > 50 and w > 5 and h > 5:
                crystal_boxes.append((x, y, w, h))
                valid_contours.append(contour)
                print(f"  Crystal {len(crystal_boxes)}: position=({x},{y}), size=({w}x{h}), area={area}")
        
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
    if not os.path.exists('test_hand.png'):
        print("Error: test_hand.png not found!")
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
            print("   Consider decreasing threshold value (currently 200)")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nGenerated files:")
    print("  - white_mask.png: Pure binary mask of white areas")
    print("  - crystal_contours.png: Contours + bounding boxes")
    print("  - crystal_analysis.png: Original | Mask | Contours side-by-side")
    print("\nTo adjust sensitivity:")
    print("  - Lower threshold (< 200) to detect more white areas")
    print("  - Raise threshold (> 200) to detect only brightest areas")

if __name__ == "__main__":
    main()