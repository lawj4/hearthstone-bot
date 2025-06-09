import cv2
import numpy as np
import os
import utils

class HeroHealthReader:
    """Class for reading hero health using color mask detection on enemy and ally health regions"""
    
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
        
        # Size restrictions for valid health indicators
        self.size_restrictions = {
            'min_area': 20,          # Minimum area of color region
            'min_width': 3,          # Minimum width of region
            'min_height': 30,         # Minimum height of region
            'max_area': 2000         # Maximum area to filter out large noise
        }
    
    def load_health_image(self, health_type='enemy'):
        """Load enemy or ally health image file
        Args:
            health_type: 'enemy' or 'ally'
        """
        filename = f'images/preprocess_{health_type}_health.png'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found! Please run hearthstone_regions.py first.")
        
        health_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if health_image is None:
            raise ValueError(f"Could not load {filename}")
        
        return health_image
    
    def detect_color_regions(self, health_image):
        """Create color mask for specific red, green, and white colors
        Args:
            health_image: Color image of health region
        Returns:
            combined_mask: Binary mask of red, green, and white regions
            red_mask: Binary mask of red regions only
            green_mask: Binary mask of green regions only
            white_mask: Binary mask of white regions only
        """
        # Create masks for each target color
        red_mask = self.create_color_mask(health_image, self.target_colors['red'], 'red')
        green_mask = self.create_color_mask(health_image, self.target_colors['green'], 'green')
        white_mask = self.create_white_mask(health_image)
        
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
    
    def extract_color_regions(self, combined_mask):
        """Extract individual color regions from mask with size restrictions"""
        # Get image dimensions
        image_height, image_width = combined_mask.shape
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color_boxes = []
        valid_contours = []
        rejected_regions = []
        
        print(f"  Image dimensions: {image_width}x{image_height}")
        print(f"  Size restrictions:")
        print(f"    Min area: {self.size_restrictions['min_area']}px²")
        print(f"    Max area: {self.size_restrictions['max_area']}px²")
        print(f"    Min width: {self.size_restrictions['min_width']}px")
        print(f"    Min height: {self.size_restrictions['min_height']}px")
        print()
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle for each color region
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Check all restrictions
            valid = True
            rejection_reasons = []
            
            # Size filters
            if area < self.size_restrictions['min_area']:
                valid = False
                rejection_reasons.append(f"area too small ({area}px²)")
            if area > self.size_restrictions['max_area']:
                valid = False
                rejection_reasons.append(f"area too large ({area}px²)")
            if w < self.size_restrictions['min_width']:
                valid = False
                rejection_reasons.append(f"width too small ({w}px)")
            if h < self.size_restrictions['min_height']:
                valid = False
                rejection_reasons.append(f"height too small ({h}px)")
            
            # Store result
            if valid:
                color_boxes.append((x, y, w, h))
                valid_contours.append(contour)
                print(f"  ✓ Color region {len(color_boxes)}: pos=({x},{y}), size=({w}x{h}), area={area}px²")
            else:
                rejected_regions.append({
                    'position': (x, y, w, h),
                    'area': area,
                    'reasons': rejection_reasons
                })
                print(f"  ✗ Rejected: pos=({x},{y}), size=({w}x{h}) - {', '.join(rejection_reasons)}")
        
        print(f"\nSummary: {len(color_boxes)} valid color regions, {len(rejected_regions)} rejected")
        return color_boxes, valid_contours
    
    def analyze_health_colors(self, health_type='enemy'):
        """Analyze color mask from health image and save results
        Args:
            health_type: 'enemy' or 'ally'
        """
        print(f"Loading images/preprocess_{health_type}_health.png...")
        
        # Load health image
        health_image = self.load_health_image(health_type)
        print(f"Health image shape: {health_image.shape}")
        
        # Create color masks
        print(f"\nDetecting colors with tolerance ±{self.color_tolerance}:")
        combined_mask, red_mask, green_mask, white_mask = self.detect_color_regions(health_image)
        
        # Calculate statistics
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        color_pixels = np.sum(combined_mask == 255)
        color_percentage = (color_pixels / total_pixels) * 100
        
        print(f"\nColor mask statistics:")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Color pixels found: {color_pixels}")
        print(f"  Color percentage: {color_percentage:.2f}%")
        
        # Extract individual color regions using contours
        color_boxes, valid_contours = self.extract_color_regions(combined_mask)
        print(f"Found {len(color_boxes)} color regions using contour detection")
        
        # Analyze what colors were found
        color_analysis = self.analyze_detected_colors(health_image, color_boxes, red_mask, green_mask, white_mask)
        
        # Create visualization with contours and bounding boxes
        visualization = health_image.copy()
        
        # Draw contours in yellow (visible against all colors)
        cv2.drawContours(visualization, valid_contours, -1, (0, 255, 255), 2)
        
        # Draw bounding boxes in blue
        for i, (x, y, w, h) in enumerate(color_boxes):
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Add region number
            cv2.putText(visualization, str(i+1), (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save results (only if DEBUG_IMAGES is True)
        if utils.DEBUG_IMAGES:
            # Save individual color masks
            cv2.imwrite(f'{health_type}_health_red_mask.png', red_mask)
            cv2.imwrite(f'{health_type}_health_green_mask.png', green_mask)
            cv2.imwrite(f'{health_type}_health_white_mask.png', white_mask)
            cv2.imwrite(f'{health_type}_health_combined_mask.png', combined_mask)
            print(f"Saved: {health_type}_health_red_mask.png")
            print(f"Saved: {health_type}_health_green_mask.png")
            print(f"Saved: {health_type}_health_white_mask.png")
            print(f"Saved: {health_type}_health_combined_mask.png")
            
            # Save visualization with contours
            cv2.imwrite(f'{health_type}_health_color_contours.png', visualization)
            print(f"Saved: {health_type}_health_color_contours.png")
            
            # Create comprehensive comparison: original | red | green | white | combined | contours
            comparison = np.zeros((health_image.shape[0], health_image.shape[1] * 6, 3), dtype=np.uint8)
            comparison[:, :health_image.shape[1]] = health_image
            comparison[:, health_image.shape[1]:health_image.shape[1]*2] = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*2:health_image.shape[1]*3] = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*3:health_image.shape[1]*4] = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*4:health_image.shape[1]*5] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*5:] = visualization
            
            cv2.imwrite(f'{health_type}_health_color_analysis.png', comparison)
            print(f"Saved: {health_type}_health_color_analysis.png")
        else:
            print("Debug images disabled (utils.DEBUG_IMAGES = False)")
        
        return combined_mask, color_boxes, color_analysis
    
    def analyze_detected_colors(self, health_image, color_boxes, red_mask, green_mask, white_mask):
        """Analyze which specific colors were detected in each region"""
        color_analysis = []
        
        for i, (x, y, w, h) in enumerate(color_boxes):
            # Extract the region
            region_red = red_mask[y:y+h, x:x+w]
            region_green = green_mask[y:y+h, x:x+w]
            region_white = white_mask[y:y+h, x:x+w]
            
            # Count pixels of each color in this region
            red_pixels = np.sum(region_red == 255)
            green_pixels = np.sum(region_green == 255)
            white_pixels = np.sum(region_white == 255)
            total_region_pixels = w * h
            
            # Determine dominant color(s)
            colors_present = []
            if red_pixels > 0:
                colors_present.append(f"red({red_pixels}px)")
            if green_pixels > 0:
                colors_present.append(f"green({green_pixels}px)")
            if white_pixels > 0:
                colors_present.append(f"white({white_pixels}px)")
            
            region_analysis = {
                'region_id': i + 1,
                'position': (x, y, w, h),
                'red_pixels': red_pixels,
                'green_pixels': green_pixels,
                'white_pixels': white_pixels,
                'total_pixels': total_region_pixels,
                'colors_present': colors_present
            }
            color_analysis.append(region_analysis)
            
            print(f"  Region {i+1}: {', '.join(colors_present) if colors_present else 'no specific colors'}")
        
        return color_analysis
    
    def analyze_both_health_regions(self):
        """Analyze both enemy and ally health regions"""
        print("Hero Health Color Mask Analysis")
        print("=" * 50)
        print("Target colors:")
        print("  Red: RGB(234,52,35) -> BGR(35,52,234)")
        print("  Green: RGB(117,251,76) -> BGR(76,251,117)")
        print("  White: RGB(255,255,255) -> BGR(255,255,255)")
        print()
        
        results = {}
        
        for health_type in ['enemy', 'ally']:
            filename = f'images/preprocess_{health_type}_health.png'
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found! Skipping {health_type} health analysis.")
                continue
            
            print(f"\n{'='*20} {health_type.upper()} HEALTH {'='*20}")
            try:
                combined_mask, color_boxes, color_analysis = self.analyze_health_colors(health_type)
                results[health_type] = {
                    'mask': combined_mask,
                    'boxes': color_boxes,
                    'analysis': color_analysis,
                    'success': True
                }
                
                if len(color_boxes) >= 1:
                    print(f"🎯 Found {len(color_boxes)} colored regions in {health_type} health area")
                else:
                    print(f"⚠️  No colored regions found in {health_type} health - might need to adjust color tolerance")
                    
            except Exception as e:
                print(f"✗ Error analyzing {health_type} health: {e}")
                results[health_type] = {'success': False, 'error': str(e)}
        
        return results

def main():
    """Test color mask detection on hero health regions"""
    print("Hearthstone Hero Health Reader - Color Mask Detection")
    print("=" * 60)
    
    # Check if health region files exist
    missing_files = []
    for health_type in ['enemy', 'ally']:
        filename = f'images/preprocess_{health_type}_health.png'
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        print("Error: Missing health region files:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please run hearthstone_regions.py first to generate these files.")
        return
    
    try:
        # Create health reader
        reader = HeroHealthReader()
        
        # Analyze both health regions
        results = reader.analyze_both_health_regions()
        
        # Print summary
        print(f"\n{'='*20} SUMMARY {'='*20}")
        for health_type, result in results.items():
            if result.get('success'):
                color_count = len(result['boxes'])
                print(f"✓ {health_type.capitalize()} health: {color_count} colored regions detected")
                
                # Show color breakdown
                if result['analysis']:
                    color_summary = {}
                    for region in result['analysis']:
                        for color_info in region['colors_present']:
                            color_name = color_info.split('(')[0]  # Extract color name
                            color_summary[color_name] = color_summary.get(color_name, 0) + 1
                    
                    if color_summary:
                        colors_found = ', '.join([f"{count} {color}" for color, count in color_summary.items()])
                        print(f"  Colors detected: {colors_found}")
            else:
                print(f"✗ {health_type.capitalize()} health: Analysis failed")
        
        if utils.DEBUG_IMAGES:
            print(f"\n✓ Check *_health_color_analysis.png files for full visualizations")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    if utils.DEBUG_IMAGES:
        print("\nGenerated files (per health type):")
        print("  - {type}_health_red_mask.png: Red color mask only")
        print("  - {type}_health_green_mask.png: Green color mask only")
        print("  - {type}_health_white_mask.png: White color mask only")
        print("  - {type}_health_combined_mask.png: Combined red + green + white mask")
        print("  - {type}_health_color_contours.png: Contours + bounding boxes")
        print("  - {type}_health_color_analysis.png: Original | Red | Green | White | Combined | Contours")
    else:
        print("\nNote: Debug images disabled (utils.DEBUG_IMAGES = False)")
        print("      Set utils.DEBUG_IMAGES = True to generate debug PNG files")
    
    print("\nTo adjust sensitivity:")
    print("  - Increase color_tolerance (currently 30) to detect more color variations")
    print("  - Decrease color_tolerance to be more strict with exact color matching")
    print("  - Adjust white_threshold (currently 240) for white detection sensitivity")
    print("\nTo adjust size restrictions:")
    print("  - min_area/max_area: Control minimum and maximum region sizes")
    print("  - min_width/min_height: Control minimum region dimensions")

if __name__ == "__main__":
    main()