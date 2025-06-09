import cv2
import numpy as np
import os
import logging
import utils

class HeroHealthReader:
    """Class for reading hero health using color mask detection on enemy and ally health regions"""
    
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger('HeroHealthReader')
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
        
        self.logger.info("HeroHealthReader initialized")
    
    def load_health_image(self, health_type='enemy'):
        """Load enemy or ally health image file
        Args:
            health_type: 'enemy' or 'ally'
        """
        filename = f'images/preprocess_{health_type}_health.png'
        if not os.path.exists(filename):
            self.logger.error(f"{filename} not found! Please run hearthstone_regions.py first.")
            raise FileNotFoundError(f"{filename} not found! Please run hearthstone_regions.py first.")
        
        health_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if health_image is None:
            self.logger.error(f"Could not load {filename}")
            raise ValueError(f"Could not load {filename}")
        
        self.logger.debug(f"Loaded {health_type} health image with shape: {health_image.shape}")
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
        
        self.logger.debug(f"{color_name.capitalize()} color detection:")
        self.logger.debug(f"  Target BGR: {target_color}")
        self.logger.debug(f"  Lower bound: {lower_bound}")
        self.logger.debug(f"  Upper bound: {upper_bound}")
        
        # Create mask for this color range
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # Count pixels found
        color_pixels = np.sum(mask == 255)
        self.logger.debug(f"  Found {color_pixels} {color_name} pixels")
        
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
        self.logger.debug("White color detection:")
        self.logger.debug(f"  Threshold: {self.white_threshold}")
        self.logger.debug(f"  Found {white_pixels} white pixels")
        
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
        
        self.logger.debug(f"Image dimensions: {image_width}x{image_height}")
        self.logger.debug("Size restrictions:")
        self.logger.debug(f"  Min area: {self.size_restrictions['min_area']}px²")
        self.logger.debug(f"  Max area: {self.size_restrictions['max_area']}px²")
        self.logger.debug(f"  Min width: {self.size_restrictions['min_width']}px")
        self.logger.debug(f"  Min height: {self.size_restrictions['min_height']}px")
        
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
                self.logger.debug(f"✓ Color region {len(color_boxes)}: pos=({x},{y}), size=({w}x{h}), area={area}px²")
            else:
                rejected_regions.append({
                    'position': (x, y, w, h),
                    'area': area,
                    'reasons': rejection_reasons
                })
                self.logger.debug(f"✗ Rejected: pos=({x},{y}), size=({w}x{h}) - {', '.join(rejection_reasons)}")
        
        self.logger.info(f"Summary: {len(color_boxes)} valid color regions, {len(rejected_regions)} rejected")
        return color_boxes, valid_contours
    
    def analyze_health_colors(self, health_type='enemy'):
        """Analyze color mask from health image and save results
        Args:
            health_type: 'enemy' or 'ally'
        """
        self.logger.info(f"Loading images/preprocess_{health_type}_health.png...")
        
        # Load health image
        health_image = self.load_health_image(health_type)
        self.logger.info(f"Health image shape: {health_image.shape}")
        
        # Create color masks
        self.logger.debug(f"Detecting colors with tolerance ±{self.color_tolerance}:")
        combined_mask, red_mask, green_mask, white_mask = self.detect_color_regions(health_image)
        
        # Calculate statistics
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        color_pixels = np.sum(combined_mask == 255)
        color_percentage = (color_pixels / total_pixels) * 100
        
        self.logger.debug("Color mask statistics:")
        self.logger.debug(f"  Total pixels: {total_pixels}")
        self.logger.debug(f"  Color pixels found: {color_pixels}")
        self.logger.debug(f"  Color percentage: {color_percentage:.2f}%")
        
        # Extract individual color regions using contours
        color_boxes, valid_contours = self.extract_color_regions(combined_mask)
        self.logger.info(f"Found {len(color_boxes)} color regions using contour detection")
        
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
            self.logger.debug(f"Saved: {health_type}_health_red_mask.png")
            self.logger.debug(f"Saved: {health_type}_health_green_mask.png")
            self.logger.debug(f"Saved: {health_type}_health_white_mask.png")
            self.logger.debug(f"Saved: {health_type}_health_combined_mask.png")
            
            # Save visualization with contours
            cv2.imwrite(f'{health_type}_health_color_contours.png', visualization)
            self.logger.debug(f"Saved: {health_type}_health_color_contours.png")
            
            # Create comprehensive comparison: original | red | green | white | combined | contours
            comparison = np.zeros((health_image.shape[0], health_image.shape[1] * 6, 3), dtype=np.uint8)
            comparison[:, :health_image.shape[1]] = health_image
            comparison[:, health_image.shape[1]:health_image.shape[1]*2] = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*2:health_image.shape[1]*3] = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*3:health_image.shape[1]*4] = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*4:health_image.shape[1]*5] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, health_image.shape[1]*5:] = visualization
            
            cv2.imwrite(f'{health_type}_health_color_analysis.png', comparison)
            self.logger.debug(f"Saved: {health_type}_health_color_analysis.png")
        else:
            self.logger.debug("Debug images disabled (utils.DEBUG_IMAGES = False)")
        
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
            
            self.logger.debug(f"Region {i+1}: {', '.join(colors_present) if colors_present else 'no specific colors'}")
        
        return color_analysis
    
    def analyze_both_health_regions(self):
        """Analyze both enemy and ally health regions"""
        self.logger.info("Hero Health Color Mask Analysis")
        self.logger.info("Target colors:")
        self.logger.info("  Red: RGB(234,52,35) -> BGR(35,52,234)")
        self.logger.info("  Green: RGB(117,251,76) -> BGR(76,251,117)")
        self.logger.info("  White: RGB(255,255,255) -> BGR(255,255,255)")
        
        results = {}
        
        for health_type in ['enemy', 'ally']:
            filename = f'images/preprocess_{health_type}_health.png'
            if not os.path.exists(filename):
                self.logger.warning(f"{filename} not found! Skipping {health_type} health analysis.")
                continue
            
            self.logger.info(f"Starting {health_type.upper()} HEALTH analysis")
            try:
                combined_mask, color_boxes, color_analysis = self.analyze_health_colors(health_type)
                results[health_type] = {
                    'mask': combined_mask,
                    'boxes': color_boxes,
                    'analysis': color_analysis,
                    'success': True
                }
                
                if len(color_boxes) >= 1:
                    self.logger.info(f"Found {len(color_boxes)} colored regions in {health_type} health area")
                else:
                    self.logger.warning(f"No colored regions found in {health_type} health - might need to adjust color tolerance")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {health_type} health: {e}")
                results[health_type] = {'success': False, 'error': str(e)}
        
        return results

def main():
    """Test color mask detection on hero health regions"""
    # Set up logging for main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('HeroHealthReaderMain')
    
    logger.info("Hearthstone Hero Health Reader - Color Mask Detection")
    
    # Check if health region files exist
    missing_files = []
    for health_type in ['enemy', 'ally']:
        filename = f'images/preprocess_{health_type}_health.png'
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        logger.error("Missing health region files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.error("Please run hearthstone_regions.py first to generate these files.")
        return
    
    try:
        # Create health reader
        reader = HeroHealthReader()
        
        # Analyze both health regions
        results = reader.analyze_both_health_regions()
        
        # Print summary to console
        print("SUMMARY:")
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
            print("✓ Check *_health_color_analysis.png files for full visualizations")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    if utils.DEBUG_IMAGES:
        print("Generated files (per health type):")
        print("  - {type}_health_red_mask.png: Red color mask only")
        print("  - {type}_health_green_mask.png: Green color mask only")
        print("  - {type}_health_white_mask.png: White color mask only")
        print("  - {type}_health_combined_mask.png: Combined red + green + white mask")
        print("  - {type}_health_color_contours.png: Contours + bounding boxes")
        print("  - {type}_health_color_analysis.png: Original | Red | Green | White | Combined | Contours")
    else:
        print("Note: Debug images disabled (utils.DEBUG_IMAGES = False)")
    
    print("To adjust sensitivity:")
    print("  - Increase color_tolerance (currently 30) to detect more color variations")
    print("  - Decrease color_tolerance to be more strict with exact color matching")
    print("  - Adjust white_threshold (currently 240) for white detection sensitivity")
    print("To adjust size restrictions:")
    print("  - min_area/max_area: Control minimum and maximum region sizes")
    print("  - min_width/min_height: Control minimum region dimensions")

if __name__ == "__main__":
    main()