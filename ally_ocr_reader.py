import cv2
import numpy as np
import pytesseract
import os
import re
import logging
import utils

class AllyBoardReader:
    """Class specifically for reading cards from test_ally_board.png using color mask detection"""
    
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger('AllyBoardReader')
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
        self.color_tolerance = 20  # Adjust this value to be more/less strict with color matching
        self.white_threshold = 250  # Threshold for white detection (240-255)
        
        # Height restrictions for valid crystals
        self.height_restrictions = {
            'min_top_border': 10,    # Minimum pixels from top of image to crystal top
            'min_bottom_border': 10, # Minimum pixels from crystal bottom to bottom of image
            'min_height': 8,         # Minimum height of crystal itself
            'max_height': 100        # Maximum height of crystal itself
        }
        
        # NEW: Number merging parameters
        self.number_merging = {
            'enabled': True,                    # Enable/disable number merging
            'max_horizontal_distance': 25,     # Maximum horizontal pixel distance between numbers to merge
            'max_vertical_distance': 10,       # Maximum vertical pixel distance between numbers to merge
            'min_overlap_ratio': 0.3,          # Minimum vertical overlap ratio (0.0-1.0) for merging
        }
        
        # OCR PARAMETERS for ally board numbers
        self.ocr_params = {
            'white_threshold': 250,      # Threshold for white text (200-255)
            'scale_factor': 10,          # Upscaling factor for better OCR
            'gaussian_blur': (7, 7),     # Gaussian blur kernel size
            'gaussian_sigma': 0.5,       # Gaussian blur sigma
            'morph_kernel_size': 1,      # Morphological operations kernel size
            'padding': 0,               # Padding around crystals
            'rotation_angles': [0, -2, 2, -5, 5],  # Try 0° first, then small angles
            
            # CONFIDENCE PARAMETERS
            'high_confidence_early_stop': 85,  # Stop trying other methods if confidence >= this value
            'low_confidence_skip_angle': 20,   # Skip remaining PSMs for angles with max confidence below this value
        }
        
        # OCR configurations for numbers (attack/health values can be 1-30+ typically)
        self.ocr_configs = [
            '--psm 8 -c tessedit_char_whitelist=0123456789',   # Single character
            '--psm 7 -c tessedit_char_whitelist=0123456789',   # Single text line
            '--psm 10 -c tessedit_char_whitelist=0123456789',  # Single character, no layout analysis
            '--psm 6 -c tessedit_char_whitelist=0123456789',   # Single uniform block
            '--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789',  # LSTM only
        ]
        
        self.logger.info("AllyBoardReader initialized with number merging capability and OCR")
    
    def load_ally_board_image(self):
        """Load test_ally_board.png file"""
        if not os.path.exists('images/preprocess_ally_board.png'):
            self.logger.error("images/preprocess_ally_board.png not found! Please run hearthstone_regions.py first.")
            raise FileNotFoundError("images/preprocess_ally_board.png not found! Please run hearthstone_regions.py first.")
        
        ally_board_image = cv2.imread('images/preprocess_ally_board.png', cv2.IMREAD_COLOR)
        if ally_board_image is None:
            self.logger.error("Could not load images/preprocess_ally_board.png")
            raise ValueError("Could not load images/preprocess_ally_board.png")
        
        self.logger.debug(f"Loaded ally board image with shape: {ally_board_image.shape}")
        return ally_board_image
    
    def detect_color_regions(self, ally_board_image):
        """Create color mask for specific red, green, and white colors"""
        # Create masks for each target color
        red_mask = self.create_color_mask(ally_board_image, self.target_colors['red'], 'red')
        green_mask = self.create_color_mask(ally_board_image, self.target_colors['green'], 'green')
        white_mask = self.create_white_mask(ally_board_image)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, white_mask)
        
        return combined_mask, red_mask, green_mask, white_mask
    
    def create_color_mask(self, image, target_color, color_name):
        """Create mask for a specific color with tolerance"""
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
        """Create mask for white/bright areas using threshold method"""
        # Convert to grayscale for white detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for white pixels
        _, white_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        # Count pixels found
        white_pixels = np.sum(white_mask == 255)
        self.logger.debug(f"White color detection:")
        self.logger.debug(f"  Threshold: {self.white_threshold}")
        self.logger.debug(f"  Found {white_pixels} white pixels")
        
        return white_mask
    
    def calculate_overlap_ratio(self, box1, box2):
        """Calculate vertical overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate vertical overlap
        top1, bottom1 = y1, y1 + h1
        top2, bottom2 = y2, y2 + h2
        
        overlap_top = max(top1, top2)
        overlap_bottom = min(bottom1, bottom2)
        
        if overlap_bottom <= overlap_top:
            return 0.0  # No overlap
        
        overlap_height = overlap_bottom - overlap_top
        min_height = min(h1, h2)
        
        return overlap_height / min_height if min_height > 0 else 0.0
    
    def should_merge_numbers(self, box1, box2):
        """Determine if two number detections should be merged"""
        if not self.number_merging['enabled']:
            return False
        
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate horizontal distance between boxes
        if x1 < x2:  # box1 is to the left of box2
            horizontal_distance = x2 - (x1 + w1)
        else:  # box2 is to the left of box1
            horizontal_distance = x1 - (x2 + w2)
        
        # Calculate vertical distance between boxes (center to center)
        center_y1 = y1 + h1 / 2
        center_y2 = y2 + h2 / 2
        vertical_distance = abs(center_y1 - center_y2)
        
        # Calculate vertical overlap ratio
        overlap_ratio = self.calculate_overlap_ratio(box1, box2)
        
        # Check all merge criteria
        horizontal_close = horizontal_distance <= self.number_merging['max_horizontal_distance']
        vertical_close = vertical_distance <= self.number_merging['max_vertical_distance']
        sufficient_overlap = overlap_ratio >= self.number_merging['min_overlap_ratio']
        
        return horizontal_close and vertical_close and sufficient_overlap
    
    def merge_number_boxes(self, boxes_to_merge):
        """Merge multiple bounding boxes into one"""
        if not boxes_to_merge:
            return None
        
        if len(boxes_to_merge) == 1:
            return boxes_to_merge[0]
        
        # Find bounding rectangle that contains all boxes
        min_x = min(box[0] for box in boxes_to_merge)
        min_y = min(box[1] for box in boxes_to_merge)
        max_x = max(box[0] + box[2] for box in boxes_to_merge)
        max_y = max(box[1] + box[3] for box in boxes_to_merge)
        
        merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        self.logger.debug(f"Merged {len(boxes_to_merge)} boxes into {merged_box}")
        return merged_box
    
    def group_nearby_numbers(self, crystal_boxes):
        """Group nearby number detections that should be merged"""
        if not self.number_merging['enabled'] or len(crystal_boxes) <= 1:
            return crystal_boxes
        
        self.logger.debug(f"Starting number merging process with {len(crystal_boxes)} initial detections")
        
        # Create groups of boxes that should be merged
        groups = []
        used_indices = set()
        
        for i, box1 in enumerate(crystal_boxes):
            if i in used_indices:
                continue
            
            # Start a new group with this box
            current_group = [box1]
            group_indices = {i}
            
            # Check all other boxes to see if they should be merged with this group
            for j, box2 in enumerate(crystal_boxes):
                if j in used_indices or j == i:
                    continue
                
                # Check if box2 should be merged with any box in current group
                should_merge_with_group = False
                for group_box in current_group:
                    if self.should_merge_numbers(group_box, box2):
                        should_merge_with_group = True
                        break
                
                if should_merge_with_group:
                    current_group.append(box2)
                    group_indices.add(j)
            
            # Mark all boxes in this group as used
            used_indices.update(group_indices)
            groups.append(current_group)
        
        # Merge boxes within each group
        merged_boxes = []
        for group in groups:
            if len(group) > 1:
                self.logger.debug(f"Merging group of {len(group)} boxes: {group}")
            merged_box = self.merge_number_boxes(group)
            if merged_box:
                merged_boxes.append(merged_box)
        
        self.logger.info(f"Number merging: {len(crystal_boxes)} → {len(merged_boxes)} detections")
        
        return merged_boxes
    
    def extract_individual_crystals(self, combined_mask):
        """Extract individual crystal contours from color mask with height restrictions"""
        # Get image dimensions for border calculations
        image_height, image_width = combined_mask.shape
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crystal_boxes = []
        valid_contours = []
        rejected_crystals = []
        
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
        
        # Apply number merging if enabled
        if self.number_merging['enabled'] and len(crystal_boxes) > 1:
            original_count = len(crystal_boxes)
            crystal_boxes = self.group_nearby_numbers(crystal_boxes)
            
            if len(crystal_boxes) != original_count:
                self.logger.info(f"Number merging applied: {original_count} → {len(crystal_boxes)} final detections")
                valid_contours = []  # Clear contours since they no longer match the merged boxes
        
        self.logger.info(f"Summary: {len(crystal_boxes)} valid crystals, {len(rejected_crystals)} rejected")
        
        # Sort crystals from left to right based on x-coordinate
        if crystal_boxes:
            crystal_boxes.sort(key=lambda box: box[0])  # Sort by x-coordinate (leftmost first)
        
        return crystal_boxes, valid_contours
    
    def extract_crystal_with_padding(self, ally_board_image, x, y, w, h):
        """Extract crystal region with padding around the borders"""
        height, width = ally_board_image.shape[:2]
        padding = self.ocr_params['padding']
        
        # Add padding but keep within image bounds
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        # Extract padded region
        padded_crystal = ally_board_image[y_start:y_end, x_start:x_end]
        
        self.logger.debug(f"Extracted crystal region with padding: {padding}px")
        return padded_crystal
    
    def add_padding_to_image(self, image, padding_pixels=30):
        """Add white padding around the image to help OCR"""
        height, width = image.shape
        
        # Create new image with padding
        padded_height = height + (2 * padding_pixels)
        padded_width = width + (2 * padding_pixels)
        
        # For ally board numbers, padding should be white (255) for inverted version
        padded_image = np.full((padded_height, padded_width), 255, dtype=np.uint8)
        
        # Place original image in center
        padded_image[padding_pixels:padding_pixels + height, 
                    padding_pixels:padding_pixels + width] = image
        
        return padded_image
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle (degrees) with padding to avoid cropping"""
        if angle == 0:
            return image
        
        height, width = image.shape[:2]
        
        # Calculate the center of the image
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to avoid cropping
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation with padding
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
        
        return rotated
    
    def preprocess_for_ocr(self, crystal_image):
        """Enhance crystal image for OCR with color detection and processing"""
        # Convert to grayscale if needed
        if len(crystal_image.shape) == 3:
            original_bgr = crystal_image.copy()
            gray = cv2.cvtColor(crystal_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crystal_image
            original_bgr = None
        
        final_mask = None
        method_used = "unknown"
        
        if original_bgr is not None:
            # METHOD 1: Use the same color detection as the color mask
            # Create masks for each color with slightly higher tolerance for OCR
            color_tolerance = 40
            red_mask = self.create_color_mask_for_ocr(original_bgr, self.target_colors['red'], color_tolerance)
            green_mask = self.create_color_mask_for_ocr(original_bgr, self.target_colors['green'], color_tolerance)
            white_mask = self.create_color_mask_for_ocr(original_bgr, self.target_colors['white'], color_tolerance)
            
            # Also try white detection with threshold method
            gray_for_white = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
            _, white_thresh_mask = cv2.threshold(gray_for_white, 200, 255, cv2.THRESH_BINARY)
            
            # Combine all color masks
            combined_mask = cv2.bitwise_or(red_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, white_mask)
            combined_mask = cv2.bitwise_or(combined_mask, white_thresh_mask)
            
            total_colored_pixels = np.sum(combined_mask == 255)
            
            if total_colored_pixels > 10:  # Found some colored pixels
                final_mask = combined_mask
                method_used = "color_detection"
            else:
                self.logger.debug("No colored pixels found, falling back to grayscale")
        
        # METHOD 2: Fallback to grayscale if color detection failed
        if final_mask is None:
            # Try multiple thresholds to find the best one
            thresholds_to_try = [150, 180, 200, 220, 240]
            best_mask = None
            best_pixel_count = 0
            
            for thresh in thresholds_to_try:
                _, test_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                pixel_count = np.sum(test_mask == 255)
                
                # We want a reasonable amount of pixels
                if 20 < pixel_count < (gray.shape[0] * gray.shape[1] * 0.8):
                    if pixel_count > best_pixel_count:
                        best_mask = test_mask
                        best_pixel_count = pixel_count
            
            if best_mask is not None:
                final_mask = best_mask
                method_used = "grayscale_threshold"
            else:
                # Last resort - use adaptive threshold
                final_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                method_used = "adaptive_threshold"
        
        # Clean up and scale the mask
        if final_mask is not None:
            # Apply light morphological operations
            kernel_size = max(1, self.ocr_params['morph_kernel_size'])
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            # Scale up for better OCR
            scale_factor = self.ocr_params['scale_factor']
            height, width = final_mask.shape
            scaled = cv2.resize(final_mask, (width * scale_factor, height * scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
            
            # Re-threshold after scaling
            _, scaled_clean = cv2.threshold(scaled, 128, 255, cv2.THRESH_BINARY)
        else:
            # Fallback - create empty mask
            scaled_clean = np.zeros((100, 100), dtype=np.uint8)
            method_used = "failed"
        
        # Create inverted version (black text on white background for OCR)
        scaled_clean_inv = cv2.bitwise_not(scaled_clean)
        
        # Add padding to help OCR recognition
        padded_inv = self.add_padding_to_image(scaled_clean_inv, padding_pixels=30)
        
        self.logger.debug(f"Processing method: {method_used}")
        
        return scaled_clean, padded_inv
    
    def create_color_mask_for_ocr(self, image, target_color, tolerance):
        """Create mask for a specific color with tolerance - for OCR preprocessing"""
        lower_bound = np.clip(target_color - tolerance, 0, 255)
        upper_bound = np.clip(target_color + tolerance, 0, 255)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask
    
    def ocr_crystal(self, crystal_image, crystal_id):
        """Apply OCR to a single crystal image"""
        self.logger.debug(f"Starting OCR on ally crystal {crystal_id}")
        
        all_results = []
        high_confidence_threshold = self.ocr_params['high_confidence_early_stop']
        low_confidence_skip_threshold = self.ocr_params['low_confidence_skip_angle']
        
        # Try different rotation angles
        for angle in self.ocr_params['rotation_angles']:
            self.logger.debug(f"Trying rotation: {angle}° for ally crystal {crystal_id}")
            
            # Rotate the crystal image
            rotated_crystal = self.rotate_image(crystal_image, angle)
            
            # Get processed versions
            processed_mask, processed_mask_inv = self.preprocess_for_ocr(rotated_crystal)
            
            angle_max_confidence = 0
            angle_skip_remaining = False
            
            # Try each PSM mode
            for config_name, config in zip(["PSM8", "PSM7", "PSM10", "PSM6", "PSM8_LSTM"], self.ocr_configs):
                if angle_skip_remaining:
                    break
                    
                try:
                    # Get OCR result with confidence
                    data = pytesseract.image_to_data(processed_mask_inv, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Extract text and confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    texts = [text for text in data['text'] if text.strip()]
                    
                    if texts and confidences:
                        # Get the text with highest confidence
                        max_conf_idx = confidences.index(max(confidences))
                        text = texts[max_conf_idx] if max_conf_idx < len(texts) else texts[0]
                        confidence = max(confidences)
                        
                        # Update angle max confidence
                        angle_max_confidence = max(angle_max_confidence, confidence)
                        
                        # Clean up text - keep only digits
                        clean_text = re.sub(r'[^0-9]', '', text)
                        
                        if clean_text and clean_text.isdigit():
                            number = int(clean_text)
                            if 0 <= number <= 99:  # Valid range for attack/health
                                result = {
                                    'number': number,
                                    'confidence': confidence,
                                    'angle': angle,
                                    'method': f"processed_mask_inv_{config_name}",
                                    'raw_text': text,
                                    'clean_text': clean_text,
                                    'image': processed_mask_inv.copy(),
                                    'rotated_original': rotated_crystal.copy()
                                }
                                all_results.append(result)
                                self.logger.debug(f"Ally crystal {crystal_id} - {config_name} at {angle}°: '{number}' (confidence: {confidence}%)")
                                
                                # EARLY STOP: If we hit high confidence
                                if confidence >= high_confidence_threshold:
                                    self.logger.debug(f"Ally crystal {crystal_id} - High confidence ({confidence}%) reached!")
                                    angle_skip_remaining = True
                                    break
                                    
                            else:
                                self.logger.debug(f"Ally crystal {crystal_id} - {config_name} at {angle}°: '{clean_text}' (out of range, conf: {confidence}%)")
                        else:
                            if text.strip():
                                self.logger.debug(f"Ally crystal {crystal_id} - {config_name} at {angle}°: '{text}' (not number, conf: {confidence}%)")
                    
                    # LOW CONFIDENCE SKIP
                    if angle_max_confidence > 0 and angle_max_confidence < low_confidence_skip_threshold:
                        self.logger.debug(f"Ally crystal {crystal_id} - Angle {angle}° max confidence ({angle_max_confidence}%) below threshold, skipping remaining PSMs")
                        angle_skip_remaining = True
                        break
                    
                except Exception as e:
                    self.logger.error(f"Ally crystal {crystal_id} - {config_name} at {angle}°: Error - {e}")
            
            # EARLY STOP: If we found a very high confidence result
            if all_results and max(r['confidence'] for r in all_results) >= high_confidence_threshold:
                self.logger.debug(f"Ally crystal {crystal_id} - Very high confidence found at {angle}°! Stopping early")
                break
        
        # Analyze all results and pick the best one
        if all_results:
            # Sort by confidence (highest first)
            all_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            best_result = all_results[0]
            
            # Save the best result for debugging (only if DEBUG_IMAGES is True)
            if utils.DEBUG_IMAGES:
                cv2.imwrite(f'ally_crystal_{crystal_id}_BEST_{best_result["angle"]}deg_{best_result["method"]}.png', best_result['image'])
                cv2.imwrite(f'ally_crystal_{crystal_id}_BEST_{best_result["angle"]}deg_original.png', best_result['rotated_original'])
            
            return best_result['number'], best_result['clean_text'], f"{best_result['angle']}deg_{best_result['method']}_conf{best_result['confidence']}"
        
        self.logger.warning(f"Ally crystal {crystal_id} - No valid number found at any rotation angle")
        return None, "", "failed"
    
    def analyze_all_crystals(self):
        """Analyze all detected crystals and apply OCR to each - NEW METHOD"""
        self.logger.info("Starting ally board OCR analysis")
        
        # Get color mask and crystal locations
        try:
            combined_mask, crystal_boxes = self.analyze_color_mask()
        except Exception as e:
            self.logger.error(f"Error getting crystals from color mask: {e}")
            return []
        
        if not crystal_boxes:
            self.logger.warning("No crystals detected! Check ally board color mask output first.")
            return []
        
        # Load original ally board image
        ally_board_image = self.load_ally_board_image()
        
        self.logger.info(f"Applying OCR to {len(crystal_boxes)} detected ally crystals")
        
        ocr_results = []
        crystal_images = []
        
        # Process each detected crystal
        for i, (x, y, w, h) in enumerate(crystal_boxes):
            crystal_id = i + 1
            self.logger.debug(f"Processing ally crystal {crystal_id}: position=({x},{y}), size=({w}x{h})")
            
            # Extract crystal region WITH PADDING for better border capture
            crystal_region = self.extract_crystal_with_padding(ally_board_image, x, y, w, h)
            crystal_images.append(crystal_region)
            
            # Apply OCR
            number, raw_text, method = self.ocr_crystal(crystal_region, crystal_id)
            
            result = {
                'id': crystal_id,
                'position': (x, y, w, h),
                'number': number,
                'raw_text': raw_text,
                'method': method,
                'success': number is not None
            }
            ocr_results.append(result)
            
            # Save individual crystal image for debugging (only if DEBUG_IMAGES is True)
            if utils.DEBUG_IMAGES:
                cv2.imwrite(f'ally_crystal_{crystal_id}.png', crystal_region)
                
                # Save processed versions for debugging
                processed_mask, processed_mask_inv = self.preprocess_for_ocr(crystal_region)
                cv2.imwrite(f'ally_crystal_{crystal_id}_processed_mask.png', processed_mask)
                cv2.imwrite(f'ally_crystal_{crystal_id}_processed_mask_inv.png', processed_mask_inv)
        
        # Create summary visualization
        self.create_ally_ocr_visualization(ally_board_image, crystal_boxes, ocr_results)
        
        # Log summary
        self.log_ally_ocr_summary(ocr_results)
        
        return ocr_results
    
    def create_ally_ocr_visualization(self, ally_board_image, crystal_boxes, ocr_results):
        """Create visualization showing OCR results on the ally board image"""
        visualization = ally_board_image.copy()
        
        for i, ((x, y, w, h), result) in enumerate(zip(crystal_boxes, ocr_results)):
            # Draw bounding box
            color = (0, 255, 0) if result['success'] else (0, 0, 255)  # Green if success, red if failed
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
            
            # Add OCR result text - SHOW THE ACTUAL NUMBER
            if result['success']:
                text = f"{result['number']}"  # Show the extracted number
                text_color = (0, 255, 0)
                # Add background rectangle for better visibility
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                cv2.rectangle(visualization, (x, y-35), (x + text_size[0] + 10, y-5), (0, 0, 0), -1)
            else:
                text = "?"
                text_color = (0, 0, 255)
                # Add background rectangle
                cv2.rectangle(visualization, (x, y-35), (x + 20, y-5), (0, 0, 0), -1)
            
            # Put larger text above the crystal
            cv2.putText(visualization, text, (x+5, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            
            # Add crystal ID below
            cv2.putText(visualization, f"#{result['id']}", (x, y+h+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save visualization (only if DEBUG_IMAGES is True)
        if utils.DEBUG_IMAGES:
            cv2.imwrite('ally_board_ocr_results.png', visualization)
            self.logger.debug("Saved: ally_board_ocr_results.png")
    
    def log_ally_ocr_summary(self, ocr_results):
        """Log summary of OCR results"""
        self.logger.info("Ally Board OCR Summary:")
        
        successful = [r for r in ocr_results if r['success']]
        failed = [r for r in ocr_results if not r['success']]
        
        self.logger.info(f"Successfully read: {len(successful)}/{len(ocr_results)} crystals")
        
        if successful:
            self.logger.info("Detected ally board numbers:")
            for result in successful:
                self.logger.info(f"  Crystal {result['id']}: {result['number']} (method: {result['method']})")
        
        if failed:
            self.logger.info("Failed to read:")
            for result in failed:
                self.logger.info(f"  Crystal {result['id']}: position {result['position'][:2]}")
    
    def analyze_color_mask(self):
        """Analyze color mask from images/preprocess_ally_board.png and save results"""
        self.logger.info("Loading images/preprocess_ally_board.png...")
        
        # Load ally board image
        ally_board_image = self.load_ally_board_image()
        self.logger.info(f"Ally board image shape: {ally_board_image.shape}")
        
        # Create color masks
        self.logger.debug(f"Detecting colors with tolerance ±{self.color_tolerance}:")
        combined_mask, red_mask, green_mask, white_mask = self.detect_color_regions(ally_board_image)
        
        # Calculate statistics
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        color_pixels = np.sum(combined_mask == 255)
        color_percentage = (color_pixels / total_pixels) * 100
        
        self.logger.debug("Color mask statistics:")
        self.logger.debug(f"  Total pixels: {total_pixels}")
        self.logger.debug(f"  Color pixels found: {color_pixels}")
        self.logger.debug(f"  Color percentage: {color_percentage:.2f}%")
        
        # Extract individual crystals using contours (with number merging)
        crystal_boxes, valid_contours = self.extract_individual_crystals(combined_mask)
        self.logger.info(f"Found {len(crystal_boxes)} crystals using contour detection")
        
        # Create visualization with contours and bounding boxes
        visualization = ally_board_image.copy()
        
        # Draw contours in yellow (visible against both red and green) - only if we have valid contours
        if valid_contours:
            cv2.drawContours(visualization, valid_contours, -1, (0, 255, 255), 2)
        
        # Draw bounding boxes in blue
        for i, (x, y, w, h) in enumerate(crystal_boxes):
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Add crystal number
            cv2.putText(visualization, str(i+1), (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save individual color masks and visualization (only if DEBUG_IMAGES is True)
        if utils.DEBUG_IMAGES:
            cv2.imwrite('ally_board_red_mask.png', red_mask)
            cv2.imwrite('ally_board_green_mask.png', green_mask)
            cv2.imwrite('ally_board_white_mask.png', white_mask)
            cv2.imwrite('ally_board_combined_mask.png', combined_mask)
            self.logger.debug("Saved: ally_board_red_mask.png")
            self.logger.debug("Saved: ally_board_green_mask.png") 
            self.logger.debug("Saved: ally_board_white_mask.png")
            self.logger.debug("Saved: ally_board_combined_mask.png")
            
            # Save visualization with contours
            cv2.imwrite('ally_board_crystal_contours.png', visualization)
            self.logger.debug("Saved: ally_board_crystal_contours.png")
            
            # Create comprehensive comparison: original | red | green | white | combined | contours
            comparison = np.zeros((ally_board_image.shape[0], ally_board_image.shape[1] * 6, 3), dtype=np.uint8)
            comparison[:, :ally_board_image.shape[1]] = ally_board_image
            comparison[:, ally_board_image.shape[1]:ally_board_image.shape[1]*2] = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, ally_board_image.shape[1]*2:ally_board_image.shape[1]*3] = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, ally_board_image.shape[1]*3:ally_board_image.shape[1]*4] = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, ally_board_image.shape[1]*4:ally_board_image.shape[1]*5] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            comparison[:, ally_board_image.shape[1]*5:] = visualization
            
            cv2.imwrite('ally_board_color_analysis.png', comparison)
            self.logger.debug("Saved: ally_board_color_analysis.png")
        else:
            self.logger.debug("Debug images disabled (utils.DEBUG_IMAGES = False)")
        
        return combined_mask, crystal_boxes
    
    def analyze_white_mask(self):
        """Compatibility method - calls analyze_color_mask for backward compatibility"""
        return self.analyze_color_mask()

def main():
    """Test color mask detection with contour analysis on ally board"""
    # Set up logging for main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('AllyBoardMain')
    
    logger.info("Hearthstone Ally Board Reader - Color Mask Detection with Number Merging and OCR")
    logger.info("Target colors:")
    logger.info("  Red: RGB(234,52,35) -> BGR(35,52,234)")
    logger.info("  Green: RGB(117,251,76) -> BGR(76,251,117)")
    logger.info("  White: RGB(255,255,255) -> BGR(255,255,255)")
    
    # Check if test_ally_board.png exists
    if not os.path.exists('images/preprocess_ally_board.png'):
        logger.error("images/preprocess_ally_board.png not found!")
        logger.error("Please run hearthstone_regions.py first to generate this file.")
        return
    
    try:
        # Create ally board reader
        reader = AllyBoardReader()
        
        # Log number merging parameters
        logger.info("Number merging parameters:")
        logger.info(f"  Enabled: {reader.number_merging['enabled']}")
        logger.info(f"  Max horizontal distance: {reader.number_merging['max_horizontal_distance']}px")
        logger.info(f"  Max vertical distance: {reader.number_merging['max_vertical_distance']}px")
        logger.info(f"  Min overlap ratio: {reader.number_merging['min_overlap_ratio']}")
        
        # Analyze all crystals with OCR
        ocr_results = reader.analyze_all_crystals()
        
        if ocr_results:
            # Count successful OCR results
            numbers_found = [r['number'] for r in ocr_results if r['success']]
            if numbers_found:
                logger.info(f"Successfully extracted {len(numbers_found)} ally board numbers: {numbers_found}")
                print(f"🎯 ALLY BOARD NUMBERS: {numbers_found}")
            else:
                logger.warning("No valid numbers detected. Check individual crystal images for debugging.")
        
        if utils.DEBUG_IMAGES:
            logger.info("Check ally_board_ocr_results.png for visualization")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    if utils.DEBUG_IMAGES:
        logger.info("Generated files:")
        logger.info("  - ally_board_ocr_results.png: Visualization with OCR results")
        logger.info("  - ally_crystal_X.png: Individual crystal images (with padding)")
        logger.info("  - ally_crystal_X_processed_mask.png: Color/text isolation")
        logger.info("  - ally_crystal_X_BEST_method.png: Successful OCR version (if any)")
    else:
        logger.info("Note: Debug images disabled (utils.DEBUG_IMAGES = False)")
    
    logger.info("To adjust sensitivity:")
    logger.info("  - Increase color_tolerance (currently 30) to detect more color variations")
    logger.info("  - Decrease color_tolerance to be more strict with exact color matching")
    logger.info("  - Adjust white_threshold (currently 240) for white detection sensitivity")
    logger.info("To adjust OCR parameters:")
    logger.info("  - Lower white_threshold in ocr_params (currently 220) if numbers aren't pure white")
    logger.info("  - Increase scale_factor for larger text")
    logger.info("  - Adjust high_confidence_early_stop for performance vs accuracy trade-off")

if __name__ == "__main__":
    main()