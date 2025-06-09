import cv2
import numpy as np
import pytesseract
import os
import re
from hero_health_color_mask import HeroHealthReader
import utils

class HeroHealthOCRReader:
    """Class to apply OCR to detected health regions from hero_health_color_mask.py."""
    
    def __init__(self):
        self.health_reader = HeroHealthReader()
        # OCR PARAMETERS - Edit these values to tune performance
        self.ocr_params = {
            'white_threshold': 220,      # Lowered from 240 - threshold for white text (200-255)
            'scale_factor': 15,          # Upscaling factor for better OCR
            'gaussian_blur': (1, 1),     # Reduced blur - kernel size (1,1) to (7,7) - use odd numbers
            'gaussian_sigma': 0.5,       # Reduced sigma - Gaussian blur sigma (0.5-2.0)
            'morph_kernel_size': 1,      # Reduced morphology - kernel size (1-5)
            'padding': 20,               # Padding around health regions (10-30)
            'rotation_angles': [0, -2, 2, -5, 5],  # Try 0° first, then very small angles for health text
            
            # CONFIDENCE PARAMETERS
            'high_confidence_early_stop': 85,  # Stop trying other methods if confidence >= this value
            'low_confidence_skip_angle': 15,   # Skip remaining PSMs for angles with max confidence below this value
        }
        
        # OCR configuration for numbers - health values can be 1-30 typically
        self.ocr_config = '--psm 8 -c tessedit_char_whitelist=0123456789'
        # Alternative configurations to try
        self.ocr_configs = [
            '--psm 8 -c tessedit_char_whitelist=0123456789',   # Single character
            '--psm 7 -c tessedit_char_whitelist=0123456789',   # Single text line
            '--psm 10 -c tessedit_char_whitelist=0123456789',  # Single character, no layout analysis
            '--psm 6 -c tessedit_char_whitelist=0123456789',   # Single uniform block
            '--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789',  # LSTM only
            '--psm 13 -c tessedit_char_whitelist=0123456789',  # Raw line
        ]
    
    def print_ocr_params(self):
        """Print current OCR parameters for easy adjustment"""
        print("Current OCR Parameters:")
        print("-" * 30)
        for param, value in self.ocr_params.items():
            print(f"  {param}: {value}")
        print("-" * 30)
        print("Tips for adjustment:")
        print("  - Lower white_threshold (200-220) if numbers aren't pure white")
        print("  - Increase gaussian_blur kernel for more smoothing")
        print("  - Increase scale_factor for larger text")
        print("  - Increase padding for more context")
        print("  - Modify rotation_angles list to try different tilts")
        print("  - high_confidence_early_stop: Stop trying methods once this confidence is reached")
        print("  - low_confidence_skip_angle: Skip remaining PSMs if angle's max confidence is below this")
        print()
    
    def extract_health_region_with_padding(self, health_image, x, y, w, h):
        """Extract health region with padding around the borders"""
        height, width = health_image.shape[:2]
        padding = self.ocr_params['padding']
        
        # Add padding but keep within image bounds
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        # Extract padded region
        padded_region = health_image[y_start:y_end, x_start:x_end]
        
        return padded_region
    
    def add_padding_to_image(self, image, padding_pixels=30):
        """Add white padding around the image to help OCR"""
        height, width = image.shape
        
        # Create new image with padding
        padded_height = height + (2 * padding_pixels)
        padded_width = width + (2 * padding_pixels)
        
        # For health numbers (white text), padding should be black (0) for inverted version
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

    def preprocess_for_ocr(self, health_region):
        """Enhance health region for OCR - process RED, GREEN, or WHITE text from health bars"""
        # Convert to grayscale if needed
        if len(health_region.shape) == 3:
            original_bgr = health_region.copy()
            gray = cv2.cvtColor(health_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = health_region
            original_bgr = None
        
        # The health region contains the colored numbers (red/green/white) detected by color mask
        # We need to create a binary mask that captures these colored pixels for OCR
        
        final_mask = None
        method_used = "unknown"
        
        if original_bgr is not None:
            # METHOD 1: Use the same color detection as the color mask
            # Define the same target colors as in hero_health_color_mask.py
            target_colors = {
                'red': np.array([35, 52, 234]),    # BGR format
                'green': np.array([76, 251, 117]),
                'white': np.array([255, 255, 255])
            }
            color_tolerance = 40  # Slightly higher tolerance than the original mask
            
            # Create masks for each color
            red_mask = self.create_color_mask_for_ocr(original_bgr, target_colors['red'], color_tolerance)
            green_mask = self.create_color_mask_for_ocr(original_bgr, target_colors['green'], color_tolerance)
            white_mask = self.create_color_mask_for_ocr(original_bgr, target_colors['white'], color_tolerance)
            
            # Also try white detection with threshold method
            gray_for_white = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
            _, white_thresh_mask = cv2.threshold(gray_for_white, 200, 255, cv2.THRESH_BINARY)
            
            # Combine all color masks
            combined_mask = cv2.bitwise_or(red_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, white_mask)
            combined_mask = cv2.bitwise_or(combined_mask, white_thresh_mask)
            
            # Count pixels found by each method
            red_pixels = np.sum(red_mask == 255)
            green_pixels = np.sum(green_mask == 255)
            white_pixels = np.sum(white_mask == 255)
            white_thresh_pixels = np.sum(white_thresh_mask == 255)
            total_colored_pixels = np.sum(combined_mask == 255)
            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            
            print(f"      Color detection: Red={red_pixels}px, Green={green_pixels}px, White={white_pixels}px, WhiteThresh={white_thresh_pixels}px")
            print(f"      Total colored pixels: {total_colored_pixels} ({(total_colored_pixels/total_pixels)*100:.1f}%)")
            
            if total_colored_pixels > 10:  # Found some colored pixels
                final_mask = combined_mask
                method_used = "color_detection"
            else:
                print(f"      No colored pixels found, falling back to grayscale")
        
        # METHOD 2: Fallback to grayscale if color detection failed or no color image
        if final_mask is None:
            # Try multiple thresholds to find the best one
            thresholds_to_try = [150, 180, 200, 220, 240]
            best_mask = None
            best_pixel_count = 0
            best_threshold = 150
            
            for thresh in thresholds_to_try:
                _, test_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                pixel_count = np.sum(test_mask == 255)
                
                # We want a reasonable amount of pixels (not too few, not too many)
                if 20 < pixel_count < (gray.shape[0] * gray.shape[1] * 0.8):  # Between 20 pixels and 80% of image
                    if pixel_count > best_pixel_count:
                        best_mask = test_mask
                        best_pixel_count = pixel_count
                        best_threshold = thresh
            
            if best_mask is not None:
                final_mask = best_mask
                method_used = f"grayscale_thresh_{best_threshold}"
                print(f"      Grayscale detection: {best_pixel_count} pixels at threshold {best_threshold}")
            else:
                # Last resort - use adaptive threshold
                final_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                method_used = "adaptive_threshold"
                adaptive_pixels = np.sum(final_mask == 255)
                print(f"      Adaptive threshold: {adaptive_pixels} pixels")
        
        # Clean up the mask
        if final_mask is not None:
            # Apply light morphological operations to connect broken parts
            kernel_size = max(1, self.ocr_params['morph_kernel_size'])
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply light Gaussian blur only if requested
            blur_kernel = self.ocr_params['gaussian_blur']
            if blur_kernel[0] > 1:
                blur_sigma = self.ocr_params['gaussian_sigma']
                blurred = cv2.GaussianBlur(final_mask, blur_kernel, blur_sigma)
                _, final_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
        
        # Scale up for better OCR
        scale_factor = self.ocr_params['scale_factor']
        if final_mask is not None:
            height, width = final_mask.shape
            scaled = cv2.resize(final_mask, (width * scale_factor, height * scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
            
            # Re-threshold after scaling to ensure clean binary image
            _, scaled_clean = cv2.threshold(scaled, 128, 255, cv2.THRESH_BINARY)
        else:
            # Fallback - create empty mask
            scaled_clean = np.zeros((100, 100), dtype=np.uint8)
            method_used = "failed"
        
        # Create inverted version (black text on white background for OCR)
        scaled_clean_inv = cv2.bitwise_not(scaled_clean)
        
        # ADD SIGNIFICANT PADDING to help OCR recognition
        padded_inv = self.add_padding_to_image(scaled_clean_inv, padding_pixels=30)
        
        print(f"      Processing method: {method_used}")
        
        return scaled_clean, padded_inv
    
    def create_color_mask_for_ocr(self, image, target_color, tolerance):
        """Create mask for a specific color with tolerance - same as color mask but for OCR"""
        lower_bound = np.clip(target_color - tolerance, 0, 255)
        upper_bound = np.clip(target_color + tolerance, 0, 255)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask
    
    def ocr_health_region(self, health_region, region_id, health_type):
        """Apply OCR to a single health region - collect all results and pick best by confidence"""
        print(f"  OCR on {health_type} health region {region_id}:")
        
        all_results = []  # Store all valid results with confidence scores
        high_confidence_threshold = self.ocr_params['high_confidence_early_stop']
        low_confidence_skip_threshold = self.ocr_params['low_confidence_skip_angle']
        
        # Try different rotation angles
        for angle in self.ocr_params['rotation_angles']:
            print(f"    Trying rotation: {angle}°")
            
            # Rotate the health region
            rotated_region = self.rotate_image(health_region, angle)
            
            # Get white mask versions
            white_mask, white_mask_inv = self.preprocess_for_ocr(rotated_region)
            
            angle_max_confidence = 0  # Track max confidence for this angle
            angle_results = []  # Store results for this angle
            angle_skip_remaining = False  # Flag to skip remaining PSMs for this angle
            
            # Try each PSM mode on white_mask_inv
            for config_name, config in zip(["PSM8", "PSM7", "PSM10", "PSM6", "PSM8_LSTM", "PSM13"], self.ocr_configs):
                if angle_skip_remaining:
                    break
                    
                try:
                    # Get OCR result with confidence (using white_mask_inv - black text on white background)
                    data = pytesseract.image_to_data(white_mask_inv, config=config, output_type=pytesseract.Output.DICT)
                    
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
                            if 1 <= number <= 99:  # Valid health range (heroes can have 1-99 health typically)
                                result = {
                                    'number': number,
                                    'confidence': confidence,
                                    'angle': angle,
                                    'method': f"white_mask_inv_{config_name}",
                                    'raw_text': text,
                                    'clean_text': clean_text,
                                    'image': white_mask_inv.copy(),
                                    'rotated_original': rotated_region.copy()
                                }
                                angle_results.append(result)
                                all_results.append(result)
                                print(f"      + {config_name} at {angle}°: '{number}' (confidence: {confidence}%)")
                                
                                # EARLY STOP: If we hit high confidence, stop trying other PSMs for this angle
                                if confidence >= high_confidence_threshold:
                                    print(f"      🎯 High confidence ({confidence}%) reached! Stopping other PSMs for {angle}°")
                                    angle_skip_remaining = True
                                    break
                                    
                            else:
                                print(f"      - {config_name} at {angle}°: '{clean_text}' (out of range 1-99, conf: {confidence}%)")
                        else:
                            if text.strip():
                                print(f"      - {config_name} at {angle}°: '{text}' (not number, conf: {confidence}%)")
                    else:
                        print(f"      - {config_name} at {angle}°: No text found")
                        
                    # LOW CONFIDENCE SKIP: Check if we should skip remaining PSMs after each attempt
                    if angle_max_confidence > 0 and angle_max_confidence < low_confidence_skip_threshold:
                        print(f"      ⏭️  Angle {angle}° max confidence ({angle_max_confidence}%) below threshold ({low_confidence_skip_threshold}%), skipping remaining PSMs")
                        angle_skip_remaining = True
                        break
                    
                except Exception as e:
                    print(f"      ✗ {config_name} at {angle}°: Error - {e}")
            
            # EARLY STOP: If we found a very high confidence result, consider stopping all angles
            if angle_results and max(r['confidence'] for r in angle_results) >= high_confidence_threshold:
                print(f"    🎯 Very high confidence ({max(r['confidence'] for r in angle_results)}%) found at {angle}°!")
                print(f"    🎯 Stopping early - no need to try remaining rotation angles")
                break
        
        # Analyze all results and pick the best one
        if all_results:
            # Sort by confidence (highest first)
            all_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            print(f"    📊 Found {len(all_results)} valid results:")
            for i, result in enumerate(all_results[:5]):  # Show top 5
                print(f"      {i+1}. Number '{result['number']}' at {result['angle']}° (confidence: {result['confidence']}%)")
            
            # Check if there's a clear winner or if results are close
            best_result = all_results[0]
            
            # If multiple results have similar high confidence, check for consensus
            high_conf_results = [r for r in all_results if r['confidence'] >= best_result['confidence'] * 0.9]
            
            if len(high_conf_results) > 1:
                # Check if most high-confidence results agree on the same number
                numbers = [r['number'] for r in high_conf_results]
                most_common = max(set(numbers), key=numbers.count)
                consensus_count = numbers.count(most_common)
                
                print(f"    🤝 Consensus check: {consensus_count}/{len(high_conf_results)} high-conf results agree on '{most_common}'")
                
                # If there's strong consensus, prefer that number even if not highest confidence
                if consensus_count > len(high_conf_results) / 2:
                    consensus_results = [r for r in high_conf_results if r['number'] == most_common]
                    best_result = max(consensus_results, key=lambda x: x['confidence'])
                    print(f"    ✓ Using consensus result: '{best_result['number']}' (confidence: {best_result['confidence']}%)")
                else:
                    print(f"    ✓ Using highest confidence: '{best_result['number']}' (confidence: {best_result['confidence']}%)")
            else:
                print(f"    ✓ Clear winner: '{best_result['number']}' (confidence: {best_result['confidence']}%)")
            
            # Save the best result for debugging (only if DEBUG_IMAGES is True)
            if utils.DEBUG_IMAGES:
                cv2.imwrite(f'{health_type}_health_region_{region_id}_BEST_{best_result["angle"]}deg_{best_result["method"]}.png', best_result['image'])
                cv2.imwrite(f'{health_type}_health_region_{region_id}_BEST_{best_result["angle"]}deg_original.png', best_result['rotated_original'])
            
            return best_result['number'], best_result['clean_text'], f"{best_result['angle']}deg_{best_result['method']}_conf{best_result['confidence']}"
        
        print(f"      ✗ No valid number found at any rotation angle")
        return None, "", "failed"
    
    def analyze_health_regions(self, health_type='enemy'):
        """Analyze health regions for enemy or ally and apply OCR to each detected region"""
        print(f"\n{health_type.capitalize()} Health OCR Analysis")
        print("=" * 40)
        
        # Get color mask and health regions from hero_health_color_mask
        try:
            combined_mask, color_boxes, color_analysis = self.health_reader.analyze_health_colors(health_type)
        except Exception as e:
            print(f"Error getting health regions from hero_health_color_mask: {e}")
            return None
        
        if not color_boxes:
            print(f"No {health_type} health regions detected! Check hero_health_color_mask.py output first.")
            return None
        
        # Load original health image
        health_image = self.health_reader.load_health_image(health_type)
        
        print(f"\nApplying OCR to {len(color_boxes)} detected {health_type} health regions...")
        print("-" * 40)
        
        ocr_results = []
        region_images = []
        
        # Process each detected health region
        for i, (x, y, w, h) in enumerate(color_boxes):
            region_id = i + 1
            print(f"\n{health_type.capitalize()} Health Region {region_id}: position=({x},{y}), size=({w}x{h})")
            
            # Extract health region WITH PADDING for better border capture
            health_region = self.extract_health_region_with_padding(health_image, x, y, w, h)
            region_images.append(health_region)
            
            # Apply OCR
            number, raw_text, method = self.ocr_health_region(health_region, region_id, health_type)
            
            result = {
                'id': region_id,
                'position': (x, y, w, h),
                'number': number,
                'raw_text': raw_text,
                'method': method,
                'success': number is not None,
                'health_type': health_type
            }
            ocr_results.append(result)
            
            # Save individual region image for debugging (only if DEBUG_IMAGES is True)
            if utils.DEBUG_IMAGES:
                cv2.imwrite(f'{health_type}_health_region_{region_id}.png', health_region)
                
                # Save processed mask versions for debugging - CRITICAL for troubleshooting
                processed_mask, processed_mask_inv = self.preprocess_for_ocr(health_region)
                cv2.imwrite(f'{health_type}_health_region_{region_id}_processed_mask.png', processed_mask)
                cv2.imwrite(f'{health_type}_health_region_{region_id}_processed_mask_inv.png', processed_mask_inv)
                
                # Also save the original color image for comparison
                if len(health_region.shape) == 3:
                    cv2.imwrite(f'{health_type}_health_region_{region_id}_original_color.png', health_region)
                else:
                    cv2.imwrite(f'{health_type}_health_region_{region_id}_original_gray.png', health_region)
                
                print(f"        Debug: Saved region images for inspection")
        
        # Create summary visualization
        self.create_health_ocr_visualization(health_image, color_boxes, ocr_results, health_type)
        
        return ocr_results
    
    def analyze_both_health_regions(self):
        """Analyze both enemy and ally health regions and return combined results"""
        print("Hero Health OCR Reader - OCR on All Detected Health Regions")
        print("=" * 65)
        
        # Print current parameters for easy adjustment
        self.print_ocr_params()
        
        results = {}
        
        for health_type in ['enemy', 'ally']:
            filename = f'images/preprocess_{health_type}_health.png'
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found! Skipping {health_type} health analysis.")
                results[health_type] = None
                continue
            
            ocr_results = self.analyze_health_regions(health_type)
            results[health_type] = ocr_results
            
            # Print summary for this health type
            if ocr_results:
                successful = [r for r in ocr_results if r['success']]
                print(f"\n{health_type.capitalize()} Health Summary:")
                print(f"Successfully read: {len(successful)}/{len(ocr_results)} regions")
                
                if successful:
                    health_values = [r['number'] for r in successful]
                    print(f"Detected health values: {health_values}")
                    
                    # For health, we typically expect only 1 region per hero
                    if len(successful) == 1:
                        print(f"🎯 Perfect! Found {health_type} health: {successful[0]['number']}")
                    elif len(successful) > 1:
                        print(f"⚠️  Multiple health values detected - using highest confidence")
                        best_result = max(successful, key=lambda x: int(x['method'].split('conf')[1]) if 'conf' in x['method'] else 0)
                        print(f"🎯 Best {health_type} health: {best_result['number']}")
                else:
                    print(f"⚠️  No valid health numbers detected for {health_type}")
        
        return results
    
    def create_health_ocr_visualization(self, health_image, color_boxes, ocr_results, health_type):
        """Create visualization showing OCR results on the health image"""
        visualization = health_image.copy()
        
        for i, ((x, y, w, h), result) in enumerate(zip(color_boxes, ocr_results)):
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
            
            # Put larger text above the region
            cv2.putText(visualization, text, (x+5, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            
            # Add region ID below
            cv2.putText(visualization, f"#{result['id']}", (x, y+h+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save visualization (only if DEBUG_IMAGES is True)
        if utils.DEBUG_IMAGES:
            cv2.imwrite(f'{health_type}_health_ocr_results.png', visualization)
            print(f"\nSaved: {health_type}_health_ocr_results.png")
            
            # Save individual region images
            for i in range(len(color_boxes)):
                print(f"Saved: {health_type}_health_region_{i+1}.png")
    
    def print_final_summary(self, results):
        """Print final summary of all health OCR results"""
        print(f"\nFinal Health OCR Summary:")
        print("=" * 40)
        
        for health_type, ocr_results in results.items():
            if ocr_results is None:
                print(f"{health_type.capitalize()} health: File not found")
                continue
                
            successful = [r for r in ocr_results if r['success']]
            
            if successful:
                # Get the best result (highest confidence)
                if len(successful) == 1:
                    health_value = successful[0]['number']
                    print(f"{health_type.capitalize()} health: {health_value}")
                else:
                    # Multiple results - pick highest confidence
                    best_result = max(successful, key=lambda x: int(x['method'].split('conf')[1]) if 'conf' in x['method'] else 0)
                    health_value = best_result['number']
                    print(f"{health_type.capitalize()} health: {health_value} (best of {len(successful)} detections)")
            else:
                print(f"{health_type.capitalize()} health: Not detected")
        
        if utils.DEBUG_IMAGES:
            print(f"\nFiles generated:")
            for health_type in ['enemy', 'ally']:
                if results.get(health_type):
                    print(f"  - {health_type}_health_ocr_results.png: Visualization with OCR results")
                    print(f"  - {health_type}_health_region_X.png: Individual region images (with padding)")
                    print(f"  - {health_type}_health_region_X_processed_mask.png: Color/text isolation")
                    print(f"  - {health_type}_health_region_X_BEST_method.png: Successful OCR version (if any)")
        else:
            print(f"\nNote: Debug images disabled (utils.DEBUG_IMAGES = False)")
            print(f"      Set utils.DEBUG_IMAGES = True to generate debug PNG files")
        
        print(f"\nTip: If no numbers detected, try lowering white_threshold from 220 to 180-200")
        print(f"Tip: Adjust high_confidence_early_stop (currently {self.ocr_params['high_confidence_early_stop']}%) to stop early when confident")


def main():
    """Main function to run OCR analysis on detected health regions"""
    print("Starting OCR analysis on detected hero health regions...")
    
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
        # Create OCR reader
        ocr_reader = HeroHealthOCRReader()
        
        # Analyze both health regions
        results = ocr_reader.analyze_both_health_regions()
        
        # Print final summary
        ocr_reader.print_final_summary(results)
        
        # Extract final health values
        enemy_health = None
        ally_health = None
        
        if results.get('enemy'):
            successful_enemy = [r for r in results['enemy'] if r['success']]
            if successful_enemy:
                enemy_health = max(successful_enemy, key=lambda x: int(x['method'].split('conf')[1]) if 'conf' in x['method'] else 0)['number']
        
        if results.get('ally'):
            successful_ally = [r for r in results['ally'] if r['success']]
            if successful_ally:
                ally_health = max(successful_ally, key=lambda x: int(x['method'].split('conf')[1]) if 'conf' in x['method'] else 0)['number']
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Enemy Health: {enemy_health if enemy_health is not None else 'Not detected'}")
        print(f"   Ally Health: {ally_health if ally_health is not None else 'Not detected'}")
        
        return {'enemy_health': enemy_health, 'ally_health': ally_health}
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()