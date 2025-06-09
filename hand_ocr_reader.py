import cv2
import numpy as np
import pytesseract
import os
import re
import logging
from hand_white_mask import HandReader
import utils

class HandOCRReader:
    """Class to apply OCR to all detected white objects from hand_white_mask.py."""
    
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger('HandOCRReader')
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
        
        self.hand_reader = HandReader()
        # OCR PARAMETERS - Edit these values to tune performance
        self.ocr_params = {
            'white_threshold': 250,      # Threshold for white text (200-255)
            'scale_factor': 10,          # Upscaling factor (increased from 8 to 20)
            'gaussian_blur': (7, 7),     # Gaussian blur kernel size (1,1) to (7,7) - use odd numbers
            'gaussian_sigma': 0.5,       # Gaussian blur sigma (0.5-2.0)
            'morph_kernel_size': 1,      # Morphological operations kernel size (1-5)
            'padding': 5,               # Padding around crystals (10-25)
            'rotation_angles': [0, -5, 5, -10, 10, -15, 15],  # Try 0° first, then small angles
            
            # NEW CONFIDENCE PARAMETERS
            'high_confidence_early_stop': 85,  # Stop trying other methods if confidence >= this value
            'low_confidence_skip_angle': 20,   # Skip remaining PSMs for angles with max confidence below this value
        }
        
        # OCR configuration for numbers
        self.ocr_config = '--psm 8 -c tessedit_char_whitelist=0123456789'
        # Alternative configurations to try
        self.ocr_configs = [
            '--psm 8 -c tessedit_char_whitelist=0123456789',   # Single character
            '--psm 10 -c tessedit_char_whitelist=0123456789',  # Single character, no layout analysis
            '--psm 7 -c tessedit_char_whitelist=0123456789',   # Single text line
            '--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789',  # LSTM only
            '--psm 6 -c tessedit_char_whitelist=0123456789',   # Single uniform block
            '--psm 13 -c tessedit_char_whitelist=0123456789',  # Raw line
        ]
        
        self.logger.info("HandOCRReader initialized")
    
    def print_ocr_params(self):
        """Log current OCR parameters for easy adjustment"""
        self.logger.info("Current OCR Parameters:")
        for param, value in self.ocr_params.items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info("Tips for adjustment:")
        self.logger.info("  - Lower white_threshold (200-220) if numbers aren't pure white")
        self.logger.info("  - Increase gaussian_blur kernel for more smoothing")
        self.logger.info("  - Increase scale_factor for larger text")
        self.logger.info("  - Increase padding for more context")
        self.logger.info("  - Modify rotation_angles list to try different tilts")
        self.logger.info("  - high_confidence_early_stop: Stop trying methods once this confidence is reached")
        self.logger.info("  - low_confidence_skip_angle: Skip remaining PSMs if angle's max confidence is below this")
    
    def extract_crystal_with_padding(self, hand_image, x, y, w, h):
        """Extract crystal region with padding around the borders"""
        height, width = hand_image.shape[:2]
        padding = self.ocr_params['padding']
        
        # Add padding but keep within image bounds
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        # Extract padded region
        padded_crystal = hand_image[y_start:y_end, x_start:x_end]
        
        self.logger.debug(f"Extracted crystal region with padding: {padding}px")
        return padded_crystal
    
    def add_padding_to_image(self, image, padding_pixels=30):
        """Add white padding around the image to help OCR"""
        height, width = image.shape
        
        # Create new image with padding
        padded_height = height + (2 * padding_pixels)
        padded_width = width + (2 * padding_pixels)
        
        # For white_mask_inv (black text on white), padding should be white (255)
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
        """Enhance crystal image for OCR with Gaussian blur and smoothing"""
        # Convert to grayscale if needed
        if len(crystal_image.shape) == 3:
            gray = cv2.cvtColor(crystal_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crystal_image
        
        # Apply Gaussian blur to smooth jagged edges and connect broken parts
        blur_kernel = self.ocr_params['gaussian_blur']
        blur_sigma = self.ocr_params['gaussian_sigma']
        blurred = cv2.GaussianBlur(gray, blur_kernel, blur_sigma)
        
        # WHITE MASK - isolate pure white pixels (the numbers)
        white_threshold = self.ocr_params['white_threshold']
        _, white_mask = cv2.threshold(blurred, white_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to connect broken parts and smooth edges
        kernel_size = self.ocr_params['morph_kernel_size']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Closing operation: dilation followed by erosion - fills gaps and connects broken parts
        closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Scale up for better OCR
        scale_factor = self.ocr_params['scale_factor']
        height, width = closed.shape
        scaled = cv2.resize(closed, (width * scale_factor, height * scale_factor), 
                          interpolation=cv2.INTER_CUBIC)
        
        # Final cleanup with morphological operations on scaled image
        kernel_scaled = np.ones((2,2), np.uint8)
        final_cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel_scaled)
        
        # Also try inverted version (black text on white background for OCR)
        final_cleaned_inv = cv2.bitwise_not(final_cleaned)
        
        # ADD SIGNIFICANT PADDING to help OCR recognition
        padded_inv = self.add_padding_to_image(final_cleaned_inv, padding_pixels=30)
        
        return final_cleaned, padded_inv
    
    def ocr_crystal(self, crystal_image, crystal_id):
        """Apply OCR to a single crystal image - collect all results and pick best by confidence"""
        self.logger.debug(f"Starting OCR on crystal {crystal_id}")
        
        all_results = []  # Store all valid results with confidence scores
        high_confidence_threshold = self.ocr_params['high_confidence_early_stop']
        low_confidence_skip_threshold = self.ocr_params['low_confidence_skip_angle']
        
        # Try different rotation angles
        for angle in self.ocr_params['rotation_angles']:
            self.logger.debug(f"Trying rotation: {angle}° for crystal {crystal_id}")
            
            # Rotate the crystal image
            rotated_crystal = self.rotate_image(crystal_image, angle)
            
            # Get white mask versions
            white_mask, white_mask_inv = self.preprocess_for_ocr(rotated_crystal)
            
            angle_max_confidence = 0  # Track max confidence for this angle
            angle_results = []  # Store results for this angle
            angle_skip_remaining = False  # Flag to skip remaining PSMs for this angle
            
            # Try each PSM mode on white_mask_inv
            for config_name, config in zip(["PSM8", "PSM10", "PSM7", "PSM8_LSTM", "PSM6", "PSM13"], self.ocr_configs):
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
                            if 0 <= number <= 10:  # Valid mana cost range
                                result = {
                                    'number': number,
                                    'confidence': confidence,
                                    'angle': angle,
                                    'method': f"white_mask_inv_{config_name}",
                                    'raw_text': text,
                                    'clean_text': clean_text,
                                    'image': white_mask_inv.copy(),
                                    'rotated_original': rotated_crystal.copy()
                                }
                                angle_results.append(result)
                                all_results.append(result)
                                self.logger.debug(f"Crystal {crystal_id} - {config_name} at {angle}°: '{number}' (confidence: {confidence}%)")
                                
                                # EARLY STOP: If we hit high confidence, stop trying other PSMs for this angle
                                if confidence >= high_confidence_threshold:
                                    self.logger.debug(f"Crystal {crystal_id} - High confidence ({confidence}%) reached! Stopping other PSMs for {angle}°")
                                    angle_skip_remaining = True
                                    break
                                    
                            else:
                                self.logger.debug(f"Crystal {crystal_id} - {config_name} at {angle}°: '{clean_text}' (out of range, conf: {confidence}%)")
                        else:
                            if text.strip():
                                self.logger.debug(f"Crystal {crystal_id} - {config_name} at {angle}°: '{text}' (not number, conf: {confidence}%)")
                    else:
                        self.logger.debug(f"Crystal {crystal_id} - {config_name} at {angle}°: No text found")
                    
                    # LOW CONFIDENCE SKIP: Check if we should skip remaining PSMs after each attempt
                    if angle_max_confidence > 0 and angle_max_confidence < low_confidence_skip_threshold:
                        self.logger.debug(f"Crystal {crystal_id} - Angle {angle}° max confidence ({angle_max_confidence}%) below threshold ({low_confidence_skip_threshold}%), skipping remaining PSMs")
                        angle_skip_remaining = True
                        break
                    
                except Exception as e:
                    self.logger.error(f"Crystal {crystal_id} - {config_name} at {angle}°: Error - {e}")
            
            # EARLY STOP: If we found a very high confidence result, consider stopping all angles
            if angle_results and max(r['confidence'] for r in angle_results) >= high_confidence_threshold:
                self.logger.debug(f"Crystal {crystal_id} - Very high confidence ({max(r['confidence'] for r in angle_results)}%) found at {angle}°! Stopping early")
                break
        
        # Analyze all results and pick the best one
        if all_results:
            # Sort by confidence (highest first)
            all_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.debug(f"Crystal {crystal_id} - Found {len(all_results)} valid results")
            for i, result in enumerate(all_results[:5]):  # Show top 5
                self.logger.debug(f"Crystal {crystal_id} - Result {i+1}: Number '{result['number']}' at {result['angle']}° (confidence: {result['confidence']}%)")
            
            # Check if there's a clear winner or if results are close
            best_result = all_results[0]
            
            # If multiple results have similar high confidence, check for consensus
            high_conf_results = [r for r in all_results if r['confidence'] >= best_result['confidence'] * 0.9]
            
            if len(high_conf_results) > 1:
                # Check if most high-confidence results agree on the same number
                numbers = [r['number'] for r in high_conf_results]
                most_common = max(set(numbers), key=numbers.count)
                consensus_count = numbers.count(most_common)
                
                self.logger.debug(f"Crystal {crystal_id} - Consensus check: {consensus_count}/{len(high_conf_results)} high-conf results agree on '{most_common}'")
                
                # If there's strong consensus, prefer that number even if not highest confidence
                if consensus_count > len(high_conf_results) / 2:
                    consensus_results = [r for r in high_conf_results if r['number'] == most_common]
                    best_result = max(consensus_results, key=lambda x: x['confidence'])
                    self.logger.info(f"Crystal {crystal_id} - Using consensus result: '{best_result['number']}' (confidence: {best_result['confidence']}%)")
                else:
                    self.logger.info(f"Crystal {crystal_id} - Using highest confidence: '{best_result['number']}' (confidence: {best_result['confidence']}%)")
            else:
                self.logger.info(f"Crystal {crystal_id} - Clear winner: '{best_result['number']}' (confidence: {best_result['confidence']}%)")
            
            # Save the best result for debugging (only if DEBUG_IMAGES is True)
            if utils.DEBUG_IMAGES:
                cv2.imwrite(f'crystal_{crystal_id}_BEST_{best_result["angle"]}deg_{best_result["method"]}.png', best_result['image'])
                cv2.imwrite(f'crystal_{crystal_id}_BEST_{best_result["angle"]}deg_original.png', best_result['rotated_original'])
            
            return best_result['number'], best_result['clean_text'], f"{best_result['angle']}deg_{best_result['method']}_conf{best_result['confidence']}"
        
        self.logger.warning(f"Crystal {crystal_id} - No valid number found at any rotation angle")
        return None, "", "failed"
    
    def analyze_all_crystals(self):
        """Analyze all detected crystals and apply OCR to each"""
        self.logger.info("Starting hand OCR analysis")
        
        # Print current parameters for easy adjustment
        self.print_ocr_params()
        
        # Get white mask and crystal locations from hand_white_mask
        try:
            white_mask, crystal_boxes = self.hand_reader.analyze_white_mask()
        except Exception as e:
            self.logger.error(f"Error getting crystals from hand_white_mask: {e}")
            return
        
        if not crystal_boxes:
            self.logger.warning("No crystals detected! Check hand_white_mask.py output first.")
            return
        
        # Load original hand image
        hand_image = self.hand_reader.load_hand_image()
        
        self.logger.info(f"Applying OCR to {len(crystal_boxes)} detected crystals")
        
        # Sort crystals from left to right based on x-coordinate
        if crystal_boxes:
            self.logger.debug("Sorting crystals from left to right...")
            crystal_boxes.sort(key=lambda box: box[0])  # Sort by x-coordinate (leftmost first)
            
            # Log sorted order
            for i, (x, y, w, h) in enumerate(crystal_boxes):
                self.logger.debug(f"Crystal {i+1}: x={x} (leftmost to rightmost)")
        
        ocr_results = []
        crystal_images = []
        
        # Process each detected crystal
        for i, (x, y, w, h) in enumerate(crystal_boxes):
            crystal_id = i + 1
            self.logger.debug(f"Processing crystal {crystal_id}: position=({x},{y}), size=({w}x{h})")
            
            # Extract crystal region WITH PADDING for better border capture
            crystal_region = self.extract_crystal_with_padding(hand_image, x, y, w, h)
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
                cv2.imwrite(f'crystal_{crystal_id}.png', crystal_region)
                
                # Save white mask versions for debugging
                white_mask, white_mask_inv = self.preprocess_for_ocr(crystal_region)
                cv2.imwrite(f'crystal_{crystal_id}_white_mask.png', white_mask)
                cv2.imwrite(f'crystal_{crystal_id}_white_mask_inv.png', white_mask_inv)
        
        # Create summary visualization
        self.create_ocr_visualization(hand_image, crystal_boxes, ocr_results)
        
        # Log summary
        self.log_ocr_summary(ocr_results)
        
        return ocr_results
    
    def create_ocr_visualization(self, hand_image, crystal_boxes, ocr_results):
        """Create visualization showing OCR results on the hand image"""
        visualization = hand_image.copy()
        
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
            cv2.imwrite('hand_ocr_results.png', visualization)
            self.logger.debug("Saved: hand_ocr_results.png")
            
            # Save individual crystal images
            for i in range(len(crystal_boxes)):
                self.logger.debug(f"Saved: crystal_{i+1}.png")
    
    def log_ocr_summary(self, ocr_results):
        """Log summary of OCR results"""
        self.logger.info("OCR Summary:")
        
        successful = [r for r in ocr_results if r['success']]
        failed = [r for r in ocr_results if not r['success']]
        
        self.logger.info(f"Successfully read: {len(successful)}/{len(ocr_results)} crystals")
        
        if successful:
            self.logger.info("Detected numbers:")
            for result in successful:
                self.logger.info(f"  Crystal {result['id']}: {result['number']} (method: {result['method']})")
        
        if failed:
            self.logger.info("Failed to read:")
            for result in failed:
                self.logger.info(f"  Crystal {result['id']}: position {result['position'][:2]}")
        
        if utils.DEBUG_IMAGES:
            self.logger.info("Files generated:")
            self.logger.info("  - hand_ocr_results.png: Visualization with OCR results")
            self.logger.info(f"  - crystal_1.png to crystal_{len(ocr_results)}.png: Individual crystal images (with padding)")
            self.logger.info("  - crystal_X_white_mask.png: White text isolation (white on black)")
            self.logger.info("  - crystal_X_white_mask_inv.png: White text isolation (black on white)")
            self.logger.info("  - crystal_X_BEST_method.png: Successful OCR version (if any)")
        else:
            self.logger.info("Note: Debug images disabled (utils.DEBUG_IMAGES = False)")
        
        self.logger.info(f"Tip: If no numbers detected, try lowering white_threshold from 240 to 220-235")
        self.logger.info(f"Tip: Adjust high_confidence_early_stop (currently {self.ocr_params['high_confidence_early_stop']}%) to stop early when confident")
        self.logger.info(f"Tip: Adjust low_confidence_skip_angle (currently {self.ocr_params['low_confidence_skip_angle']}%) to skip poor angles faster")


def main():
    """Main function to run OCR analysis on detected crystals"""
    # Set up logging for main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('HandOCRMain')
    
    logger.info("Starting OCR analysis on detected white crystals...")
    
    # Check if test_hand.png exists
    file_name = "images/preprocess_hand.png"
    if not os.path.exists(file_name):
        logger.error(f"{file_name} not found! Please run hearthstone_regions.py first to generate this file.")
        return
    
    try:
        # Create OCR reader
        ocr_reader = HandOCRReader()
        
        # Analyze all crystals
        results = ocr_reader.analyze_all_crystals()
        
        if results:
            # Count successful OCR results
            numbers_found = [r['number'] for r in results if r['success']]
            if numbers_found:
                logger.info(f"Successfully extracted {len(numbers_found)} mana costs: {numbers_found}")
            else:
                logger.warning("No valid numbers detected. Check individual crystal images for debugging.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()