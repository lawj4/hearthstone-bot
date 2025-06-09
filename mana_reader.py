import cv2
import numpy as np
import pytesseract
import os
import re
import logging
import utils

class ManaCrystalReader:
    """Class specifically for reading mana crystals in #/# format from preprocess_mana_crystals.png"""
    
    def __init__(self, save_debug_images=utils.DEBUG_IMAGES):
        # Set up logging
        self.logger = logging.getLogger('ManaCrystalReader')
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
        
        # Use only OCR configurations that work reliably for #/# pattern
        self.ocr_configs = [
            '--psm 7 -c tessedit_char_whitelist=0123456789/',   # Single text line - PRIMARY
            '--psm 6 -c tessedit_char_whitelist=0123456789/',   # Single uniform block - BACKUP
        ]
        self.save_debug_images = save_debug_images
        
        self.logger.info("ManaCrystalReader initialized")
    
    def load_mana_crystals_image(self):
        """Load the mana crystals region from saved screenshot"""
        filename = os.path.join(utils.IMAGE_DIR, 'preprocess_mana_crystals.png')
        if not os.path.exists(filename):
            self.logger.error(f"Mana crystals file not found: {filename}")
            raise FileNotFoundError(f"Mana crystals file not found: {filename}")
        
        mana_crystals_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self.logger.debug(f"Loaded mana crystals image: {filename}")
        return mana_crystals_image
    
    def preprocess_for_ocr(self, image):
        """Enhance image for better OCR of #/# format"""
        # Resize significantly for better OCR
        scale_factor = 4
        height, width = image.shape
        resized = cv2.resize(image, (width * scale_factor, height * scale_factor))
        
        # Apply threshold to get pure black text on white background
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        
        # FLOOD FILL: Turn white noise areas to black
        h, w = binary.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask needs to be 2 pixels larger
        
        # Flood fill from top-right corner (w-1, 0)
        cv2.floodFill(binary, mask, (w-1, 0), 0)
        
        # Reset mask for next flood fill
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill from bottom-right corner (w-1, h-1)
        cv2.floodFill(binary, mask, (w-1, h-1), 0)
        
        # Reset mask for next flood fill
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill from top-center (w//2, 0)
        cv2.floodFill(binary, mask, (w//2, h//2-50), 0)
        binary = cv2.bitwise_not(binary)
        
        self.logger.debug(f"Preprocessed image for OCR: scaled by {scale_factor}x, flood filled, and inverted")
        return binary
    
    def read_mana_fraction(self, image):
        """Read the #/# mana format using optimized OCR"""
        try:
            processed = self.preprocess_for_ocr(image)
            
            # Try each OCR configuration until we get a valid #/# result
            for config in self.ocr_configs:
                try:
                    # Get OCR result
                    text = pytesseract.image_to_string(processed, config=config).strip()
                    
                    # Clean up the text (remove whitespace, newlines)
                    clean_text = re.sub(r'\s+', '', text)
                    
                    self.logger.debug(f"OCR attempt with config '{config}': raw='{text}', clean='{clean_text}'")
                    
                    # STRICT: Look for exact #/# pattern
                    match = re.search(r'^(\d+)/(\d+)$', clean_text)  # Must be exactly #/#
                    if match:
                        current_mana = int(match.group(1))
                        max_mana = int(match.group(2))
                        
                        # Validate reasonable mana values (0-10 each)
                        if 0 <= current_mana <= 10 and 0 <= max_mana <= 10:
                            self.logger.info(f"Successfully read mana: {current_mana}/{max_mana}")
                            return current_mana, max_mana, text
                        else:
                            self.logger.debug(f"Mana values out of range: {current_mana}/{max_mana}")
                
                except Exception as e:
                    self.logger.debug(f"OCR config '{config}' failed: {e}")
                    continue  # Try next config
            
            self.logger.warning("No valid mana pattern found in any OCR attempt")
            return None, None, ""
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            return None, None, ""
    
    def analyze_mana_crystals(self):
        """Analyze mana crystals from preprocess_mana_crystals.png"""
        filename = os.path.join(utils.IMAGE_DIR, 'preprocess_mana_crystals.png')
        if not os.path.exists(filename):
            self.logger.error(f"{filename} not found! Please run hearthstone_regions.py first to generate the region files.")
            return None, None
        
        self.logger.debug("Starting mana crystal analysis...")
        
        # Load the mana crystals image
        mana_image = self.load_mana_crystals_image()
        
        # Read the mana fraction
        current_mana, max_mana, raw_text = self.read_mana_fraction(mana_image)
        
        result = {
            'current_mana': current_mana,
            'max_mana': max_mana,
            'raw_text': raw_text,
            'success': current_mana is not None and max_mana is not None
        }
        
        if result['success']:
            self.logger.info(f"Mana analysis successful: {current_mana}/{max_mana}")
        else:
            self.logger.warning(f"Mana analysis failed - Raw text: '{raw_text}'")
        
        return result, mana_image
    
    def save_analysis_results(self, result, mana_image, prefix='mana_crystals_analysis'):
        """Save the mana crystal analysis results as images"""
        if not self.save_debug_images:
            self.logger.debug("Debug image saving disabled")
            return
            
        self.logger.debug("Saving mana analysis debug images...")
        
        # Create visualization
        result_image = mana_image.copy()
        if len(result_image.shape) == 2:  # grayscale
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        
        # Add text overlay with results
        if result['success']:
            mana_text = f"{result['current_mana']}/{result['max_mana']}"
            color = (0, 255, 0)  # Green for success
        else:
            mana_text = f"Failed: '{result['raw_text']}'"
            color = (0, 0, 255)  # Red for failure
        
        # Put text on image
        cv2.putText(result_image, mana_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Save overview image
        cv2.imwrite(f'{prefix}_overview.png', result_image)
        self.logger.debug(f"Saved: {prefix}_overview.png")
        
        # Save original and processed versions
        cv2.imwrite(f'{prefix}_original.png', mana_image)
        processed = self.preprocess_for_ocr(mana_image)
        cv2.imwrite(f'{prefix}_processed.png', processed)
        
        self.logger.debug(f"Saved: {prefix}_original.png")
        self.logger.debug(f"Saved: {prefix}_processed.png")

def main():
    """Main function to analyze mana crystals from preprocess_mana_crystals.png"""
    # Set up logging for main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('ManaCrystalMain')
    
    reader = ManaCrystalReader()
    
    logger.info("Hearthstone Mana Crystal Reader (Optimized)")
    filename = os.path.join(utils.IMAGE_DIR, 'preprocess_mana_crystals.png')
    
    # Check if test file exists
    if not os.path.exists(filename):
        logger.error(f"Missing file: {filename}")
        logger.error("Please run hearthstone_regions.py first to generate this file.")
        return
    
    # Analyze mana crystals
    result, mana_image = reader.analyze_mana_crystals()
    
    if result is None:
        return
        
    # Print results to console
    if result['success']:
        print(f"✓ Current Mana: {result['current_mana']}/{result['max_mana']}")
        print(f"✓ Available: {result['current_mana']}, Used: {result['max_mana'] - result['current_mana']}")
    else:
        print(f"✗ Detection failed - Raw text: '{result['raw_text']}'")
    
    # Save analysis results
    reader.save_analysis_results(result, mana_image, 'mana_crystals_analysis')

if __name__ == "__main__":
    main()