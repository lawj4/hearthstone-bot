import cv2
import numpy as np
import pytesseract
import os
import re
import utils

class ManaCrystalReader:
    """Class specifically for reading mana crystals in #/# format from preprocess_mana_crystals.png"""
    
    def __init__(self, save_debug_images=utils.DEBUG_IMAGES):
        # Use only OCR configurations that work reliably for #/# pattern
        self.ocr_configs = [
            '--psm 7 -c tessedit_char_whitelist=0123456789/',   # Single text line - PRIMARY
            '--psm 6 -c tessedit_char_whitelist=0123456789/',   # Single uniform block - BACKUP
        ]
        self.save_debug_images = save_debug_images
    
    def load_mana_crystals_image(self):
        """Load the mana crystals region from saved screenshot"""
        filename = os.path.join(utils.IMAGE_DIR, 'preprocess_mana_crystals.png')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Mana crystals file not found: {filename}")
        
        mana_crystals_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
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
                    
                    # STRICT: Look for exact #/# pattern
                    match = re.search(r'^(\d+)/(\d+)$', clean_text)  # Must be exactly #/#
                    if match:
                        current_mana = int(match.group(1))
                        max_mana = int(match.group(2))
                        
                        # Validate reasonable mana values (0-10 each)
                        if 0 <= current_mana <= 10 and 0 <= max_mana <= 10:
                            return current_mana, max_mana, text
                
                except Exception as e:
                    continue  # Try next config
            
            return None, None, ""
            
        except Exception as e:
            if self.save_debug_images:
                print(f"OCR Error: {e}")
            return None, None, ""
    
    def analyze_mana_crystals(self):
        """Analyze mana crystals from preprocess_mana_crystals.png"""
        filename = os.path.join(utils.IMAGE_DIR, 'preprocess_mana_crystals.png')
        if not os.path.exists(filename):
            if self.save_debug_images:
                print("Error: preprocess_mana_crystals.png not found!")
                print("Please run hearthstone_regions.py first to generate the region files.")
            return None, None
        
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
        
        return result, mana_image
    
    def save_analysis_results(self, result, mana_image, prefix='mana_crystals_analysis'):
        """Save the mana crystal analysis results as images"""
        if not self.save_debug_images:
            return
            
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
        print(f"Saved: {prefix}_overview.png")
        
        # Save original and processed versions
        cv2.imwrite(f'{prefix}_original.png', mana_image)
        processed = self.preprocess_for_ocr(mana_image)
        cv2.imwrite(f'{prefix}_processed.png', processed)
        
        print(f"Saved: {prefix}_original.png")
        print(f"Saved: {prefix}_processed.png")

def main():
    """Main function to analyze mana crystals from preprocess_mana_crystals.png"""
    reader = ManaCrystalReader()
    
    print("Hearthstone Mana Crystal Reader (Optimized)")
    filename = os.path.join(utils.IMAGE_DIR, 'preprocess_mana_crystals.png')
    
    # Check if test file exists
    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        print("Please run hearthstone_regions.py first to generate this file.")
        return
    
    # Analyze mana crystals
    result, mana_image = reader.analyze_mana_crystals()
    
    if result is None:
        return
        
    # Print results
    if result['success']:
        print(f"✓ Current Mana: {result['current_mana']}/{result['max_mana']}")
        print(f"✓ Available: {result['current_mana']}, Used: {result['max_mana'] - result['current_mana']}")
    else:
        print(f"✗ Detection failed - Raw text: '{result['raw_text']}'")
    
    # Save analysis results
    reader.save_analysis_results(result, mana_image, 'mana_crystals_analysis')

if __name__ == "__main__":
    main()