import cv2
import numpy as np
import pytesseract
import os
import re
import utils

class ManaCrystalReader:
    """Class specifically for reading mana crystals in #/# format from preprocess_mana_crystals.png"""
    
    def __init__(self, save_debug_images=False):
        # OCR configuration for numbers and slash
        self.ocr_config = '--psm 8 -c tessedit_char_whitelist=0123456789/'
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
        
        # Apply morphological operations to clean up text
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Crop to focus on the text area - adjust these values as needed
        h, w = cleaned.shape
        
        # Define crop boundaries (adjust these percentages)
        crop_top = int(h * 0.4)      # Start 30% down from top
        crop_bottom = int(h * 0.8)   # End 70% down from top  
        crop_left = int(w * 0.15)     # Start 20% from left
        crop_right = int(w * 0.85)    # End 80% from left
        
        # Apply the crop
        cropped = cleaned[crop_top:crop_bottom, crop_left:crop_right]
        
        return cropped
    
    def read_mana_fraction(self, image):
        """Read the #/# mana format from the entire image"""
        try:
            processed = self.preprocess_for_ocr(image)
            
            # Try OCR on the processed image
            text = pytesseract.image_to_string(processed, config=self.ocr_config).strip()
            
            # Clean up the text (remove whitespace, newlines)
            text = re.sub(r'\s+', '', text)
            
            # Look for #/# pattern
            match = re.search(r'(\d+)/(\d+)', text)
            if match:
                current_mana = int(match.group(1))
                max_mana = int(match.group(2))
                
                # Validate reasonable mana values (0-10 each)
                if 0 <= current_mana <= 10 and 0 <= max_mana <= 10:
                    return current_mana, max_mana, text
            
            return None, None, text
            
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
    
    def try_alternative_preprocessing(self, image):
        """Try different preprocessing approaches if initial OCR fails"""
        methods = []
        
        # Method 1: Higher contrast
        contrast = cv2.convertScaleAbs(image, alpha=2.0, beta=0)
        methods.append(("high_contrast", contrast))
        
        # Method 2: Gaussian blur before threshold
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        _, thresh_blur = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        methods.append(("blur_threshold", thresh_blur))
        
        # Method 3: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        methods.append(("adaptive", adaptive))
        
        # Method 4: Erosion + Dilation
        kernel = np.ones((2,2), np.uint8)
        eroded = cv2.erode(image, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        methods.append(("erode_dilate", dilated))
        
        return methods

def main():
    """Main function to analyze mana crystals from preprocess_mana_crystals.png"""
    reader = ManaCrystalReader(save_debug_images=False)
    
    print("Hearthstone Mana Crystal Reader")
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
        print(f"Current Mana: {result['current_mana']}/{result['max_mana']}")
        print(f"Available: {result['current_mana']}, Used: {result['max_mana'] - result['current_mana']}")
    else:
        print(f"Detection failed - Raw text: '{result['raw_text']}'")
        
        # Try alternative preprocessing methods
        print("Trying alternative methods...")
        alt_methods = reader.try_alternative_preprocessing(mana_image)
        
        for method_name, processed_img in alt_methods:
            try:
                # Scale up the alternative processed image
                scale_factor = 4
                h, w = processed_img.shape
                scaled = cv2.resize(processed_img, (w * scale_factor, h * scale_factor))
                
                alt_text = pytesseract.image_to_string(scaled, config=reader.ocr_config).strip()
                alt_text = re.sub(r'\s+', '', alt_text)
                
                match = re.search(r'(\d+)/(\d+)', alt_text)
                if match:
                    current = int(match.group(1))
                    maximum = int(match.group(2))
                    if 0 <= current <= 10 and 0 <= maximum <= 10:
                        print(f"✓ {method_name}: {current}/{maximum}")
                        # Save the successful method
                        cv2.imwrite(f'mana_crystals_analysis_{method_name}.png', scaled)
                        continue
                
                print(f"✗ {method_name}: '{alt_text}'")
                
            except Exception as e:
                print(f"✗ {method_name}: Error - {e}")
    
    # Save analysis results (only if debug images enabled)
    if reader.save_debug_images:
        reader.save_analysis_results(result, mana_image, 'mana_crystals_analysis')

if __name__ == "__main__":
    main()