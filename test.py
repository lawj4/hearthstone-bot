import cv2
import numpy as np
import pytesseract
import os
import re
from hand_white_mask import HandReader
import itertools

class OCRParamFinder:
    """Class to automatically find optimal OCR parameters by testing combinations"""
    
    def __init__(self):
        self.hand_reader = HandReader()
        
        # PARAMETER RANGES TO TEST
        self.param_ranges = {
            'white_threshold': [200, 210, 220, 230, 240, 250],
            'scale_factor': [6, 8, 10, 12],
            'gaussian_blur': [(1,1), (3,3), (5,5), (7,7)],
            'gaussian_sigma': [0.5, 1.0, 1.5, 2.0],
            'morph_kernel_size': [1, 2, 3, 4, 5],
            'padding': [10, 15, 20, 25]
        }
        
        # OCR configurations to try
        self.ocr_configs = [
            ('PSM8_OEM3', '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'),
            ('PSM8_OEM1', '--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789'),
            ('PSM7_OEM3', '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'),
            ('PSM6_OEM3', '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'),
            ('PSM13_OEM3', '--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789'),
        ]
        
        self.target_count = 8  # Looking for 8 valid numbers
        self.successful_params = []
    
    def extract_crystal_with_padding(self, hand_image, x, y, w, h, padding):
        """Extract crystal region with specified padding"""
        height, width = hand_image.shape[:2]
        
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        return hand_image[y_start:y_end, x_start:x_end]
    
    def preprocess_crystal(self, crystal_image, params):
        """Preprocess crystal with given parameters"""
        # Convert to grayscale
        if len(crystal_image.shape) == 3:
            gray = cv2.cvtColor(crystal_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crystal_image
        
        # Apply Gaussian blur
        blur_kernel = params['gaussian_blur']
        blur_sigma = params['gaussian_sigma']
        blurred = cv2.GaussianBlur(gray, blur_kernel, blur_sigma)
        
        # White threshold
        _, white_mask = cv2.threshold(blurred, params['white_threshold'], 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel_size = params['morph_kernel_size']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Scale up
        scale_factor = params['scale_factor']
        height, width = closed.shape
        scaled = cv2.resize(closed, (width * scale_factor, height * scale_factor), 
                          interpolation=cv2.INTER_CUBIC)
        
        # Final cleanup
        kernel_scaled = np.ones((2,2), np.uint8)
        final_cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel_scaled)
        
        return final_cleaned, cv2.bitwise_not(final_cleaned)
    
    def test_ocr_on_crystal(self, crystal_image, params):
        """Test OCR on a single crystal with given parameters"""
        # Preprocess
        processed, processed_inv = self.preprocess_crystal(crystal_image, params)
        
        # Try OCR on both versions
        for img in [processed, processed_inv]:
            for config_name, config in self.ocr_configs:
                try:
                    text = pytesseract.image_to_string(img, config=config).strip()
                    text = re.sub(r'[^0-9]', '', text)
                    
                    if text and text.isdigit():
                        number = int(text)
                        if 0 <= number <= 10:
                            return number
                except:
                    continue
        
        return None
    
    def test_parameter_combination(self, params):
        """Test a specific parameter combination on all crystals"""
        try:
            # Get crystals
            white_mask, crystal_boxes = self.hand_reader.analyze_white_mask()
            if not crystal_boxes:
                return 0, []
            
            hand_image = self.hand_reader.load_hand_image()
            detected_numbers = []
            
            # Test each crystal
            for i, (x, y, w, h) in enumerate(crystal_boxes):
                crystal_region = self.extract_crystal_with_padding(
                    hand_image, x, y, w, h, params['padding']
                )
                
                number = self.test_ocr_on_crystal(crystal_region, params)
                if number is not None:
                    detected_numbers.append(number)
            
            return len(detected_numbers), detected_numbers
            
        except Exception as e:
            return 0, []
    
    def smart_parameter_search(self):
        """Smart parameter search - test most promising combinations first"""
        print("OCR Parameter Finder - Searching for 8 detected numbers")
        print("=" * 60)
        
        # Start with reasonable defaults and vary one parameter at a time
        base_params = {
            'white_threshold': 240,
            'scale_factor': 8,
            'gaussian_blur': (3,3),
            'gaussian_sigma': 1.0,
            'morph_kernel_size': 2,
            'padding': 15
        }
        
        print("Testing base parameters...")
        count, numbers = self.test_parameter_combination(base_params)
        print(f"Base params: {count} numbers detected: {numbers}")
        
        if count >= self.target_count:
            print(f"🎯 SUCCESS! Base parameters work!")
            return base_params
        
        # Test variations of each parameter
        for param_name in self.param_ranges:
            print(f"\nTesting variations of {param_name}...")
            
            for value in self.param_ranges[param_name]:
                test_params = base_params.copy()
                test_params[param_name] = value
                
                count, numbers = self.test_parameter_combination(test_params)
                print(f"  {param_name}={value}: {count} numbers: {numbers}")
                
                if count >= self.target_count:
                    print(f"🎯 SUCCESS! Found working parameters!")
                    self.save_successful_params(test_params, count, numbers)
                    return test_params
                elif count > self.test_parameter_combination(base_params)[0]:
                    # This is better than base, update base
                    base_params[param_name] = value
                    print(f"  ↑ Updated base {param_name} to {value}")
        
        print(f"⚠️  No single parameter change reached {self.target_count}. Trying combinations...")
        return self.brute_force_search(base_params)
    
    def brute_force_search(self, best_params):
        """Try multiple parameter combinations if smart search fails"""
        print("\nTrying parameter combinations...")
        
        # Test most promising combinations
        promising_combos = [
            # Lower threshold + more blur
            {'white_threshold': 220, 'gaussian_blur': (5,5), 'morph_kernel_size': 3},
            {'white_threshold': 210, 'gaussian_blur': (7,7), 'morph_kernel_size': 4},
            
            # Higher scaling + processing
            {'scale_factor': 12, 'morph_kernel_size': 4, 'padding': 20},
            {'scale_factor': 10, 'gaussian_blur': (5,5), 'morph_kernel_size': 3},
            
            # Aggressive morphology
            {'morph_kernel_size': 5, 'gaussian_blur': (5,5), 'white_threshold': 220},
        ]
        
        for i, combo in enumerate(promising_combos):
            test_params = best_params.copy()
            test_params.update(combo)
            
            count, numbers = self.test_parameter_combination(test_params)
            print(f"Combo {i+1}: {count} numbers: {numbers}")
            print(f"  Params: {combo}")
            
            if count >= self.target_count:
                print(f"🎯 SUCCESS! Found working combination!")
                self.save_successful_params(test_params, count, numbers)
                return test_params
        
        print(f"❌ Could not find parameters that detect {self.target_count} numbers")
        return best_params
    
    def save_successful_params(self, params, count, numbers):
        """Save successful parameters to a file"""
        with open('successful_ocr_params.txt', 'w') as f:
            f.write(f"Successful OCR Parameters - Detected {count} numbers: {numbers}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Copy these values to hand_ocr_reader.py:\n\n")
            f.write("self.ocr_params = {\n")
            for key, value in params.items():
                f.write(f"    '{key}': {value},\n")
            f.write("}\n")
        
        print(f"💾 Saved successful parameters to 'successful_ocr_params.txt'")
    
    def run_search(self):
        """Main function to run the parameter search"""
        if not os.path.exists('test_hand.png'):
            print("Error: test_hand.png not found!")
            return
        
        print(f"🎯 Target: Find parameters that detect {self.target_count} numbers")
        print(f"📊 Will test {len(self.param_ranges)} different parameter types")
        print()
        
        # Suppress hand_reader output during search
        import sys
        from contextlib import redirect_stdout
        import io
        
        # Run smart search
        with redirect_stdout(io.StringIO()):
            optimal_params = self.smart_parameter_search()
        
        print(f"\n🏁 FINAL RESULT:")
        print("=" * 40)
        
        # Test final params with output
        final_count, final_numbers = self.test_parameter_combination(optimal_params)
        print(f"Detected {final_count} numbers: {final_numbers}")
        print(f"\nOptimal parameters:")
        for key, value in optimal_params.items():
            print(f"  {key}: {value}")

def main():
    """Run the OCR parameter finder"""
    finder = OCRParamFinder()
    finder.run_search()

if __name__ == "__main__":
    main()