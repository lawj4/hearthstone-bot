import cv2
import time
import os
from datetime import datetime
import traceback
import pytesseract
import numpy as np
import re

# Import all the modules
from screenshot_capture import ScreenCapture
from hearthstone_regions import HearthstoneRegions
from hand_ocr_reader import HandOCRReader
from ally_ocr_reader import AllyBoardReader
from mana_reader import ManaCrystalReader
from hero_health_ocr_reader import HeroHealthOCRReader
import utils

class HearthstoneMonitor:
    """Main monitoring class that takes screenshots and analyzes game state"""
    
    def __init__(self, buffer_seconds=utils.BUFFER):
        self.buffer_seconds = buffer_seconds
        self.capturer = ScreenCapture()
        self.regions = HearthstoneRegions()
        self.hand_reader = HandOCRReader()
        self.ally_reader = AllyBoardReader()
        self.mana_reader = ManaCrystalReader(save_debug_images=False)
        self.health_reader = HeroHealthOCRReader()
        
        # Set debug images to False for continuous monitoring
        utils.DEBUG_IMAGES = False
    
    def take_screenshot_and_extract_regions(self):
        """Take screenshot and extract all game regions"""
        try:
            screenshot = self.capturer.capture_color()
            regions = self.regions.save_regions_as_images(screenshot, 'preprocess', hand_in_color=True)
            return True, regions
        except Exception as e:
            return False, None
    
    def analyze_hand_cards(self):
        """Analyze hand cards and return mana costs"""
        try:
            hand_results = self.hand_reader.analyze_all_crystals()
            if hand_results:
                successful_reads = [r for r in hand_results if r['success']]
                mana_costs = [r['number'] for r in successful_reads]
                return mana_costs
            return []
        except Exception:
            return []
    
    def analyze_ally_board(self):
        """Analyze ally board and return positions"""
        try:
            combined_mask, crystal_boxes = self.ally_reader.analyze_color_mask()
            return len(crystal_boxes) if crystal_boxes else 0
        except Exception:
            return 0
    
    def analyze_mana_crystals(self):
        """Analyze current mana crystals"""
        try:
            result, _ = self.mana_reader.analyze_mana_crystals()
            if result and result['success']:
                return result['current_mana'], result['max_mana']
            return None, None
        except Exception:
            return None, None
    
    def read_health_ocr(self, health_image):
        """Read health value from health region using OCR"""
        try:
            # Preprocess for OCR
            if len(health_image.shape) == 3:
                gray = cv2.cvtColor(health_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = health_image
            
            # Scale up for better OCR
            height, width = gray.shape
            scaled = cv2.resize(gray, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
            
            # Threshold to get clean text
            _, binary = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
            
            # OCR with number-only whitelist
            config = '--psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(binary, config=config).strip()
            
            # Extract numbers only
            numbers = re.findall(r'\d+', text)
            if numbers:
                return int(numbers[0])
            return None
        except Exception:
            return None
    
    def analyze_health_values(self):
        """Analyze ally and enemy health using hero health OCR reader"""
        try:
            # Use the dedicated health OCR reader
            results = self.health_reader.analyze_both_health_regions()
            
            ally_health = None
            enemy_health = None
            
            # Extract enemy health
            if results.get('enemy'):
                successful_enemy = [r for r in results['enemy'] if r['success']]
                if successful_enemy:
                    enemy_health = max(successful_enemy, key=lambda x: int(x['method'].split('conf')[1]) if 'conf' in x['method'] else 0)['number']
            
            # Extract ally health
            if results.get('ally'):
                successful_ally = [r for r in results['ally'] if r['success']]
                if successful_ally:
                    ally_health = max(successful_ally, key=lambda x: int(x['method'].split('conf')[1]) if 'conf' in x['method'] else 0)['number']
            
            return ally_health, enemy_health
        except Exception:
            return None, None
    
    def run_single_analysis(self):
        """Run a single analysis cycle"""
        # Take screenshot and extract regions
        screenshot_success, regions = self.take_screenshot_and_extract_regions()
        if not screenshot_success:
            return False
        
        # Analyze each component
        hand_costs = self.analyze_hand_cards()
        ally_count = self.analyze_ally_board()
        
        # Analyze mana crystals - THIS WAS MISSING!
        current_mana, max_mana = self.analyze_mana_crystals()
        
        # Analyze health values
        ally_health, enemy_health = self.analyze_health_values()
        
        # Format output
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Build output parts
        parts = [timestamp]
        
        # Mana
        if current_mana is not None and max_mana is not None:
            parts.append(f"Mana:{current_mana}/{max_mana}")
        else:
            parts.append("Mana:?/?")
        
        # Hand cards
        if hand_costs:
            costs_str = ','.join(map(str, hand_costs))
            parts.append(f"Hand:[{costs_str}]")
        else:
            parts.append("Hand:[]")
        
        # Ally board
        parts.append(f"Allies:{ally_count}")
        
        # Health values
        if ally_health is not None:
            parts.append(f"MyHP:{ally_health}")
        else:
            parts.append("MyHP:?")
            
        if enemy_health is not None:
            parts.append(f"EnemyHP:{enemy_health}")
        else:
            parts.append("EnemyHP:?")
        
        # Print single line
        print(" | ".join(parts))
        
        return True
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        try:
            while True:
                self.run_single_analysis()
                time.sleep(self.buffer_seconds)
        except KeyboardInterrupt:
            print("\nStopped")
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main function with configurable buffer time"""
    import sys
    
    buffer_seconds = 5
    if len(sys.argv) > 1:
        try:
            buffer_seconds = float(sys.argv[1])
            if buffer_seconds < 1:
                buffer_seconds = 1
        except ValueError:
            pass
    
    monitor = HearthstoneMonitor(buffer_seconds)
    
    # Print header
    print("Time | Mana | Hand | Allies | MyHP | EnemyHP")
    print("-" * 50)
    
    # Run monitoring
    monitor.run_continuous_monitoring()

if __name__ == "__main__":
    main()