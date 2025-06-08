import cv2
import numpy as np
import os
import math

class HandReader:
    """Simplified class to detect exactly 10 blue hexagonal mana cost borders in Hearthstone hand"""
    
    def __init__(self):
        # Blue color ranges in HSV for mana cost detection
        self.blue_ranges = [
            (np.array([100, 80, 80]), np.array([120, 255, 255])),  # Standard blue
            (np.array([90, 60, 60]), np.array([130, 255, 255])),   # Wider range
            (np.array([105, 100, 100]), np.array([115, 255, 255]))  # Narrow precise range
        ]
        
        # Fixed parameters for 10-card hand
        self.expected_card_count = 10
        self.expected_radius = 1224  # Keep 1024px radius
        self.radius_tolerance = 200  # Allow ±200px variation
        self.spacing_tolerance = 0.3  # Allow 30% spacing variation
    
    def load_hand_image(self, filename='test_hand.png'):
        """Load the hand region image"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Hand image not found: {filename}")
        
        hand_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        return hand_image
    
    def create_blue_mask(self, image):
        """Create mask for blue hexagons using multiple color ranges"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Combine multiple blue ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for blue_lower, blue_upper in self.blue_ranges:
            mask = cv2.inRange(hsv, blue_lower, blue_upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def has_white_center(self, image, contour):
        """Check if the contour has white content in its center (mana cost number)"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center region (40% of the bounding box)
        center_size = 0.4
        center_w = int(w * center_size)
        center_h = int(h * center_size)
        center_x = x + (w - center_w) // 2
        center_y = y + (h - center_h) // 2
        
        # Extract center region
        center_region = image[center_y:center_y+center_h, center_x:center_x+center_w]
        
        if center_region.size == 0:
            return False, 0
        
        # Convert to grayscale if needed
        if len(center_region.shape) == 3:
            gray_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_center = center_region
        
        # Find white/light pixels
        _, white_mask = cv2.threshold(gray_center, 200, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of white pixels
        white_pixels = np.sum(white_mask == 255)
        total_pixels = white_mask.size
        white_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Require at least 15% white pixels in center
        has_white = white_percentage > 10
        
        return has_white, white_percentage
    
    def analyze_card_triangles_with_center(self, hexagons, fixed_center, fixed_radius):
        """Analyze triangles formed by each pair of adjacent cards with the circle center"""
        if len(hexagons) < 2:
            return None
        
        # Get card centers sorted by x-coordinate
        card_centers = []
        for hexagon in hexagons:
            x, y, w, h = hexagon['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            card_centers.append((center_x, center_y))
        
        # Sort by x-coordinate (left to right)
        card_centers.sort(key=lambda p: p[0])
        
        # Analyze each pair of adjacent cards with the center
        triangle_analyses = []
        
        for i in range(len(card_centers) - 1):
            card1 = card_centers[i]
            card2 = card_centers[i + 1]
            center = fixed_center
            
            # Calculate the three sides of the triangle
            # Side 1: card1 to card2 (chord)
            chord_length = math.sqrt((card2[0] - card1[0])**2 + (card2[1] - card1[1])**2)
            
            # Side 2: center to card1 (radius)
            radius1 = math.sqrt((card1[0] - center[0])**2 + (card1[1] - center[1])**2)
            
            # Side 3: center to card2 (radius)
            radius2 = math.sqrt((card2[0] - center[0])**2 + (card2[1] - center[1])**2)
            
            # Calculate angles of the triangle
            # Angle at center (between the two radii)
            # Using law of cosines: c² = a² + b² - 2ab*cos(C)
            if radius1 > 0 and radius2 > 0:
                cos_center_angle = (radius1**2 + radius2**2 - chord_length**2) / (2 * radius1 * radius2)
                cos_center_angle = max(-1, min(1, cos_center_angle))  # Clamp to valid range
                center_angle = math.acos(cos_center_angle)
                center_angle_degrees = math.degrees(center_angle)
            else:
                center_angle_degrees = 0
            
            # Angles at the cards (should be equal for isosceles triangle)
            if chord_length > 0:
                cos_card1_angle = (radius1**2 + chord_length**2 - radius2**2) / (2 * radius1 * chord_length)
                cos_card1_angle = max(-1, min(1, cos_card1_angle))
                card1_angle_degrees = math.degrees(math.acos(cos_card1_angle))
                
                cos_card2_angle = (radius2**2 + chord_length**2 - radius1**2) / (2 * radius2 * chord_length)
                cos_card2_angle = max(-1, min(1, cos_card2_angle))
                card2_angle_degrees = math.degrees(math.acos(cos_card2_angle))
            else:
                card1_angle_degrees = 0
                card2_angle_degrees = 0
            
            # Check if this forms a proper isosceles triangle
            # Radii should be approximately equal to fixed_radius
            radius_tolerance = fixed_radius * 0.05  # 5% tolerance
            radius1_good = abs(radius1 - fixed_radius) <= radius_tolerance
            radius2_good = abs(radius2 - fixed_radius) <= radius_tolerance
            
            # Card angles should be approximately equal (isosceles triangle)
            angle_tolerance = 5.0  # 5 degrees tolerance
            angles_equal = abs(card1_angle_degrees - card2_angle_degrees) <= angle_tolerance
            
            # Triangle is "good" if both radii are correct and angles are equal
            is_good_triangle = radius1_good and radius2_good and angles_equal
            
            triangle_analysis = {
                'pair': (i+1, i+2),  # 1-based card numbers
                'card1': card1,
                'card2': card2,
                'chord_length': chord_length,
                'radius1': radius1,
                'radius2': radius2,
                'center_angle_degrees': center_angle_degrees,
                'card1_angle_degrees': card1_angle_degrees,
                'card2_angle_degrees': card2_angle_degrees,
                'radius1_deviation': abs(radius1 - fixed_radius),
                'radius2_deviation': abs(radius2 - fixed_radius),
                'angle_difference': abs(card1_angle_degrees - card2_angle_degrees),
                'radius1_good': radius1_good,
                'radius2_good': radius2_good,
                'angles_equal': angles_equal,
                'is_good_triangle': is_good_triangle
            }
            
            triangle_analyses.append(triangle_analysis)
        
        # Calculate overall statistics
        good_triangles = sum(1 for t in triangle_analyses if t['is_good_triangle'])
        total_triangles = len(triangle_analyses)
        
        # Check if all chord lengths are similar (equidistant cards)
        chord_lengths = [t['chord_length'] for t in triangle_analyses]
        if chord_lengths:
            avg_chord = sum(chord_lengths) / len(chord_lengths)
            chord_deviations = [abs(c - avg_chord) for c in chord_lengths]
            max_chord_deviation = max(chord_deviations)
            chord_tolerance = avg_chord * 0.15  # 15% tolerance
            chords_equal = max_chord_deviation <= chord_tolerance
        else:
            avg_chord = 0
            max_chord_deviation = 0
            chords_equal = False
        
        # Check if all center angles are similar (regular polygon)
        center_angles = [t['center_angle_degrees'] for t in triangle_analyses]
        if center_angles:
            avg_center_angle = sum(center_angles) / len(center_angles)
            angle_deviations = [abs(a - avg_center_angle) for a in center_angles]
            max_angle_deviation = max(angle_deviations)
            angle_tolerance = 2.0  # 2 degrees tolerance
            angles_regular = max_angle_deviation <= angle_tolerance
        else:
            avg_center_angle = 0
            max_angle_deviation = 0
            angles_regular = False
        
        return {
            'triangle_analyses': triangle_analyses,
            'good_triangles': good_triangles,
            'total_triangles': total_triangles,
            'all_triangles_good': good_triangles == total_triangles and total_triangles > 0,
            'chord_lengths': chord_lengths,
            'avg_chord_length': avg_chord,
            'max_chord_deviation': max_chord_deviation,
            'chords_equal': chords_equal,
            'center_angles': center_angles,
            'avg_center_angle': avg_center_angle,
            'max_angle_deviation': max_angle_deviation,
            'angles_regular': angles_regular,
            'perfect_arrangement': (good_triangles == total_triangles and 
                                  chords_equal and angles_regular and total_triangles > 0)
        }
    
    def detect_ten_card_hand(self, image):
        """Detect exactly 10 blue hexagon mana costs in hand"""
        # Create blue mask
        blue_mask = self.create_blue_mask(image)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hexagon_candidates = []
        
        # Find all potential hexagon candidates
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by reasonable hexagon size for mana costs
            if 200 < area < 1500:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (should be roughly square/circular)
                aspect_ratio = w / h if h > 0 else 0
                if 0.6 < aspect_ratio < 1.4:
                    # Check for white center (mana cost number)
                    has_white, white_pct = self.has_white_center(image, contour)
                    
                    if has_white:
                        hexagon_data = {
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'white_percentage': white_pct
                        }
                        hexagon_candidates.append(hexagon_data)
        
        print(f"Found {len(hexagon_candidates)} hexagon candidates")
        
        # If we don't have exactly 10, try to filter or find missing ones
        if len(hexagon_candidates) != 10:
            if len(hexagon_candidates) > 10:
                # Too many candidates - keep the 10 best ones
                hexagon_candidates.sort(key=lambda x: x['white_percentage'], reverse=True)
                hexagon_candidates = hexagon_candidates[:10]
                print(f"Filtered down to 10 best candidates based on white percentage")
            else:
                print(f"Warning: Only found {len(hexagon_candidates)} candidates, expected 10")
        
        return hexagon_candidates, blue_mask
    
    def draw_with_triangle_analysis(self, image, hexagons, triangle_analysis=None):
        """Draw visualization with triangle analysis"""
        result_image = image.copy()
        image_height, image_width = image.shape[:2]
        
        # Calculate fixed circle position
        fixed_center_x = image_width // 2  # 512px
        fixed_center_y = image_height + 500  # 750px below top
        fixed_radius = 1024
        fixed_center = (fixed_center_x, fixed_center_y)
        
        # Draw the fixed circle
        cv2.circle(result_image, fixed_center, fixed_radius, (0, 255, 255), 6)    # Cyan
        cv2.circle(result_image, fixed_center, fixed_radius, (255, 0, 255), 4)    # Magenta
        cv2.circle(result_image, fixed_center, fixed_radius, (255, 255, 0), 2)    # Yellow
        
        # Draw triangles and analysis
        if triangle_analysis and triangle_analysis['triangle_analyses']:
            for i, tri in enumerate(triangle_analysis['triangle_analyses']):
                card1, card2 = tri['card1'], tri['card2']
                
                # Choose color based on triangle quality
                if tri['is_good_triangle']:
                    triangle_color = (0, 255, 0)  # Green for good triangles
                    line_thickness = 3
                else:
                    triangle_color = (0, 0, 255)  # Red for bad triangles
                    line_thickness = 2
                
                # Draw the triangle: center -> card1 -> card2 -> center
                cv2.line(result_image, fixed_center, card1, triangle_color, line_thickness)
                cv2.line(result_image, fixed_center, card2, triangle_color, line_thickness)
                cv2.line(result_image, card1, card2, triangle_color, line_thickness)
                
                # Draw chord length at midpoint
                mid_x = (card1[0] + card2[0]) // 2
                mid_y = (card1[1] + card2[1]) // 2
                chord_text = f"{tri['chord_length']:.0f}"
                cv2.putText(result_image, chord_text, (mid_x-20, mid_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw each detected hexagon
        for i, hexagon_data in enumerate(hexagons):
            contour = hexagon_data['contour']
            x, y, w, h = hexagon_data['bbox']
            
            # Color based on overall analysis
            if triangle_analysis and triangle_analysis.get('perfect_arrangement', False):
                border_color = (0, 255, 0)  # Green for perfect
                text_color = (0, 255, 0)
            else:
                border_color = (0, 255, 255)  # Yellow for imperfect
                text_color = (0, 255, 255)
            
            # Draw contour and bounding box
            cv2.drawContours(result_image, [contour], -1, border_color, 3)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(result_image, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Label with card number
            cv2.putText(result_image, str(i+1), (x-5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
            
            # Show distance from center
            distance = math.sqrt((center_x - fixed_center_x)**2 + (center_y - fixed_center_y)**2)
            dist_text = f"{distance:.0f}"
            cv2.putText(result_image, dist_text, (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add status information
        info_lines = [
            f"Cards: {len(hexagons)}/10",
            f"Fixed Arc: {fixed_radius}px radius"
        ]
        
        if triangle_analysis:
            good_tri = triangle_analysis['good_triangles']
            total_tri = triangle_analysis['total_triangles']
            
            if triangle_analysis['all_triangles_good']:
                info_lines.append(f"✓ All triangles good ({good_tri}/{total_tri})")
            else:
                info_lines.append(f"✗ Bad triangles ({good_tri}/{total_tri})")
            
            if triangle_analysis['chords_equal']:
                info_lines.append(f"✓ Cards equidistant ({triangle_analysis['avg_chord_length']:.0f}px)")
            else:
                info_lines.append(f"✗ Cards not equidistant (dev: {triangle_analysis['max_chord_deviation']:.0f}px)")
            
            if triangle_analysis['angles_regular']:
                info_lines.append(f"✓ Regular angles ({triangle_analysis['avg_center_angle']:.1f}°)")
            else:
                info_lines.append(f"✗ Irregular angles (dev: {triangle_analysis['max_angle_deviation']:.1f}°)")
            
            if triangle_analysis['perfect_arrangement']:
                info_lines.append("✓ PERFECT ARRANGEMENT")
            else:
                info_lines.append("✗ IMPERFECT ARRANGEMENT")
        
        # Draw info with background
        for i, line in enumerate(info_lines):
            y_pos = 30 + (i * 35)
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(result_image, (5, y_pos-25), (text_size[0]+15, y_pos+5), (0, 0, 0), -1)
            cv2.putText(result_image, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result_image
    
    def analyze_ten_card_hand(self, filename='test_hand.png'):
        """Main function to detect and analyze a 10-card hand with triangle analysis"""
        print(f"Loading hand image: {filename}")
        
        if not os.path.exists(filename):
            print(f"Error: {filename} not found!")
            return None
        
        # Load image
        hand_image = self.load_hand_image(filename)
        print(f"Image loaded: {hand_image.shape}")
        image_height, image_width = hand_image.shape[:2]
        
        # Detect cards
        hexagons, blue_mask = self.detect_ten_card_hand(hand_image)
        
        print(f"\nDetection Results:")
        print(f"Cards detected: {len(hexagons)}")
        
        # Set up fixed center for analysis
        fixed_center_x = image_width // 2
        fixed_center_y = image_height + 500
        fixed_center = (fixed_center_x, fixed_center_y)
        fixed_radius = 1024
        
        # Perform triangle analysis
        triangle_analysis = self.analyze_card_triangles_with_center(hexagons, fixed_center, fixed_radius)
        
        if triangle_analysis:
            print(f"\nTriangle Analysis Results:")
            print(f"  Good triangles: {triangle_analysis['good_triangles']}/{triangle_analysis['total_triangles']}")
            print(f"  All triangles good: {'✓' if triangle_analysis['all_triangles_good'] else '✗'}")
            print(f"  Cards equidistant: {'✓' if triangle_analysis['chords_equal'] else '✗'}")
            print(f"  Regular angles: {'✓' if triangle_analysis['angles_regular'] else '✗'}")
            print(f"  Perfect arrangement: {'✓' if triangle_analysis['perfect_arrangement'] else '✗'}")
            
            print(f"\nDetailed Triangle Analysis:")
            for tri in triangle_analysis['triangle_analyses']:
                status = "✓" if tri['is_good_triangle'] else "✗"
                print(f"  Cards {tri['pair']}: {status} "
                      f"chord={tri['chord_length']:.0f}px, "
                      f"r1={tri['radius1']:.0f}px, r2={tri['radius2']:.0f}px, "
                      f"center_angle={tri['center_angle_degrees']:.1f}°, "
                      f"card_angles=({tri['card1_angle_degrees']:.1f}°, {tri['card2_angle_degrees']:.1f}°)")
            
            print(f"\nSummary Statistics:")
            print(f"  Average chord length: {triangle_analysis['avg_chord_length']:.1f}px")
            print(f"  Max chord deviation: {triangle_analysis['max_chord_deviation']:.1f}px")
            print(f"  Average center angle: {triangle_analysis['avg_center_angle']:.1f}°")
            print(f"  Max angle deviation: {triangle_analysis['max_angle_deviation']:.1f}°")
        
        # Draw visualization
        result_image = self.draw_with_triangle_analysis(hand_image, hexagons, triangle_analysis)
        
        # Save results
        cv2.imwrite('hexagon_borders.png', result_image)
        cv2.imwrite('blue_mask.png', blue_mask)
        
        print(f"\nSaved results:")
        print(f"  - hexagon_borders.png: Card detection with triangle analysis")
        print(f"  - blue_mask.png: Blue color detection mask")
        
        return {
            'card_count': len(hexagons),
            'hexagon_data': hexagons,
            'triangle_analysis': triangle_analysis,
            'result_image': result_image,
            'blue_mask': blue_mask,
            'perfect_arrangement': triangle_analysis and triangle_analysis.get('perfect_arrangement', False)
        }

def main():
    """Main function to test triangle analysis"""
    reader = HandReader()
    
    print("Hearthstone Hand Reader - Triangle Analysis")
    print("=" * 50)
    
    # Check if test file exists
    if not os.path.exists('test_hand.png'):
        print("Missing file: test_hand.png")
        print("Please run hearthstone_regions.py first to generate this file.")
        return
    
    # Analyze hand with triangle analysis
    result = reader.analyze_ten_card_hand('test_hand.png')
    
    if result:
        print(f"\n" + "="*50)
        print(f"FINAL RESULT:")
        if result['perfect_arrangement']:
            print(f"✓ PERFECT 10-CARD ARRANGEMENT DETECTED!")
            print(f"  - All triangles are proper isosceles triangles")
            print(f"  - Cards are equidistant from each other")
            print(f"  - All cards are on the 1024px radius circle")
            print(f"  - Center angles form a regular polygon")
        else:
            print(f"✗ Arrangement is not perfect")
            print(f"  Cards found: {result['card_count']}/10")
            if result['triangle_analysis']:
                ta = result['triangle_analysis']
                print(f"  Good triangles: {ta['good_triangles']}/{ta['total_triangles']}")
                if not ta['chords_equal']:
                    print(f"  Cards not equidistant")
                if not ta['angles_regular']:
                    print(f"  Angles not regular")
        
        print(f"\nCheck hexagon_borders.png to see the triangle analysis visualization")

if __name__ == "__main__":
    main()