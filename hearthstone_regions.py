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
        self.expected_radius = 1024  # Keep 1024px radius
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
        has_white = white_percentage > 15
        
        return has_white, white_percentage
    
    def adjust_circle_center_for_visibility(self, original_center, radius, image_height):
        """Adjust circle center so the arc edge is visible in the hand region"""
        if original_center is None:
            return None
        
        center_x, center_y = original_center
        
        # For a 1024px radius with ~250px image height,
        # center should be at y = radius + some buffer below the image
        # This positions the circle so its top edge curves through the hand region
        adjusted_center_y = radius + (image_height * 0.3)  # radius + 30% of image height
        
        return (center_x, adjusted_center_y)
        """Find the circle that the center points lie on using least squares fitting"""
        if len(centers) < 3:
            return None, 0
        
        # Convert to numpy array
        points = np.array(centers, dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]
        
        # Set up least squares system for circle fitting
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
            # Solve the least squares problem
            params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            center_x, center_y, c = params
            
            # Calculate radius
            radius = math.sqrt(center_x**2 + center_y**2 + c)
            
            return (center_x, center_y), radius
            
        except np.linalg.LinAlgError:
            # Fallback to centroid calculation
            center_x = sum(p[0] for p in centers) / len(centers)
            center_y = sum(p[1] for p in centers) / len(centers)
            
            distances = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in centers]
            radius = sum(distances) / len(distances)
            
            return (center_x, center_y), radius
    
    def validate_ten_card_arrangement(self, hexagons):
        """Validate that we have exactly 10 cards in proper arc arrangement"""
        if len(hexagons) != self.expected_card_count:
            print(f"Expected {self.expected_card_count} cards, found {len(hexagons)}")
            return False, None
        
        # Extract center points
        centers = []
        for hexagon in hexagons:
            x, y, w, h = hexagon['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        
        # Sort by x-coordinate (left to right)
        sorted_data = sorted(zip(centers, hexagons), key=lambda item: item[0][0])
        sorted_centers = [item[0] for item in sorted_data]
        sorted_hexagons = [item[1] for item in sorted_data]
        
        # Fit circle to points
        circle_center, circle_radius = self.fit_circle_to_points(sorted_centers)
        
        if circle_center is None:
            print("Failed to fit circle to card centers")
            return False, None
        
        print(f"Circle fit: radius={circle_radius:.1f}px, center=({circle_center[0]:.1f}, {circle_center[1]:.1f})")
        
        # Check if radius is within expected range for 10 cards
        min_radius = self.expected_radius - self.radius_tolerance
        max_radius = self.expected_radius + self.radius_tolerance
        
        if not (min_radius <= circle_radius <= max_radius):
            print(f"Radius {circle_radius:.1f}px outside expected range {min_radius}-{max_radius}px for 10 cards")
            return False, None
        
        # Validate that all points lie on the circle
        valid_count = 0
        distances_from_center = []
        
        for center in sorted_centers:
            distance = math.sqrt((center[0] - circle_center[0])**2 + (center[1] - circle_center[1])**2)
            distances_from_center.append(distance)
            
            # Check if distance is close to radius (within 15% tolerance)
            ratio = distance / circle_radius if circle_radius > 0 else 0
            if 0.85 <= ratio <= 1.15:
                valid_count += 1
        
        # Require at least 8 out of 10 cards to be on the arc
        is_valid_arrangement = valid_count >= 8
        
        # Calculate spacing between consecutive cards
        consecutive_distances = []
        for i in range(len(sorted_centers) - 1):
            dist = math.sqrt((sorted_centers[i+1][0] - sorted_centers[i][0])**2 + 
                           (sorted_centers[i+1][1] - sorted_centers[i][1])**2)
            consecutive_distances.append(dist)
        
        avg_spacing = sum(consecutive_distances) / len(consecutive_distances) if consecutive_distances else 0
        
        analysis = {
            'centers': sorted_centers,
            'hexagons': sorted_hexagons,
            'circle_center': circle_center,
            'circle_radius': circle_radius,
            'is_valid_arc': is_valid_arrangement,
            'valid_points_on_arc': valid_count,
            'distances_from_center': distances_from_center,
            'consecutive_distances': consecutive_distances,
            'avg_spacing': avg_spacing
        }
        
        print(f"Arc validation: {valid_count}/10 cards on proper arc")
        print(f"Average spacing: {avg_spacing:.1f}px")
        
        return is_valid_arrangement, analysis
    
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
                # Sort by white percentage (higher is better for mana cost numbers)
                hexagon_candidates.sort(key=lambda x: x['white_percentage'], reverse=True)
                hexagon_candidates = hexagon_candidates[:10]
                print(f"Filtered down to 10 best candidates based on white percentage")
            else:
                print(f"Warning: Only found {len(hexagon_candidates)} candidates, expected 10")
        
        # Validate the 10-card arrangement
        is_valid, analysis = self.validate_ten_card_arrangement(hexagon_candidates)
        
        if is_valid and analysis:
            # Return the validated and sorted hexagons
            return analysis['hexagons'], blue_mask, analysis
        else:
            # Return unsorted candidates if validation fails
            return hexagon_candidates, blue_mask, analysis
    
    def draw_ten_card_visualization(self, image, hexagons, analysis=None):
        """Draw detailed visualization for 10-card hand detection with all analysis info"""
        result_image = image.copy()
        
        # Draw the arc circle if available
        if analysis and analysis['circle_center'] and analysis['circle_radius']:
            center = (int(analysis['circle_center'][0]), int(analysis['circle_center'][1]))
            radius = int(analysis['circle_radius'])
            cv2.circle(result_image, center, radius, (255, 0, 255), 2)  # Magenta circle
            cv2.circle(result_image, center, 5, (255, 0, 255), -1)  # Larger center point
            
            # Draw radius line to first card for reference
            if hexagons:
                x, y, w, h = hexagons[0]['bbox']
                card_center = (x + w // 2, y + h // 2)
                cv2.line(result_image, center, card_center, (255, 0, 255), 1)
        
        # Draw each detected hexagon with detailed info
        for i, hexagon_data in enumerate(hexagons):
            contour = hexagon_data['contour']
            x, y, w, h = hexagon_data['bbox']
            
            # Use different colors based on validation
            if analysis and analysis.get('is_valid_arc', False):
                border_color = (0, 255, 0)  # Green for valid
                text_color = (0, 255, 0)
            else:
                border_color = (0, 255, 255)  # Yellow for questionable
                text_color = (0, 255, 255)
            
            # Draw contour border (thicker)
            cv2.drawContours(result_image, [contour], -1, border_color, 3)
            
            # Draw bounding rectangle
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Draw center point (larger)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(result_image, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Label with card number (1-10) - larger text
            cv2.putText(result_image, str(i+1), (x-5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
            
            # Add distance from center if available
            if analysis and 'distances_from_center' in analysis and i < len(analysis['distances_from_center']):
                distance = analysis['distances_from_center'][i]
                dist_text = f"{distance:.0f}"
                cv2.putText(result_image, dist_text, (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add comprehensive status information with larger, more visible text
        info_y = 40
        line_height = 35
        
        # Card count
        card_count_text = f"Cards: {len(hexagons)}/10"
        cv2.putText(result_image, card_count_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        info_y += line_height
        
        if analysis:
            # Arc radius info
            if analysis['circle_radius']:
                radius_text = f"Arc Radius: {analysis['circle_radius']:.0f}px"
                cv2.putText(result_image, radius_text, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                info_y += line_height
                
                # Expected vs actual
                expected_text = f"Expected: {self.expected_radius}px (+/-{self.radius_tolerance})"
                cv2.putText(result_image, expected_text, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                info_y += line_height
            
            # Cards on arc info
            if 'valid_points_on_arc' in analysis:
                arc_text = f"Cards on Arc: {analysis['valid_points_on_arc']}/10"
                cv2.putText(result_image, arc_text, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                info_y += line_height
            
            # Average spacing
            if analysis.get('avg_spacing'):
                spacing_text = f"Avg Spacing: {analysis['avg_spacing']:.0f}px"
                cv2.putText(result_image, spacing_text, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                info_y += line_height
            
            # Overall validation status (larger, prominent)
            if len(hexagons) == 10:
                if analysis.get('is_valid_arc', False):
                    status_text = "✓ VALID 10-CARD HAND"
                    status_color = (0, 255, 0)
                else:
                    status_text = "✗ INVALID ARRANGEMENT"
                    status_color = (0, 0, 255)
            else:
                status_text = f"✗ INCOMPLETE ({len(hexagons)}/10)"
                status_color = (0, 165, 255)  # Orange
            
            cv2.putText(result_image, status_text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        return result_image
    
    def analyze_ten_card_hand(self, filename='test_hand.png'):
        """Main function to detect and analyze a 10-card hand"""
        print(f"Loading hand image: {filename}")
        
        if not os.path.exists(filename):
            print(f"Error: {filename} not found!")
            return None
        
        # Load image
        hand_image = self.load_hand_image(filename)
        print(f"Image loaded: {hand_image.shape}")
        
        # Detect 10-card hand
        hexagons, blue_mask, analysis = self.detect_ten_card_hand(hand_image)
        
        print(f"\nDetection Results:")
        print(f"Cards detected: {len(hexagons)}")
        
        if analysis:
            if analysis.get('is_valid_arc'):
                print(f"✓ Valid 10-card arc arrangement")
            else:
                print(f"✗ Invalid arc arrangement")
            
            if analysis.get('circle_radius'):
                print(f"Arc radius: {analysis['circle_radius']:.1f}px")
                print(f"Expected radius: {self.expected_radius}±{self.radius_tolerance}px")
            
            if analysis.get('valid_points_on_arc'):
                print(f"Cards on arc: {analysis['valid_points_on_arc']}/10")
            
            if analysis.get('avg_spacing'):
                print(f"Average card spacing: {analysis['avg_spacing']:.1f}px")
        
        # Print details for each card
        for i, hexagon_data in enumerate(hexagons):
            x, y, w, h = hexagon_data['bbox']
            area = hexagon_data['area']
            white_pct = hexagon_data['white_percentage']
            
            print(f"Card {i+1}: pos=({x},{y}), size=({w}x{h}), "
                  f"area={area}, white={white_pct:.1f}%")
        
        # Draw visualization
        result_image = self.draw_ten_card_visualization(hand_image, hexagons, analysis)
        
        # Save results
        cv2.imwrite('ten_card_hand_detection.png', result_image)
        cv2.imwrite('blue_mask.png', blue_mask)
        
        # Also save with the original naming for compatibility
        cv2.imwrite('hexagon_borders.png', result_image)
        
        print(f"\nSaved results:")
        print(f"  - ten_card_hand_detection.png: 10-card detection visualization")
        print(f"  - hexagon_borders.png: Same visualization (original name)")
        print(f"  - blue_mask.png: Blue color detection mask")
        
        return {
            'card_count': len(hexagons),
            'hexagon_data': hexagons,
            'result_image': result_image,
            'blue_mask': blue_mask,
            'analysis': analysis,
            'is_valid_ten_card_hand': len(hexagons) == 10 and analysis and analysis.get('is_valid_arc', False)
        }

def main():
    """Main function to test 10-card hand detection"""
    reader = HandReader()
    
    print("Hearthstone Hand Reader - 10 Card Detection")
    print("=" * 45)
    
    # Check if test file exists
    if not os.path.exists('test_hand.png'):
        print("Missing file: test_hand.png")
        print("Please run hearthstone_regions.py first to generate this file.")
        return
    
    # Analyze 10-card hand
    result = reader.analyze_ten_card_hand('test_hand.png')
    
    if result:
        print(f"\n" + "="*45)
        print(f"FINAL RESULT:")
        if result['is_valid_ten_card_hand']:
            print(f"✓ Successfully detected valid 10-card hand!")
        else:
            print(f"✗ Could not detect valid 10-card hand")
            print(f"  Cards found: {result['card_count']}/10")
        
        print(f"\nCheck ten_card_hand_detection.png to verify the detection")

if __name__ == "__main__":
    main()