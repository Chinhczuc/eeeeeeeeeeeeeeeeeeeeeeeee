import numpy as np
import os

# Try to import cv2 and ultralytics, fall back gracefully if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - using fallback detection")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - using basic detection")

class BilliardsDetector:
    """YOLOv8-based billiards object detection system"""
    
    def __init__(self):
        self.confidence_threshold = 0.5
        self.model = None
        self.ball_colors = {
            'cue': (255, 255, 255),      # White
            '1': (255, 255, 0),          # Yellow (solid)
            '2': (0, 0, 255),            # Blue (solid)
            '3': (255, 0, 0),            # Red (solid)
            '4': (128, 0, 128),          # Purple (solid)
            '5': (255, 165, 0),          # Orange (solid)
            '6': (0, 128, 0),            # Green (solid)
            '7': (128, 0, 0),            # Maroon (solid)
            '8': (0, 0, 0),              # Black
            '9': (255, 255, 0),          # Yellow (stripe)
            '10': (0, 0, 255),           # Blue (stripe)
            '11': (255, 0, 0),           # Red (stripe)
            '12': (128, 0, 128),         # Purple (stripe)
            '13': (255, 165, 0),         # Orange (stripe)
            '14': (0, 128, 0),           # Green (stripe)
            '15': (128, 0, 0),           # Maroon (stripe)
        }
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize YOLOv8 model for billiards detection"""
        if not YOLO_AVAILABLE:
            print("YOLO not available - using basic detection methods")
            self.model = None
            return
            
        try:
            # Try to load a pre-trained YOLOv8 model
            # In a real implementation, you would train this on billiards data
            self.model = YOLO('yolov8n.pt')  # Nano version for speed
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            # Fallback to color-based detection
            self.model = None
    
    def set_confidence(self, confidence):
        """Set detection confidence threshold"""
        self.confidence_threshold = confidence
    
    def detect(self, frame):
        """
        Detect billiards objects in the frame
        Returns dictionary with detected objects
        """
        detections = {
            'table': None,
            'balls': [],
            'cue_stick': None,
            'pockets': []
        }
        
        if self.model is not None:
            detections = self._yolo_detection(frame)
        else:
            detections = self._color_based_detection(frame)
        
        return detections
    
    def _yolo_detection(self, frame):
        """Use YOLOv8 for object detection"""
        detections = {
            'table': None,
            'balls': [],
            'cue_stick': None,
            'pockets': []
        }
        
        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Map class IDs to object types (this would be based on your training)
                        obj_info = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': (center_x, center_y),
                            'confidence': confidence
                        }
                        
                        # For demo purposes, treat all detections as balls
                        # In a real implementation, you'd have specific classes
                        if confidence > self.confidence_threshold:
                            obj_info['type'] = f'ball_{len(detections["balls"])}'
                            obj_info['color'] = self._estimate_ball_color(frame, center_x, center_y)
                            detections['balls'].append(obj_info)
        
        except Exception as e:
            print(f"YOLOv8 detection error: {e}")
            # Fallback to color-based detection
            return self._color_based_detection(frame)
        
        return detections
    
    def _color_based_detection(self, frame):
        """Fallback color-based detection for billiards objects"""
        detections = {
            'table': None,
            'balls': [],
            'cue_stick': None,
            'pockets': []
        }
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect table (green felt)
        table_detection = self._detect_table(hsv)
        if table_detection:
            detections['table'] = table_detection
        
        # Detect balls using circle detection
        balls = self._detect_balls_hough(gray, frame)
        detections['balls'] = balls
        
        # Detect cue stick (elongated wooden object)
        cue_stick = self._detect_cue_stick(frame)
        if cue_stick:
            detections['cue_stick'] = cue_stick
        
        return detections
    
    def _detect_table(self, hsv_frame):
        """Detect billiards table using green color detection"""
        # Define range for green color (table felt)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # Create mask for green areas
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        # Find largest contour (should be the table)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only consider it a table if it's large enough
            if area > 50000:  # Minimum table area
                x, y, w, h = cv2.boundingRect(largest_contour)
                return {
                    'bbox': [x, y, x + w, y + h],
                    'contour': largest_contour,
                    'area': area,
                    'confidence': min(1.0, area / 100000)  # Normalize confidence
                }
        
        return None
    
    def _detect_balls_hough(self, gray_frame, color_frame):
        """Detect balls using Hough Circle Transform"""
        balls = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,  # Minimum distance between circle centers
            param1=50,   # Upper threshold for edge detection
            param2=30,   # Accumulator threshold for center detection
            minRadius=8, # Minimum circle radius
            maxRadius=25 # Maximum circle radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Estimate ball color
                ball_color = self._estimate_ball_color(color_frame, x, y)
                ball_type = self._classify_ball_by_color(ball_color)
                
                ball_info = {
                    'center': (x, y),
                    'radius': r,
                    'bbox': [x - r, y - r, x + r, y + r],
                    'color': ball_color,
                    'type': ball_type,
                    'confidence': 0.8  # Static confidence for color-based detection
                }
                balls.append(ball_info)
        
        return balls
    
    def _detect_cue_stick(self, frame):
        """Detect cue stick using line detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Find the longest line that could be a cue stick
            longest_line = None
            max_length = 0
            
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                
                # Calculate line endpoints
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Calculate line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length > max_length:
                    max_length = length
                    longest_line = {
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'angle': theta,
                        'confidence': min(1.0, length / 500)
                    }
            
            return longest_line
        
        return None
    
    def _estimate_ball_color(self, frame, x, y, radius=10):
        """Estimate the dominant color of a ball at given coordinates"""
        # Extract region around the ball center
        y1, y2 = max(0, y - radius), min(frame.shape[0], y + radius)
        x1, x2 = max(0, x - radius), min(frame.shape[1], x + radius)
        
        region = frame[y1:y2, x1:x2]
        
        if region.size > 0:
            # Calculate mean color
            mean_color = np.mean(region.reshape(-1, 3), axis=0)
            return tuple(map(int, mean_color))
        
        return (128, 128, 128)  # Gray default
    
    def _classify_ball_by_color(self, color):
        """Classify ball type based on its color"""
        b, g, r = color
        
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return 'cue'  # White cue ball
        elif r < 50 and g < 50 and b < 50:
            return '8'    # Black 8-ball
        elif r > g and r > b:
            return 'red'  # Red ball
        elif b > r and b > g:
            return 'blue' # Blue ball
        elif g > r and g > b:
            return 'green' # Green ball
        elif r > 150 and g > 150:
            return 'yellow' # Yellow ball
        else:
            return 'unknown'
    
    def get_detection_info(self):
        """Get information about the current detection setup"""
        return {
            'model_type': 'YOLOv8' if self.model else 'Color-based',
            'confidence_threshold': self.confidence_threshold,
            'supported_objects': ['table', 'balls', 'cue_stick', 'pockets']
        }
