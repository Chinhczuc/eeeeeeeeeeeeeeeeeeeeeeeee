import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
import math

class PerformanceMonitor:
    """Monitor application performance metrics"""
    
    def __init__(self):
        self.frame_times = []
        self.detection_times = []
        self.max_samples = 30  # Keep last 30 samples
    
    def log_frame_time(self, frame_time: float):
        """Log frame processing time"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
    
    def log_detection_time(self, detection_time: float):
        """Log detection processing time"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > self.max_samples:
            self.detection_times.pop(0)
    
    def get_average_fps(self) -> float:
        """Calculate average FPS"""
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_average_detection_time(self) -> float:
        """Calculate average detection time"""
        if not self.detection_times:
            return 0.0
        return sum(self.detection_times) / len(self.detection_times)

class GeometryUtils:
    """Utility functions for geometric calculations"""
    
    @staticmethod
    def distance_point_to_line(point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        # Distance formula
        distance = abs(a * x0 + b * y0 + c) / math.sqrt(a*a + b*b)
        return distance
    
    @staticmethod
    def point_in_circle(point: Tuple[float, float], 
                       circle_center: Tuple[float, float], 
                       radius: float) -> bool:
        """Check if point is inside circle"""
        dx = point[0] - circle_center[0]
        dy = point[1] - circle_center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        return distance <= radius
    
    @staticmethod
    def line_circle_intersection(line_start: Tuple[float, float],
                                line_end: Tuple[float, float],
                                circle_center: Tuple[float, float],
                                radius: float) -> List[Tuple[float, float]]:
        """Find intersection points between line and circle"""
        x1, y1 = line_start
        x2, y2 = line_end
        cx, cy = circle_center
        
        # Convert to vector form
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy
        
        # Quadratic equation coefficients
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - radius*radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return []  # No intersection
        
        discriminant = math.sqrt(discriminant)
        
        if a == 0:
            return []  # Line has zero length
        
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        intersections = []
        
        # Check if intersection points are on the line segment
        for t in [t1, t2]:
            if 0 <= t <= 1:
                ix = x1 + t * dx
                iy = y1 + t * dy
                intersections.append((ix, iy))
        
        return intersections
    
    @staticmethod
    def angle_between_points(p1: Tuple[float, float], 
                           p2: Tuple[float, float], 
                           p3: Tuple[float, float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        # Vectors from p2 to p1 and p2 to p3
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angle using dot product
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag_v1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
        mag_v2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0
        
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to avoid numerical errors
        
        return math.acos(cos_angle)

class ColorUtils:
    """Utility functions for color processing and analysis"""
    
    @staticmethod
    def bgr_to_hsv(bgr_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert BGR color to HSV"""
        b, g, r = bgr_color
        # Create a 1x1 pixel image for conversion
        pixel = np.uint8([[[b, g, r]]])
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        return tuple(hsv_pixel[0, 0])
    
    @staticmethod
    def color_distance(color1: Tuple[int, int, int], 
                      color2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between two colors in RGB space"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
    
    @staticmethod
    def is_color_similar(color1: Tuple[int, int, int], 
                        color2: Tuple[int, int, int], 
                        threshold: float = 50.0) -> bool:
        """Check if two colors are similar within threshold"""
        return ColorUtils.color_distance(color1, color2) < threshold
    
    @staticmethod
    def get_dominant_color(image_region: np.ndarray) -> Tuple[int, int, int]:
        """Get dominant color in an image region"""
        if image_region.size == 0:
            return (128, 128, 128)  # Gray default
        
        # Reshape image to list of pixels
        pixels = image_region.reshape(-1, 3)
        
        # Calculate mean color
        mean_color = np.mean(pixels, axis=0)
        return tuple(map(int, mean_color))

class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [0, 2π) range"""
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle
    
    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """Convert degrees to radians"""
        return degrees * math.pi / 180.0
    
    @staticmethod
    def radians_to_degrees(radians: float) -> float:
        """Convert radians to degrees"""
        return radians * 180.0 / math.pi
    
    @staticmethod
    def interpolate(start: float, end: float, t: float) -> float:
        """Linear interpolation between start and end"""
        return start + (end - start) * MathUtils.clamp(t, 0.0, 1.0)
    
    @staticmethod
    def smooth_step(t: float) -> float:
        """Smooth step function for smooth transitions"""
        t = MathUtils.clamp(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def is_valid_point(point: Tuple[float, float], 
                      image_width: int, 
                      image_height: int) -> bool:
        """Check if point is within image bounds"""
        x, y = point
        return 0 <= x < image_width and 0 <= y < image_height
    
    @staticmethod
    def is_valid_bbox(bbox: List[int], 
                     image_width: int, 
                     image_height: int) -> bool:
        """Check if bounding box is valid"""
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 >= image_width or y2 >= image_height:
            return False
        
        # Check that it's a valid rectangle
        if x1 >= x2 or y1 >= y2:
            return False
        
        return True
    
    @staticmethod
    def sanitize_detection(detection: Dict, 
                          image_width: int, 
                          image_height: int) -> Optional[Dict]:
        """Sanitize and validate detection data"""
        if not isinstance(detection, dict):
            return None
        
        # Check center point
        if 'center' in detection:
            if not ValidationUtils.is_valid_point(detection['center'], image_width, image_height):
                return None
        
        # Check bounding box
        if 'bbox' in detection:
            if not ValidationUtils.is_valid_bbox(detection['bbox'], image_width, image_height):
                return None
        
        # Check confidence
        if 'confidence' in detection:
            confidence = detection['confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                detection['confidence'] = 0.5  # Default confidence
        
        return detection

class ConfigManager:
    """Manage application configuration and settings"""
    
    def __init__(self):
        self.default_config = {
            'detection': {
                'confidence_threshold': 0.5,
                'max_balls': 16,
                'ball_radius_range': (8, 25)
            },
            'overlay': {
                'opacity': 0.7,
                'line_thickness': 2,
                'colors': {
                    'aim_line': (0, 255, 0),
                    'trajectory': (255, 0, 0),
                    'target_circle': (0, 255, 255)
                }
            },
            'physics': {
                'table_friction': 0.02,
                'ball_restitution': 0.95,
                'wall_restitution': 0.8
            },
            'camera': {
                'width': 1280,
                'height': 720,
                'fps': 30
            }
        }
        self.config = self.default_config.copy()
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = self.default_config.copy()

# Global instances
performance_monitor = PerformanceMonitor()
config_manager = ConfigManager()

# Helper functions for common operations
def safe_normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Safely normalize a vector, returning zero vector if magnitude is zero"""
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.zeros_like(vector)
    return vector / magnitude

def create_rotation_matrix(angle_radians: float) -> np.ndarray:
    """Create 2D rotation matrix"""
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])

def rotate_point(point: Tuple[float, float], 
                angle_radians: float, 
                center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
    """Rotate point around center by given angle"""
    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]
    
    # Rotate
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    
    # Translate back
    return (new_x + center[0], new_y + center[1])

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
