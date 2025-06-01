import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

# Try to import cv2, fall back to PIL if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class OverlayRenderer:
    """Renders visual overlays for billiards aiming assistance"""
    
    def __init__(self):
        self.opacity = 0.7
        self.colors = {
            'aim_line': (0, 255, 0),      # Green
            'trajectory': (255, 0, 0),     # Red
            'target_circle': (0, 255, 255), # Cyan
            'success_zone': (0, 255, 0),   # Green
            'warning_zone': (255, 255, 0), # Yellow
            'danger_zone': (255, 0, 0),    # Red
            'cue_ball': (255, 255, 255),   # White
            'ghost_ball': (200, 200, 200), # Light gray
        }
        self.line_thickness = 2
        self.circle_thickness = 2
    
    def set_opacity(self, opacity):
        """Set overlay opacity (0.0 to 1.0)"""
        self.opacity = max(0.0, min(1.0, opacity))
    
    def render_overlays(self, frame, detections, analysis):
        """
        Render all overlays on the frame
        
        Args:
            frame: Input video frame
            detections: Detected objects from BilliardsDetector
            analysis: Shot analysis from TrajectoryCalculator
        
        Returns:
            Frame with overlays rendered
        """
        overlay = frame.copy()
        
        # Render table boundaries
        if detections.get('table'):
            overlay = self._render_table_overlay(overlay, detections['table'])
        
        # Render detected balls
        if detections.get('balls'):
            overlay = self._render_ball_overlays(overlay, detections['balls'])
        
        # Render aiming assistance
        if analysis.get('recommended_shot'):
            overlay = self._render_aiming_overlay(overlay, analysis['recommended_shot'], detections)
        
        # Render trajectory prediction
        if analysis.get('trajectory'):
            overlay = self._render_trajectory_overlay(overlay, analysis['trajectory'])
        
        # Render cue stick overlay
        if detections.get('cue_stick'):
            overlay = self._render_cue_stick_overlay(overlay, detections['cue_stick'])
        
        # Render shot difficulty indicators
        if analysis.get('difficulty_zones'):
            overlay = self._render_difficulty_zones(overlay, analysis['difficulty_zones'])
        
        # Blend overlay with original frame
        result = cv2.addWeighted(frame, 1 - self.opacity, overlay, self.opacity, 0)
        
        # Add informational text overlays
        result = self._render_text_overlays(result, analysis, detections)
        
        return result
    
    def _render_table_overlay(self, frame, table_info):
        """Render table boundary and pocket indicators"""
        if 'bbox' in table_info:
            x1, y1, x2, y2 = table_info['bbox']
            
            # Draw table boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw corner pockets (estimated positions)
            pocket_radius = 15
            pockets = [
                (x1 + 20, y1 + 20),  # Top-left
                (x2 - 20, y1 + 20),  # Top-right
                (x1 + 20, y2 - 20),  # Bottom-left
                (x2 - 20, y2 - 20),  # Bottom-right
                ((x1 + x2) // 2, y1 + 10),  # Top-center
                ((x1 + x2) // 2, y2 - 10),  # Bottom-center
            ]
            
            for pocket in pockets:
                cv2.circle(frame, pocket, pocket_radius, (0, 0, 0), -1)
                cv2.circle(frame, pocket, pocket_radius, (255, 255, 255), 2)
        
        return frame
    
    def _render_ball_overlays(self, frame, balls):
        """Render overlays for detected balls"""
        for ball in balls:
            center = ball['center']
            
            # Draw ball circle
            if 'radius' in ball:
                radius = ball['radius']
            else:
                radius = 12  # Default radius
            
            # Color code based on ball type
            if ball.get('type') == 'cue':
                color = self.colors['cue_ball']
                cv2.circle(frame, center, radius + 3, color, 2)
            else:
                # Regular ball
                cv2.circle(frame, center, radius + 2, (255, 255, 255), 1)
            
            # Add ball number/type label
            if ball.get('type'):
                label = str(ball['type']).replace('ball_', '')
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                label_x = center[0] - label_size[0] // 2
                label_y = center[1] + label_size[1] // 2
                
                cv2.putText(frame, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _render_aiming_overlay(self, frame, shot_info, detections):
        """Render aiming line and target indicators"""
        cue_ball = self._find_cue_ball(detections['balls'])
        target_ball = self._find_target_ball(detections['balls'], shot_info.get('target_ball'))
        
        if cue_ball and target_ball:
            cue_center = cue_ball['center']
            target_center = target_ball['center']
            
            # Calculate aiming line
            if shot_info.get('aim_point'):
                aim_point = shot_info['aim_point']
            else:
                aim_point = target_center
            
            # Draw aiming line from cue ball to aim point
            cv2.line(frame, cue_center, aim_point, self.colors['aim_line'], self.line_thickness)
            
            # Draw target circle around target ball
            if 'radius' in target_ball:
                target_radius = target_ball['radius'] + 5
            else:
                target_radius = 17
            
            cv2.circle(frame, target_center, target_radius, self.colors['target_circle'], 2)
            
            # Draw ghost ball position (where cue ball should contact target ball)
            if shot_info.get('ghost_ball_position'):
                ghost_pos = shot_info['ghost_ball_position']
                cv2.circle(frame, ghost_pos, 12, self.colors['ghost_ball'], 2)
                cv2.line(frame, cue_center, ghost_pos, self.colors['aim_line'], 1)
            
            # Draw power indicator
            if shot_info.get('power'):
                power = shot_info['power']
                self._draw_power_indicator(frame, cue_center, power)
        
        return frame
    
    def _render_trajectory_overlay(self, frame, trajectory_info):
        """Render predicted ball trajectories"""
        if 'cue_ball_path' in trajectory_info:
            path = trajectory_info['cue_ball_path']
            
            # Draw trajectory line
            for i in range(len(path) - 1):
                pt1 = tuple(map(int, path[i]))
                pt2 = tuple(map(int, path[i + 1]))
                
                # Fade the line as it gets further from the start
                alpha = max(0.3, 1.0 - (i / len(path)))
                color = tuple(int(c * alpha) for c in self.colors['trajectory'])
                
                cv2.line(frame, pt1, pt2, color, 1)
            
            # Draw trajectory dots for better visualization
            for i, point in enumerate(path[::3]):  # Every 3rd point
                if i < 10:  # Limit number of dots
                    pt = tuple(map(int, point))
                    cv2.circle(frame, pt, 2, self.colors['trajectory'], -1)
        
        # Draw target ball trajectory if available
        if 'target_ball_path' in trajectory_info:
            target_path = trajectory_info['target_ball_path']
            for i in range(len(target_path) - 1):
                pt1 = tuple(map(int, target_path[i]))
                pt2 = tuple(map(int, target_path[i + 1]))
                cv2.line(frame, pt1, pt2, (255, 165, 0), 1)  # Orange
        
        return frame
    
    def _render_cue_stick_overlay(self, frame, cue_stick_info):
        """Render cue stick guidance overlay"""
        if 'start' in cue_stick_info and 'end' in cue_stick_info:
            start = cue_stick_info['start']
            end = cue_stick_info['end']
            
            # Draw extended cue stick line
            cv2.line(frame, start, end, (139, 69, 19), 3)  # Brown color
            
            # Draw cue tip indicator
            cv2.circle(frame, end, 5, (255, 255, 0), -1)
        
        return frame
    
    def _render_difficulty_zones(self, frame, difficulty_zones):
        """Render shot difficulty visualization zones"""
        for zone in difficulty_zones:
            center = zone.get('center')
            radius = zone.get('radius', 30)
            difficulty = zone.get('difficulty', 'medium')
            
            if center:
                if difficulty == 'easy':
                    color = self.colors['success_zone']
                elif difficulty == 'medium':
                    color = self.colors['warning_zone']
                else:
                    color = self.colors['danger_zone']
                
                # Draw difficulty zone circle
                cv2.circle(frame, center, radius, color, 1)
        
        return frame
    
    def _render_text_overlays(self, frame, analysis, detections):
        """Render informational text overlays"""
        y_offset = 30
        
        # Shot recommendation
        if analysis.get('recommended_shot'):
            shot = analysis['recommended_shot']
            
            # Success probability
            if 'success_probability' in shot:
                prob_text = f"Success Rate: {shot['success_probability']:.1%}"
                self._draw_text_with_background(frame, prob_text, (10, y_offset))
                y_offset += 25
            
            # Difficulty
            if 'difficulty' in shot:
                diff_text = f"Difficulty: {shot['difficulty'].title()}"
                self._draw_text_with_background(frame, diff_text, (10, y_offset))
                y_offset += 25
            
            # Recommended power
            if 'power' in shot:
                power_text = f"Power: {shot['power']:.0%}"
                self._draw_text_with_background(frame, power_text, (10, y_offset))
                y_offset += 25
        
        # Ball count
        ball_count = len(detections.get('balls', []))
        count_text = f"Balls Detected: {ball_count}"
        self._draw_text_with_background(frame, count_text, (10, y_offset))
        
        return frame
    
    def _draw_text_with_background(self, frame, text, position, font_scale=0.6):
        """Draw text with a semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_height - 5),
                     (position[0] + text_width + 5, position[1] + baseline + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness)
    
    def _draw_power_indicator(self, frame, center, power):
        """Draw power indicator near cue ball"""
        # Draw power bar
        bar_length = int(50 * power)
        bar_start = (center[0] + 20, center[1] - 10)
        bar_end = (center[0] + 20 + bar_length, center[1] - 10)
        
        # Color based on power level
        if power < 0.3:
            color = (0, 255, 0)  # Green (soft)
        elif power < 0.7:
            color = (0, 255, 255)  # Yellow (medium)
        else:
            color = (0, 0, 255)  # Red (hard)
        
        cv2.line(frame, bar_start, bar_end, color, 4)
        
        # Draw power percentage
        power_text = f"{power:.0%}"
        cv2.putText(frame, power_text, (center[0] + 25, center[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _find_cue_ball(self, balls):
        """Find the cue ball from detected balls"""
        for ball in balls:
            if ball.get('type') == 'cue':
                return ball
        
        # If no cue ball found, assume the whitest ball is the cue ball
        whitest_ball = None
        max_whiteness = 0
        
        for ball in balls:
            if 'color' in ball:
                b, g, r = ball['color']
                whiteness = (r + g + b) / 3
                if whiteness > max_whiteness:
                    max_whiteness = whiteness
                    whitest_ball = ball
        
        return whitest_ball
    
    def _find_target_ball(self, balls, target_type):
        """Find the target ball from detected balls"""
        for ball in balls:
            if ball.get('type') == target_type:
                return ball
        
        # If target not found, return the first non-cue ball
        for ball in balls:
            if ball.get('type') != 'cue':
                return ball
        
        return None
