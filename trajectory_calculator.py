import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class TrajectoryCalculator:
    """Calculate ball trajectories and shot recommendations for billiards"""
    
    def __init__(self):
        self.table_friction = 0.02  # Rolling friction coefficient
        self.ball_radius = 11.25    # Standard pool ball radius in mm (scaled)
        self.table_bounds = None
        self.pocket_positions = []
        
        # Physics constants
        self.restitution = 0.95     # Ball-to-ball collision coefficient
        self.wall_restitution = 0.8 # Ball-to-wall collision coefficient
        
        # Game-specific settings
        self.game_rules = {
            '8-ball': {
                'target_groups': ['solids', 'stripes'],
                'winning_ball': '8'
            },
            '9-ball': {
                'target_sequence': True,
                'winning_ball': '9'
            },
            'straight': {
                'any_ball': True
            }
        }
    
    def analyze_shot(self, detections: Dict, game_type: str = '8-ball') -> Dict:
        """
        Analyze the current table state and provide shot recommendations
        
        Args:
            detections: Detected objects from BilliardsDetector
            game_type: Type of billiards game
            
        Returns:
            Dictionary containing shot analysis and recommendations
        """
        analysis = {
            'recommended_shot': None,
            'trajectory': None,
            'difficulty_zones': [],
            'warnings': [],
            'alternative_shots': []
        }
        
        # Update table boundaries from detection
        if detections.get('table'):
            self._update_table_bounds(detections['table'])
        
        # Find cue ball and target balls
        cue_ball = self._find_cue_ball(detections.get('balls', []))
        target_balls = self._find_target_balls(detections.get('balls', []), game_type)
        
        if not cue_ball:
            analysis['warnings'].append("Cue ball not detected")
            return analysis
        
        if not target_balls:
            analysis['warnings'].append("No valid target balls detected")
            return analysis
        
        # Calculate best shot
        best_shot = self._calculate_best_shot(cue_ball, target_balls, detections)
        if best_shot:
            analysis['recommended_shot'] = best_shot
            
            # Calculate trajectory for the recommended shot
            trajectory = self._calculate_trajectory(cue_ball, best_shot)
            analysis['trajectory'] = trajectory
        
        # Calculate difficulty zones
        difficulty_zones = self._calculate_difficulty_zones(cue_ball, target_balls)
        analysis['difficulty_zones'] = difficulty_zones
        
        # Find alternative shots
        alternatives = self._find_alternative_shots(cue_ball, target_balls, best_shot)
        analysis['alternative_shots'] = alternatives[:3]  # Top 3 alternatives
        
        # Check for potential fouls or warnings
        warnings = self._check_shot_warnings(cue_ball, detections.get('balls', []), best_shot)
        analysis['warnings'].extend(warnings)
        
        return analysis
    
    def _update_table_bounds(self, table_info: Dict):
        """Update table boundaries and pocket positions"""
        if 'bbox' in table_info:
            x1, y1, x2, y2 = table_info['bbox']
            self.table_bounds = {
                'left': x1,
                'top': y1,
                'right': x2,
                'bottom': y2,
                'width': x2 - x1,
                'height': y2 - y1
            }
            
            # Estimate pocket positions
            self.pocket_positions = [
                (x1 + 20, y1 + 20),      # Top-left
                (x2 - 20, y1 + 20),      # Top-right
                (x1 + 20, y2 - 20),      # Bottom-left
                (x2 - 20, y2 - 20),      # Bottom-right
                ((x1 + x2) // 2, y1 + 10),  # Top-center
                ((x1 + x2) // 2, y2 - 10),  # Bottom-center
            ]
    
    def _find_cue_ball(self, balls: List[Dict]) -> Optional[Dict]:
        """Find the cue ball from detected balls"""
        for ball in balls:
            if ball.get('type') == 'cue':
                return ball
        
        # Fallback: find whitest ball
        whitest_ball = None
        max_brightness = 0
        
        for ball in balls:
            if 'color' in ball:
                b, g, r = ball['color']
                brightness = (r + g + b) / 3
                if brightness > max_brightness:
                    max_brightness = brightness
                    whitest_ball = ball
        
        return whitest_ball
    
    def _find_target_balls(self, balls: List[Dict], game_type: str) -> List[Dict]:
        """Find valid target balls based on game type"""
        target_balls = []
        
        for ball in balls:
            if ball.get('type') != 'cue':  # Exclude cue ball
                # For now, consider all non-cue balls as targets
                # In a real implementation, this would be game-specific
                target_balls.append(ball)
        
        return target_balls
    
    def _calculate_best_shot(self, cue_ball: Dict, target_balls: List[Dict], detections: Dict) -> Optional[Dict]:
        """Calculate the best shot recommendation"""
        if not target_balls:
            return None
        
        best_shot = None
        highest_score = 0
        
        for target_ball in target_balls:
            # For each target ball, calculate optimal shot
            shot = self._analyze_shot_to_target(cue_ball, target_ball, detections)
            
            if shot and shot.get('score', 0) > highest_score:
                highest_score = shot['score']
                best_shot = shot
        
        return best_shot
    
    def _analyze_shot_to_target(self, cue_ball: Dict, target_ball: Dict, detections: Dict) -> Optional[Dict]:
        """Analyze a shot from cue ball to specific target ball"""
        cue_pos = np.array(cue_ball['center'])
        target_pos = np.array(target_ball['center'])
        
        # Find best pocket for this target ball
        best_pocket, pocket_angle = self._find_best_pocket(target_pos)
        
        if not best_pocket:
            return None
        
        # Calculate required ghost ball position
        ghost_ball_pos = self._calculate_ghost_ball_position(target_pos, best_pocket)
        
        # Calculate aim point (contact point on target ball)
        aim_point = self._calculate_aim_point(cue_pos, target_pos, ghost_ball_pos)
        
        # Calculate shot angle and distance
        shot_vector = target_pos - cue_pos
        shot_distance = np.linalg.norm(shot_vector)
        shot_angle = math.degrees(math.atan2(shot_vector[1], shot_vector[0]))
        
        # Calculate difficulty and success probability
        difficulty = self._calculate_shot_difficulty(cue_pos, target_pos, best_pocket, detections)
        success_prob = max(0.1, 1.0 - difficulty)
        
        # Calculate required power
        power = self._calculate_required_power(shot_distance, difficulty)
        
        # Score this shot (higher is better)
        score = success_prob * (1.0 / max(1.0, difficulty)) * (1.0 / max(1.0, shot_distance / 100))
        
        return {
            'target_ball': target_ball.get('type', 'unknown'),
            'target_position': target_pos.tolist(),
            'ghost_ball_position': ghost_ball_pos.tolist(),
            'aim_point': aim_point.tolist(),
            'pocket': best_pocket,
            'angle': shot_angle,
            'distance': shot_distance,
            'power': power,
            'difficulty': self._get_difficulty_label(difficulty),
            'success_probability': success_prob,
            'score': score
        }
    
    def _find_best_pocket(self, ball_position: np.ndarray) -> Tuple[Optional[Tuple], float]:
        """Find the best pocket for a given ball position"""
        if not self.pocket_positions:
            return None, 0
        
        best_pocket = None
        best_angle = float('inf')
        
        for pocket in self.pocket_positions:
            pocket_pos = np.array(pocket)
            
            # Calculate angle difficulty (straight shots are easier)
            vector_to_pocket = pocket_pos - ball_position
            distance_to_pocket = np.linalg.norm(vector_to_pocket)
            
            # Prefer closer pockets
            angle_difficulty = distance_to_pocket / 100  # Normalize
            
            if angle_difficulty < best_angle:
                best_angle = angle_difficulty
                best_pocket = pocket
        
        return best_pocket, best_angle
    
    def _calculate_ghost_ball_position(self, target_pos: np.ndarray, pocket_pos: Tuple) -> np.ndarray:
        """Calculate where the cue ball should be to pot the target ball"""
        target_to_pocket = np.array(pocket_pos) - target_pos
        target_to_pocket_unit = target_to_pocket / np.linalg.norm(target_to_pocket)
        
        # Ghost ball is one ball diameter away from target ball
        # in the opposite direction of the pocket
        ball_diameter = self.ball_radius * 2
        ghost_ball_pos = target_pos - (target_to_pocket_unit * ball_diameter)
        
        return ghost_ball_pos
    
    def _calculate_aim_point(self, cue_pos: np.ndarray, target_pos: np.ndarray, ghost_pos: np.ndarray) -> np.ndarray:
        """Calculate the exact point to aim at on the target ball"""
        # The aim point is the contact point between cue ball and target ball
        cue_to_ghost = ghost_pos - cue_pos
        cue_to_ghost_unit = cue_to_ghost / np.linalg.norm(cue_to_ghost)
        
        # Contact point is one ball radius away from ghost ball center
        contact_point = ghost_pos - (cue_to_ghost_unit * self.ball_radius)
        
        return contact_point
    
    def _calculate_shot_difficulty(self, cue_pos: np.ndarray, target_pos: np.ndarray, 
                                  pocket_pos: Tuple, detections: Dict) -> float:
        """Calculate shot difficulty (0 = easy, 1 = very hard)"""
        difficulty = 0.0
        
        # Distance factor
        shot_distance = np.linalg.norm(target_pos - cue_pos)
        distance_factor = min(1.0, shot_distance / 300)  # Normalize to table size
        difficulty += distance_factor * 0.3
        
        # Angle factor
        if pocket_pos:
            target_to_pocket = np.array(pocket_pos) - target_pos
            cue_to_target = target_pos - cue_pos
            
            # Calculate angle between shot direction and pocket direction
            angle_rad = self._angle_between_vectors(cue_to_target, target_to_pocket)
            angle_factor = abs(angle_rad) / math.pi  # Normalize to 0-1
            difficulty += angle_factor * 0.4
        
        # Obstruction factor (balls in the way)
        obstruction_factor = self._calculate_obstruction(cue_pos, target_pos, detections.get('balls', []))
        difficulty += obstruction_factor * 0.3
        
        return min(1.0, difficulty)
    
    def _calculate_obstruction(self, start_pos: np.ndarray, end_pos: np.ndarray, balls: List[Dict]) -> float:
        """Calculate how much other balls obstruct the shot path"""
        obstruction = 0.0
        shot_line = end_pos - start_pos
        shot_distance = np.linalg.norm(shot_line)
        
        if shot_distance == 0:
            return 1.0
        
        shot_unit = shot_line / shot_distance
        
        for ball in balls:
            ball_pos = np.array(ball['center'])
            
            # Skip if it's the start or end ball
            if np.linalg.norm(ball_pos - start_pos) < 5 or np.linalg.norm(ball_pos - end_pos) < 5:
                continue
            
            # Calculate distance from ball to shot line
            start_to_ball = ball_pos - start_pos
            projection_length = np.dot(start_to_ball, shot_unit)
            
            # Only consider balls between start and end
            if 0 <= projection_length <= shot_distance:
                projection_point = start_pos + projection_length * shot_unit
                distance_to_line = np.linalg.norm(ball_pos - projection_point)
                
                # If ball is close to the shot line, it's an obstruction
                ball_radius = ball.get('radius', self.ball_radius)
                if distance_to_line < ball_radius * 2.5:  # Buffer zone
                    obstruction += 0.3
        
        return min(1.0, obstruction)
    
    def _calculate_required_power(self, distance: float, difficulty: float) -> float:
        """Calculate required shot power (0.0 to 1.0)"""
        # Base power on distance
        base_power = min(0.8, distance / 200)  # Normalize to reasonable range
        
        # Adjust for difficulty
        power_adjustment = difficulty * 0.2
        
        total_power = base_power + power_adjustment
        return max(0.1, min(1.0, total_power))
    
    def _get_difficulty_label(self, difficulty: float) -> str:
        """Convert difficulty score to label"""
        if difficulty < 0.3:
            return 'easy'
        elif difficulty < 0.6:
            return 'medium'
        else:
            return 'hard'
    
    def _calculate_trajectory(self, cue_ball: Dict, shot_info: Dict) -> Dict:
        """Calculate predicted ball trajectory"""
        trajectory = {
            'cue_ball_path': [],
            'target_ball_path': []
        }
        
        # Simulate cue ball path
        cue_pos = np.array(cue_ball['center'])
        target_pos = np.array(shot_info['target_position'])
        
        # Calculate initial velocity based on power
        power = shot_info.get('power', 0.5)
        max_velocity = 200  # pixels per "time unit"
        initial_velocity = power * max_velocity
        
        # Direction vector
        direction = target_pos - cue_pos
        direction = direction / np.linalg.norm(direction)
        velocity = direction * initial_velocity
        
        # Simulate physics
        position = cue_pos.copy()
        time_steps = 50
        
        for step in range(time_steps):
            # Add current position to path
            trajectory['cue_ball_path'].append(position.tolist())
            
            # Update position
            position += velocity * 0.1  # Time step
            
            # Apply friction
            velocity *= (1 - self.table_friction)
            
            # Check for table boundaries
            if self.table_bounds:
                if position[0] <= self.table_bounds['left'] or position[0] >= self.table_bounds['right']:
                    velocity[0] *= -self.wall_restitution
                if position[1] <= self.table_bounds['top'] or position[1] >= self.table_bounds['bottom']:
                    velocity[1] *= -self.wall_restitution
            
            # Stop if velocity is too low
            if np.linalg.norm(velocity) < 1:
                break
        
        return trajectory
    
    def _calculate_difficulty_zones(self, cue_ball: Dict, target_balls: List[Dict]) -> List[Dict]:
        """Calculate difficulty zones around the table"""
        zones = []
        
        cue_pos = np.array(cue_ball['center'])
        
        for target_ball in target_balls[:3]:  # Limit to first 3 balls
            target_pos = np.array(target_ball['center'])
            distance = np.linalg.norm(target_pos - cue_pos)
            
            # Create difficulty zone
            if distance < 100:
                difficulty = 'easy'
                radius = 20
            elif distance < 200:
                difficulty = 'medium'
                radius = 25
            else:
                difficulty = 'hard'
                radius = 30
            
            zones.append({
                'center': target_pos.tolist(),
                'radius': radius,
                'difficulty': difficulty
            })
        
        return zones
    
    def _find_alternative_shots(self, cue_ball: Dict, target_balls: List[Dict], best_shot: Optional[Dict]) -> List[Dict]:
        """Find alternative shot options"""
        alternatives = []
        
        for target_ball in target_balls:
            shot = self._analyze_shot_to_target(cue_ball, target_ball, {'balls': target_balls})
            
            # Skip if this is the best shot
            if best_shot and shot and shot.get('target_ball') == best_shot.get('target_ball'):
                continue
            
            if shot:
                alternatives.append(shot)
        
        # Sort by score
        alternatives.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return alternatives
    
    def _check_shot_warnings(self, cue_ball: Dict, all_balls: List[Dict], shot_info: Optional[Dict]) -> List[str]:
        """Check for potential warnings or fouls"""
        warnings = []
        
        if not shot_info:
            return warnings
        
        # Check for potential scratches
        cue_pos = np.array(cue_ball['center'])
        target_pos = np.array(shot_info['target_position'])
        
        # Check if cue ball might go in pocket
        for pocket in self.pocket_positions:
            pocket_pos = np.array(pocket)
            distance_to_pocket = np.linalg.norm(cue_pos - pocket_pos)
            
            if distance_to_pocket < 50:  # Very close to pocket
                warnings.append("Risk of scratch - cue ball near pocket")
                break
        
        # Check for difficult angle
        if shot_info.get('difficulty') == 'hard':
            warnings.append("Difficult shot - consider alternative")
        
        # Check for potential ball clusters
        obstruction = self._calculate_obstruction(cue_pos, target_pos, all_balls)
        if obstruction > 0.5:
            warnings.append("Path partially blocked - difficult shot")
        
        return warnings
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        return math.acos(cos_angle)
