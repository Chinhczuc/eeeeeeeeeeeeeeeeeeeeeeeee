"""
Standalone Billiards AI Assistant - No external dependencies needed
"""

import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import math
import random
import json
import time
from threading import Thread

class BilliardsAI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Billiards Assistant v1.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Game state
        self.balls = []
        self.cue_ball = None
        self.target_ball = None
        self.recommended_shot = None
        self.game_running = False
        
        # Table dimensions
        self.table_width = 800
        self.table_height = 400
        self.table_x = 50
        self.table_y = 100
        
        # Initialize UI
        self.create_ui()
        self.create_demo_table()
        
    def create_ui(self):
        """Create the user interface"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎱 AI Billiards Assistant", 
            font=("Arial", 24, "bold"),
            bg='#2c3e50', 
            fg='#ecf0f1'
        )
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#34495e', width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Controls title
        tk.Label(
            left_panel, 
            text="⚙️ Controls", 
            font=("Arial", 16, "bold"),
            bg='#34495e', 
            fg='#ecf0f1'
        ).pack(pady=10)
        
        # Game controls
        controls_frame = tk.Frame(left_panel, bg='#34495e')
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_btn = tk.Button(
            controls_frame,
            text="▶️ Start Analysis",
            command=self.start_analysis,
            bg='#27ae60',
            fg='white',
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            pady=10
        )
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = tk.Button(
            controls_frame,
            text="⏹️ Stop Analysis", 
            command=self.stop_analysis,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            pady=10,
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        self.new_game_btn = tk.Button(
            controls_frame,
            text="🎯 New Game",
            command=self.new_game,
            bg='#3498db',
            fg='white',
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            pady=10
        )
        self.new_game_btn.pack(fill=tk.X, pady=5)
        
        # Game type selection
        tk.Label(
            left_panel,
            text="Game Type:",
            bg='#34495e',
            fg='#ecf0f1',
            font=("Arial", 12, "bold")
        ).pack(pady=(20, 5))
        
        self.game_type = tk.StringVar(value="8-ball")
        game_combo = ttk.Combobox(
            left_panel,
            textvariable=self.game_type,
            values=["8-ball", "9-ball", "Straight Pool", "Snooker"],
            state="readonly"
        )
        game_combo.pack(pady=5, padx=10, fill=tk.X)
        
        # Analysis display
        tk.Label(
            left_panel,
            text="📊 Shot Analysis:",
            bg='#34495e',
            fg='#ecf0f1',
            font=("Arial", 12, "bold")
        ).pack(pady=(20, 5))
        
        self.analysis_text = tk.Text(
            left_panel,
            height=15,
            bg='#2c3e50',
            fg='#ecf0f1',
            font=("Courier", 10),
            relief=tk.FLAT
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Right panel - Table view
        right_panel = tk.Frame(main_frame, bg='#2c3e50')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Table canvas
        self.canvas = Canvas(
            right_panel,
            width=900,
            height=500,
            bg='#27ae60',  # Green felt
            relief=tk.RAISED,
            borderwidth=2
        )
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
    def create_demo_table(self):
        """Create a demo billiards table with balls"""
        # Draw table borders
        self.canvas.create_rectangle(
            self.table_x, self.table_y,
            self.table_x + self.table_width, self.table_y + self.table_height,
            outline="#8B4513", width=8, fill="#27ae60"
        )
        
        # Draw pockets
        pocket_radius = 20
        pockets = [
            (self.table_x, self.table_y),  # Top-left
            (self.table_x + self.table_width, self.table_y),  # Top-right
            (self.table_x, self.table_y + self.table_height),  # Bottom-left
            (self.table_x + self.table_width, self.table_y + self.table_height),  # Bottom-right
            (self.table_x + self.table_width//2, self.table_y),  # Top-center
            (self.table_x + self.table_width//2, self.table_y + self.table_height),  # Bottom-center
        ]
        
        for pocket in pockets:
            self.canvas.create_oval(
                pocket[0] - pocket_radius, pocket[1] - pocket_radius,
                pocket[0] + pocket_radius, pocket[1] + pocket_radius,
                fill="black", outline="#8B4513", width=3
            )
        
        # Generate random ball positions
        self.generate_balls()
        self.draw_balls()
        
    def generate_balls(self):
        """Generate random ball positions"""
        self.balls = []
        
        # Ball types and colors
        ball_types = [
            ("cue", "#FFFFFF"),     # White cue ball
            ("1", "#FFFF00"),       # Yellow
            ("2", "#0000FF"),       # Blue
            ("3", "#FF0000"),       # Red
            ("4", "#800080"),       # Purple
            ("5", "#FFA500"),       # Orange
            ("6", "#008000"),       # Green
            ("7", "#800000"),       # Maroon
            ("8", "#000000"),       # Black
        ]
        
        for ball_type, color in ball_types:
            # Random position within table bounds
            x = random.randint(
                self.table_x + 30, 
                self.table_x + self.table_width - 30
            )
            y = random.randint(
                self.table_y + 30, 
                self.table_y + self.table_height - 30
            )
            
            ball = {
                "type": ball_type,
                "color": color,
                "x": x,
                "y": y,
                "radius": 12
            }
            
            self.balls.append(ball)
            
            if ball_type == "cue":
                self.cue_ball = ball
                
    def draw_balls(self):
        """Draw all balls on the canvas"""
        # Clear existing balls
        self.canvas.delete("ball")
        self.canvas.delete("label")
        
        for ball in self.balls:
            x, y = ball["x"], ball["y"]
            r = ball["radius"]
            
            # Draw ball
            ball_id = self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=ball["color"],
                outline="black" if ball["type"] == "cue" else "white",
                width=3 if ball["type"] == "cue" else 2,
                tags="ball"
            )
            
            # Draw ball number/type
            if ball["type"] != "cue":
                self.canvas.create_text(
                    x, y,
                    text=ball["type"],
                    fill="white" if ball["color"] != "#FFFF00" else "black",
                    font=("Arial", 8, "bold"),
                    tags="label"
                )
                
    def start_analysis(self):
        """Start the AI analysis"""
        self.game_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start analysis thread
        analysis_thread = Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
    def stop_analysis(self):
        """Stop the AI analysis"""
        self.game_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
    def run_analysis(self):
        """Run continuous AI analysis"""
        while self.game_running:
            self.analyze_current_state()
            self.update_display()
            time.sleep(1)  # Update every second
            
    def analyze_current_state(self):
        """Analyze current table state and recommend shot"""
        if not self.cue_ball:
            return
            
        # Find best target ball
        best_shot = None
        best_score = 0
        
        for ball in self.balls:
            if ball["type"] == "cue":
                continue
                
            # Calculate shot difficulty and success probability
            distance = self.calculate_distance(
                self.cue_ball["x"], self.cue_ball["y"],
                ball["x"], ball["y"]
            )
            
            # Simple scoring: closer balls are easier
            score = max(0, 300 - distance) / 300
            
            # Add angle factor (prefer straight shots)
            angle_factor = self.calculate_angle_difficulty(self.cue_ball, ball)
            score *= (1 - angle_factor * 0.3)
            
            if score > best_score:
                best_score = score
                best_shot = {
                    "target": ball,
                    "distance": distance,
                    "angle": self.calculate_angle(self.cue_ball, ball),
                    "success_rate": score,
                    "difficulty": "Easy" if score > 0.7 else "Medium" if score > 0.4 else "Hard",
                    "power": min(1.0, distance / 200)
                }
                
        self.recommended_shot = best_shot
        self.target_ball = best_shot["target"] if best_shot else None
        
    def calculate_distance(self, x1, y1, x2, y2):
        """Calculate distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
    def calculate_angle(self, ball1, ball2):
        """Calculate angle between two balls"""
        dx = ball2["x"] - ball1["x"]
        dy = ball2["y"] - ball1["y"]
        return math.degrees(math.atan2(dy, dx))
        
    def calculate_angle_difficulty(self, cue_ball, target_ball):
        """Calculate angle difficulty (0 = easy, 1 = hard)"""
        # Find nearest pocket
        pockets = [
            (self.table_x, self.table_y),
            (self.table_x + self.table_width, self.table_y),
            (self.table_x, self.table_y + self.table_height),
            (self.table_x + self.table_width, self.table_y + self.table_height),
        ]
        
        min_angle_diff = float('inf')
        for pocket in pockets:
            # Angle from cue to target
            angle1 = math.atan2(
                target_ball["y"] - cue_ball["y"],
                target_ball["x"] - cue_ball["x"]
            )
            # Angle from target to pocket
            angle2 = math.atan2(
                pocket[1] - target_ball["y"],
                pocket[0] - target_ball["x"]
            )
            
            angle_diff = abs(angle1 - angle2)
            min_angle_diff = min(min_angle_diff, angle_diff)
            
        return min_angle_diff / math.pi
        
    def update_display(self):
        """Update the analysis display"""
        self.root.after(0, self._update_display_ui)
        
    def _update_display_ui(self):
        """Update UI elements (must run in main thread)"""
        # Clear previous overlays
        self.canvas.delete("overlay")
        
        # Draw aiming line if we have a recommended shot
        if self.recommended_shot and self.target_ball:
            # Draw aiming line
            self.canvas.create_line(
                self.cue_ball["x"], self.cue_ball["y"],
                self.target_ball["x"], self.target_ball["y"],
                fill="#00FF00", width=3, tags="overlay"
            )
            
            # Highlight target ball
            r = self.target_ball["radius"] + 8
            self.canvas.create_oval(
                self.target_ball["x"] - r, self.target_ball["y"] - r,
                self.target_ball["x"] + r, self.target_ball["y"] + r,
                outline="#00FFFF", width=3, tags="overlay"
            )
            
        # Update analysis text
        self.analysis_text.delete(1.0, tk.END)
        
        if self.recommended_shot:
            shot = self.recommended_shot
            analysis = f"""
🎯 RECOMMENDED SHOT
==================
Target Ball: {shot['target']['type']}
Distance: {shot['distance']:.1f} pixels
Angle: {shot['angle']:.1f}°
Success Rate: {shot['success_rate']:.1%}
Difficulty: {shot['difficulty']}
Power: {shot['power']:.1%}

📊 GAME STATISTICS
==================
Game Type: {self.game_type.get()}
Balls on Table: {len(self.balls)}
Analysis: Active

💡 TIPS
==================
• Aim for the highlighted ball
• Use {shot['power']:.0%} power
• Success rate: {shot['success_rate']:.1%}
• Watch for obstacles
            """
        else:
            analysis = """
❌ NO SHOT ANALYSIS
==================
• Click 'Start Analysis' to begin
• Move balls by clicking on them
• Try 'New Game' for fresh setup
            """
            
        self.analysis_text.insert(tk.END, analysis)
        
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        # Find clicked ball
        for ball in self.balls:
            distance = self.calculate_distance(event.x, event.y, ball["x"], ball["y"])
            if distance <= ball["radius"]:
                # Move ball to new random position
                ball["x"] = random.randint(
                    self.table_x + 30, 
                    self.table_x + self.table_width - 30
                )
                ball["y"] = random.randint(
                    self.table_y + 30, 
                    self.table_y + self.table_height - 30
                )
                self.draw_balls()
                break
                
    def on_mouse_move(self, event):
        """Handle mouse movement for aiming preview"""
        if not self.game_running or not self.cue_ball:
            return
            
        # Show aiming preview line
        self.canvas.delete("preview")
        self.canvas.create_line(
            self.cue_ball["x"], self.cue_ball["y"],
            event.x, event.y,
            fill="#FFFF00", width=1, dash=(5, 5), tags="preview"
        )
        
    def new_game(self):
        """Start a new game"""
        self.stop_analysis()
        self.generate_balls()
        self.draw_balls()
        self.canvas.delete("overlay")
        self.canvas.delete("preview")
        
        messagebox.showinfo(
            "New Game", 
            f"New {self.game_type.get()} game started!\n\n"
            "Click balls to move them around.\n"
            "Start Analysis to get AI recommendations."
        )

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = BilliardsAI(root)
    root.mainloop()

if __name__ == "__main__":
    main()