import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
import threading
import queue

# Try to import OpenCV, fall back to PIL if not available
try:
    import cv2
    CV2_AVAILABLE = True
    from billiards_detector import BilliardsDetector
except ImportError:
    CV2_AVAILABLE = False
    from simple_detector import SimpleBilliardsDetector as BilliardsDetector

from overlay_renderer import OverlayRenderer
from trajectory_calculator import TrajectoryCalculator

# Page configuration
st.set_page_config(
    page_title="AI Billiards Assistant",
    page_icon="🎱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = BilliardsDetector()
if 'overlay_renderer' not in st.session_state:
    st.session_state.overlay_renderer = OverlayRenderer()
if 'trajectory_calc' not in st.session_state:
    st.session_state.trajectory_calc = TrajectoryCalculator()
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'overlay_enabled' not in st.session_state:
    st.session_state.overlay_enabled = True
if 'game_type' not in st.session_state:
    st.session_state.game_type = "8-ball"

def main():
    st.title("🎱 AI Billiards Aiming Assistant")
    st.markdown("Real-time computer vision powered billiards shot analysis and guidance")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Camera controls
        st.subheader("Camera")
        camera_col1, camera_col2 = st.columns(2)
        
        with camera_col1:
            if st.button("📷 Start Camera", disabled=st.session_state.camera_active):
                st.session_state.camera_active = True
                st.rerun()
        
        with camera_col2:
            if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_active):
                st.session_state.camera_active = False
                st.rerun()
        
        # Game settings
        st.subheader("Game Settings")
        game_type = st.selectbox(
            "Game Type",
            ["8-ball", "9-ball", "Straight Pool", "Snooker"],
            index=0
        )
        st.session_state.game_type = game_type
        
        # Overlay settings
        st.subheader("Overlay Settings")
        overlay_enabled = st.checkbox("Enable AI Guidance", value=st.session_state.overlay_enabled)
        st.session_state.overlay_enabled = overlay_enabled
        
        overlay_opacity = st.slider("Overlay Opacity", 0.1, 1.0, 0.7, 0.1)
        st.session_state.overlay_renderer.set_opacity(overlay_opacity)
        
        aim_line_color = st.color_picker("Aim Line Color", "#00FF00")
        trajectory_color = st.color_picker("Trajectory Color", "#FF0000")
        
        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
        st.session_state.detector.set_confidence(confidence_threshold)
        
        # Performance info
        st.subheader("Performance")
        if 'fps' in st.session_state:
            st.metric("FPS", f"{st.session_state.fps:.1f}")
        if 'detection_time' in st.session_state:
            st.metric("Detection Time", f"{st.session_state.detection_time:.3f}s")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Video Feed")
        video_placeholder = st.empty()
        
        if st.session_state.camera_active:
            run_camera_feed(video_placeholder)
        else:
            video_placeholder.image(
                create_placeholder_image(),
                caption="Camera feed will appear here when started",
                use_column_width=True
            )
    
    with col2:
        st.subheader("Analysis")
        
        # Shot analysis display
        analysis_placeholder = st.empty()
        
        if 'current_analysis' in st.session_state:
            display_shot_analysis(analysis_placeholder)
        else:
            analysis_placeholder.info("Start camera to see shot analysis")
        
        # Ball detection info
        st.subheader("Detected Objects")
        detection_placeholder = st.empty()
        
        if 'detections' in st.session_state:
            display_detections(detection_placeholder)
        else:
            detection_placeholder.info("No detections available")

def run_camera_feed(placeholder):
    """Run the camera feed with real-time processing"""
    if not CV2_AVAILABLE:
        # Demo mode with simulated billiards table
        run_demo_mode(placeholder)
        return
        
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            placeholder.error("Could not access camera. Please check camera permissions.")
            st.session_state.camera_active = False
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                placeholder.error("Failed to read from camera")
                break
            
            # Process frame
            start_time = time.time()
            processed_frame = process_frame(frame)
            processing_time = time.time() - start_time
            
            # Update performance metrics
            fps_counter += 1
            if fps_counter % 10 == 0:  # Update FPS every 10 frames
                fps_end_time = time.time()
                st.session_state.fps = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            st.session_state.detection_time = processing_time
            
            # Convert frame for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            # Small delay to prevent overwhelming the display
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        
    except Exception as e:
        placeholder.error(f"Camera error: {str(e)}")
        st.session_state.camera_active = False

def run_demo_mode(placeholder):
    """Run demo mode with simulated billiards table"""
    fps_counter = 0
    fps_start_time = time.time()
    
    while st.session_state.camera_active:
        # Create demo billiards table image
        start_time = time.time()
        demo_frame = create_demo_billiards_image()
        processed_frame = process_frame(demo_frame)
        processing_time = time.time() - start_time
        
        # Update performance metrics
        fps_counter += 1
        if fps_counter % 10 == 0:
            fps_end_time = time.time()
            st.session_state.fps = 10 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
        
        st.session_state.detection_time = processing_time
        
        # Display processed frame
        placeholder.image(processed_frame, channels="RGB", use_column_width=True)
        
        # Small delay
        time.sleep(0.1)  # 10 FPS for demo

def process_frame(frame):
    """Process a single frame with detection and overlay"""
    try:
        # Detect billiards objects
        detections = st.session_state.detector.detect(frame)
        st.session_state.detections = detections
        
        # Calculate trajectories and analysis
        if detections['balls'] and st.session_state.overlay_enabled:
            analysis = st.session_state.trajectory_calc.analyze_shot(
                detections, st.session_state.game_type
            )
            st.session_state.current_analysis = analysis
            
            # Render overlays
            if CV2_AVAILABLE:
                frame = st.session_state.overlay_renderer.render_overlays(
                    frame, detections, analysis
                )
            else:
                # Use PIL-based overlay rendering
                frame = render_overlays_pil(frame, detections, analysis)
        
        return frame
        
    except Exception as e:
        st.error(f"Frame processing error: {str(e)}")
        return frame

def create_demo_billiards_image():
    """Create a demo billiards table image"""
    # Create a green billiards table
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(34, 139, 34))  # Forest green
    draw = ImageDraw.Draw(img)
    
    # Draw table borders
    border_width = 40
    draw.rectangle([border_width, border_width, width-border_width, height-border_width], 
                   outline=(139, 69, 19), width=8)  # Brown border
    
    # Draw pockets (black circles)
    pocket_radius = 25
    pockets = [
        (border_width + 10, border_width + 10),  # Top-left
        (width - border_width - 10, border_width + 10),  # Top-right
        (border_width + 10, height - border_width - 10),  # Bottom-left
        (width - border_width - 10, height - border_width - 10),  # Bottom-right
        (width // 2, border_width + 5),  # Top-center
        (width // 2, height - border_width - 5),  # Bottom-center
    ]
    
    for pocket in pockets:
        draw.ellipse([pocket[0]-pocket_radius, pocket[1]-pocket_radius, 
                     pocket[0]+pocket_radius, pocket[1]+pocket_radius], 
                    fill=(0, 0, 0))
    
    # Convert PIL image to numpy array
    return np.array(img)

def render_overlays_pil(frame, detections, analysis):
    """Render overlays using PIL when OpenCV is not available"""
    # Convert numpy array to PIL Image
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = frame
    
    draw = ImageDraw.Draw(img)
    
    # Draw detected balls
    if detections.get('balls'):
        for ball in detections['balls']:
            center = ball['center']
            radius = ball.get('radius', 12)
            
            # Draw ball circle
            if ball.get('type') == 'cue':
                color = (255, 255, 255)  # White for cue ball
                width = 3
            else:
                color = (255, 255, 0)  # Yellow for other balls
                width = 2
            
            draw.ellipse([center[0]-radius, center[1]-radius, 
                         center[0]+radius, center[1]+radius], 
                        outline=color, width=width)
            
            # Draw ball label
            if ball.get('type'):
                label = str(ball['type'])
                draw.text((center[0]-5, center[1]-5), label, fill=(255, 255, 255))
    
    # Draw aiming line if analysis available
    if analysis.get('recommended_shot') and detections.get('balls'):
        shot = analysis['recommended_shot']
        cue_ball = None
        target_ball = None
        
        # Find cue ball and target ball
        for ball in detections['balls']:
            if ball.get('type') == 'cue':
                cue_ball = ball
            elif ball.get('type') == shot.get('target_ball'):
                target_ball = ball
        
        if cue_ball and target_ball:
            # Draw aiming line
            draw.line([cue_ball['center'], target_ball['center']], 
                     fill=(0, 255, 0), width=3)
            
            # Draw target indicator
            center = target_ball['center']
            radius = target_ball.get('radius', 12) + 8
            draw.ellipse([center[0]-radius, center[1]-radius, 
                         center[0]+radius, center[1]+radius], 
                        outline=(0, 255, 255), width=2)
    
    return np.array(img)

def display_shot_analysis(placeholder):
    """Display shot analysis information"""
    analysis = st.session_state.current_analysis
    
    with placeholder.container():
        if analysis.get('recommended_shot'):
            shot = analysis['recommended_shot']
            st.success("🎯 Recommended Shot")
            st.write(f"**Target:** {shot.get('target_ball', 'Unknown')}")
            st.write(f"**Difficulty:** {shot.get('difficulty', 'Unknown')}")
            st.write(f"**Success Rate:** {shot.get('success_probability', 0):.1%}")
            
            if shot.get('angle'):
                st.write(f"**Aim Angle:** {shot['angle']:.1f}°")
            if shot.get('power'):
                st.write(f"**Recommended Power:** {shot['power']:.1%}")
        else:
            st.info("No clear shot recommendation available")
        
        # Additional analysis
        if analysis.get('cue_ball_position'):
            st.write("**Cue Ball Position:** Detected")
        
        if analysis.get('warnings'):
            for warning in analysis['warnings']:
                st.warning(f"⚠️ {warning}")

def display_detections(placeholder):
    """Display detection information"""
    detections = st.session_state.detections
    
    with placeholder.container():
        if detections.get('table'):
            st.success("✅ Table detected")
        else:
            st.error("❌ No table detected")
        
        balls_count = len(detections.get('balls', []))
        st.metric("Balls Detected", balls_count)
        
        if detections.get('cue_stick'):
            st.success("✅ Cue stick detected")
        
        # List detected balls
        if detections.get('balls'):
            st.write("**Detected Balls:**")
            for i, ball in enumerate(detections['balls'][:5]):  # Show first 5
                ball_type = ball.get('type', f'Ball {i+1}')
                confidence = ball.get('confidence', 0)
                st.write(f"• {ball_type} ({confidence:.2f})")

def create_placeholder_image():
    """Create a placeholder image when camera is not active"""
    if CV2_AVAILABLE:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img.fill(50)  # Dark gray background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Camera Not Active"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        
        return img
    else:
        # Use PIL for placeholder image
        img = Image.new('RGB', (640, 480), color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        
        text = "Camera Not Active - Demo Mode"
        # Get text size (approximate)
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (640 - text_width) // 2
        y = (480 - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255))
        
        return np.array(img)

if __name__ == "__main__":
    main()
