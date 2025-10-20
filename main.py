import mediapipe as mp
import time
import numpy as np
import cv2
import os
import sys

# Variable to track hover start time and active button
hover_start = None
active_button = None

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, 'assets')

# Check if assets directory exists
if not os.path.exists(assets_dir):
    print(f"Error: Assets directory not found at {assets_dir}")
    print("Please ensure the 'assets' folder exists with the clothing images.")
    print("\nRequired images:")
    print("- vest1.png, vest2.png, vest3.png, vest4.png")
    print("- bra1.png, bra2.png")
    sys.exit(1)

# Load the 2D clothing images with transparency
try:
    first_img = cv2.imread(os.path.join(assets_dir, 'vest1.png'), cv2.IMREAD_UNCHANGED)
    second_img = cv2.imread(os.path.join(assets_dir, 'vest2.png'), cv2.IMREAD_UNCHANGED)
    third_img = cv2.imread(os.path.join(assets_dir, 'vest3.png'), cv2.IMREAD_UNCHANGED)
    fourth_img = cv2.imread(os.path.join(assets_dir, 'vest4.png'), cv2.IMREAD_UNCHANGED)
    fifth_img = cv2.imread(os.path.join(assets_dir, 'bra1.png'), cv2.IMREAD_UNCHANGED)
    sixth_img = cv2.imread(os.path.join(assets_dir, 'bra2.png'), cv2.IMREAD_UNCHANGED)
    
    # Check if images are loaded properly
    images = [first_img, second_img, third_img, fourth_img, fifth_img, sixth_img]
    image_names = ['vest1.png', 'vest2.png', 'vest3.png', 'vest4.png', 'bra1.png', 'bra2.png']
    
    for i, img in enumerate(images):
        if img is None:
            print(f"Error: Could not load {image_names[i]} from assets folder")
            print("Please ensure all required images are in the assets folder.")
            sys.exit(1)
    
    # Create icon versions
    icon1_img = cv2.resize(first_img, (40, 40))
    icon2_img = cv2.resize(second_img, (40, 40))
    icon3_img = cv2.resize(third_img, (40, 40))
    icon4_img = cv2.resize(fourth_img, (40, 40))
    icon5_img = cv2.resize(fifth_img, (40, 40))
    icon6_img = cv2.resize(sixth_img, (40, 40))
    
except Exception as e:
    print(f"Error loading images: {e}")
    print("\nPlease ensure the following files exist in the 'assets' folder:")
    print("- vest1.png")
    print("- vest2.png")
    print("- vest3.png")
    print("- vest4.png")
    print("- bra1.png")
    print("- bra2.png")
    sys.exit(1)

# Define the button area (x, y, width, height)
button1_x, button1_y, button_width, button_height = 20, 10, 50, 50
button2_x, button2_y = 20, 80
button3_x, button3_y = 20, 160
button4_x, button4_y = 20, 240
button5_x, button5_y = 20, 320
button6_x, button6_y = 20, 400

# Set the default image to be displayed
current_active_image = first_img

# Clothing item names and prices
clothing_info = [
    ("ENITRE STUDIOS", "$200"),
    ("ISA BOULDER", "$390"),
    ("RICK OWENS", "$469"),
    ("BORIS BIDJAN", "$957"),
    ("HUNZA G", "$245"),
    ("VERSACE", "$899")
]


def draw_button(image, x, y, width, height, icon, label="", label2="", hover=False):
    """Draw a styled button with an icon"""
    # Scale up if hovered and change colors
    if hover:
        x, y = x, y - 10
        width, height = width, height + 10
        color = (0, 128, 0)  # Color for hover state
        shadow_color = (0, 200, 0, 127)  # Slightly darker for the shadow
    else:
        color = (128, 0, 0)  # Default color
        shadow_color = (200, 0, 0, 127) # Default shadow color

    radius = 10
    shadow_offset = 5
    draw_transparent_rectangle(image, (x + radius + shadow_offset, y + shadow_offset),
                               (x + width - radius + shadow_offset, y + height + shadow_offset), shadow_color)
    draw_transparent_rectangle(image, (x + shadow_offset, y + radius + shadow_offset),
                               (x + width + shadow_offset, y + height - radius + shadow_offset), shadow_color)

    # Draw the button (rectangle with rounded corners)
    draw_transparent_rectangle(image, (x + radius, y), (x + width - radius, y + height), color)
    draw_transparent_rectangle(image, (x, y + radius), (x + width, y + height - radius), color)
    draw_transparent_ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color)
    draw_transparent_ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color)
    draw_transparent_ellipse(image, (x + radius, y + height - radius), (radius, radius), 90, 0, 90, color)
    draw_transparent_ellipse(image, (x + width - radius, y + height - radius), (radius, radius), 0, 0, 90, color)

    # Overlay the icon image
    overlay_transparent(image, icon, x + (width - icon.shape[1]) // 2, y + (height - icon.shape[0]) // 2)

    # Add label to the button (optional)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.25, 1)[0]
        text_x = x + 10 + icon.shape[1]
        text_y = y + (height + text_size[1]) // 2 - 5
        cv2.putText(image, label, (text_x, text_y), font, 0.4, (255, 255, 255), 1)

    # Add second label to the button (optional)
    if label2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.25, 1)[0]
        text_x = x + 10 + icon.shape[1]
        text_y = y + (height + text_size[1]) // 2 + 10
        cv2.putText(image, label2, (text_x, text_y), font, 0.4, (255, 255, 255), 1)


def draw_transparent_rectangle(image, start_point, end_point, color, thickness=-1):
    """Draw a transparent rectangle"""
    overlay = image.copy()
    cv2.rectangle(overlay, start_point, end_point, color, thickness)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)  # 30% transparency


def draw_transparent_ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness=-1):
    """Draw a transparent ellipse"""
    overlay = image.copy()
    cv2.ellipse(overlay, center, axes, angle, startAngle, endAngle, color, thickness)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)  # 30% transparency


def overlay_transparent(background, overlay, x, y):
    """Overlay an image onto another image with transparency"""
    bg_height, bg_width = background.shape[:2]
    h, w = overlay.shape[:2]

    # Calculate the overlay region
    x1, x2 = max(x, 0), min(x + w, bg_width)
    y1, y2 = max(y, 0), min(y + h, bg_height)

    overlay_img = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    mask = overlay_img[:, :, 3:] / 255.0
    background[y1:y2, x1:x2] = (1.0 - mask) * background[y1:y2, x1:x2] + mask * overlay_img[:, :, :3]

    return background


def is_body_orientation_acceptable(landmarks):
    """Check if the body is properly oriented for clothing overlay"""
    left_shoulder = landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the angle or difference - this is a simple example, adjust as needed
    shoulder_difference = abs(left_shoulder.y - right_shoulder.y)

    # Define a threshold for the difference
    threshold = 0.1  # adjust this value based on experimentation

    return shoulder_difference < threshold


def draw_rasengan(image, center, radius):
    """Draw a visual effect on the fingertip"""
    # Draw the bright core
    core_radius = radius // 4
    cv2.circle(image, center, core_radius, (255, 255, 255), -1)

    # Draw the swirling pattern
    for i in range(radius, core_radius, -1):
        alpha = (radius - i) / radius
        color = (255 - int(255 * alpha), 255, 255 - int(255 * alpha))
        noise = np.random.randint(-1, 2)  # Add some noise for the swirl effect
        cv2.circle(image, (center[0] + noise, center[1] + noise), i, color, 2)

    # Add an outer glow
    glow_radius = radius + 10
    for i in range(1, glow_radius - radius):
        alpha = (glow_radius - radius - i) / (glow_radius - radius)
        overlay_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
        cv2.circle(image, center, radius + i, overlay_color, 1)


def find_camera():
    """Find an available camera index"""
    for i in range(5):  # Try first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera found at index {i}")
                return cap
            cap.release()
    return None


# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Try to find an available camera
print("Looking for camera...")
cap = find_camera()

if cap is None:
    print("Error: No camera found. Please ensure your webcam is connected.")
    sys.exit(1)

print("\nVirtual Try-On Application Started!")
print("Instructions:")
print("- Stand in front of the camera with your upper body visible")
print("- Use your right index finger to hover over clothing buttons")
print("- Hold your finger on a button for 1 second to select")
print("- Press 'q' to quit\n")

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        # Initialize default values for shoulder_width and shoulder_to_hip_height
        shoulder_width = 200  # default value, adjust as needed
        shoulder_to_hip_height = 400  # default value, adjust as needed
        center_x = 0  # default value
        center_y = 0  # default value

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw buttons (color based on active state)
        is_button1_active = (current_active_image is first_img)
        is_button2_active = (current_active_image is second_img)
        is_button3_active = (current_active_image is third_img)
        is_button4_active = (current_active_image is fourth_img)
        is_button5_active = (current_active_image is fifth_img)
        is_button6_active = (current_active_image is sixth_img)
        
        # Draw all buttons
        buttons = [
            (button1_x, button1_y, icon1_img, is_button1_active),
            (button2_x, button2_y, icon2_img, is_button2_active),
            (button3_x, button3_y, icon3_img, is_button3_active),
            (button4_x, button4_y, icon4_img, is_button4_active),
            (button5_x, button5_y, icon5_img, is_button5_active),
            (button6_x, button6_y, icon6_img, is_button6_active)
        ]
        
        for i, (x, y, icon, is_active) in enumerate(buttons):
            label, price = clothing_info[i]
            draw_button(image, x, y, button_width, button_height, 
                       icon=icon, label=label, label2=price, hover=is_active)

        # Check if pose landmarks are available
        if results.pose_landmarks and is_body_orientation_acceptable(results.pose_landmarks):
            index_finger_tip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX.value]
            index_x, index_y = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])
            draw_rasengan(image, (index_x, index_y), 10)  # Adjust the radius as needed

            index_finger = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX.value]
            finger_x, finger_y = int(index_finger.x * image.shape[1]), int(index_finger.y * image.shape[0])

            # Check which button is being hovered
            button_positions = [
                (button1_x, button1_y, 1),
                (button2_x, button2_y, 2),
                (button3_x, button3_y, 3),
                (button4_x, button4_y, 4),
                (button5_x, button5_y, 5),
                (button6_x, button6_y, 6)
            ]
            
            hovering = False
            for bx, by, button_num in button_positions:
                if bx <= finger_x <= bx + button_width and by <= finger_y <= by + button_height:
                    hovering = True
                    if hover_start is None:
                        hover_start = time.time()
                        active_button = button_num
                    break
            
            if not hovering:
                hover_start = None  # Reset timer if hand moves away

            # Calculate clothing overlay position
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP.value]
            right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP.value]

            # Calculate width (shoulder to shoulder) and height (shoulder to hip)
            shoulder_width = abs(right_shoulder.x - left_shoulder.x) * image.shape[1] * 1.25
            shoulder_to_hip_height = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2) * image.shape[0] * 1.25

            # Calculate center position for the clothing image
            center_x = int(((left_shoulder.x + right_shoulder.x) / 2) * image.shape[1]) - (int(shoulder_width) // 2)
            center_y = int(((left_hip.y - left_shoulder.y) / 2 + left_shoulder.y) * image.shape[0]) - (
                        int(shoulder_to_hip_height) // 2) - 10

            if current_active_image is not None and shoulder_width > 0 and shoulder_to_hip_height > 0:
                shoulder_width = abs(shoulder_width)
                shoulder_to_hip_height = abs(shoulder_to_hip_height)
                resized_clothing_img = cv2.resize(current_active_image,
                                                  (int(shoulder_width), int(shoulder_to_hip_height)))
                image = overlay_transparent(image, resized_clothing_img, center_x, center_y)

        # Check if the hand has hovered for 1 second over any button
        if hover_start and (time.time() - hover_start) >= 1:
            clothing_images = [first_img, second_img, third_img, fourth_img, fifth_img, sixth_img]
            if 1 <= active_button <= 6:
                current_active_image = clothing_images[active_button - 1]
            hover_start = None  # Reset hover start time

        # Add instructions on screen
        cv2.putText(image, "Hover finger over button to select", (10, image.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize the image for a larger display
        display_image = cv2.resize(image, (0, 0), fx=2, fy=2)
        cv2.imshow('Virtual Try-On - Hand Gesture Control', display_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

print("\nThank you for using Virtual Try-On!")
cap.release()
cv2.destroyAllWindows()