import cv2
import numpy as np
import os
import csv
from make_preds import preds

def nothing(x):
    pass

# Create a window for sliders and camera feed
cv2.namedWindow("Color Adjustments and Feed", cv2.WINDOW_NORMAL)

# Create trackbars for HSV thresholds
cv2.createTrackbar("H Low", "Color Adjustments and Feed", 65, 179, nothing)
cv2.createTrackbar("H High", "Color Adjustments and Feed", 85, 179, nothing)
cv2.createTrackbar("S Low", "Color Adjustments and Feed", 99, 255, nothing)
cv2.createTrackbar("S High", "Color Adjustments and Feed", 255, 255, nothing)
cv2.createTrackbar("V Low", "Color Adjustments and Feed", 0, 255, nothing)
cv2.createTrackbar("V High", "Color Adjustments and Feed", 255, 255, nothing)

# Constants
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
SMOOTHING_WINDOW = 5
MAX_FRAMES = 100
kernel = np.ones((3, 3), np.uint8)  # Kernel for denoising


def denoise_mask(mask):
    """Denoise the mask using morphological operations."""
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    return dilated_mask


def moving_average(points, window_size):
    """Applies moving average smoothing to a list of points."""
    if len(points) < window_size:
        return points
    smoothed_points = []
    for i in range(len(points)):
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        window = points[window_start:window_end]
        avg_x = int(sum(p[0] for p in window) / len(window))
        avg_y = int(sum(p[1] for p in window) / len(window))
        smoothed_points.append((avg_x, avg_y))
    return smoothed_points


def process_frame(frame, points_list, start_capture, frame_count):
    """Processes a single frame, isolates the green tip, and tracks its center."""
    # Get HSV thresholds from sliders
    h_low = cv2.getTrackbarPos("H Low", "Color Adjustments and Feed")
    h_high = cv2.getTrackbarPos("H High", "Color Adjustments and Feed")
    s_low = cv2.getTrackbarPos("S Low", "Color Adjustments and Feed")
    s_high = cv2.getTrackbarPos("S High", "Color Adjustments and Feed")
    v_low = cv2.getTrackbarPos("V Low", "Color Adjustments and Feed")
    v_high = cv2.getTrackbarPos("V High", "Color Adjustments and Feed")

    lower_green = np.array([h_low, s_low, v_low])
    upper_green = np.array([h_high, s_high, v_high])

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Denoise the mask
    denoised_mask = denoise_mask(mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(denoised_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

            # Add the point to the list if capturing
            if start_capture and frame_count < MAX_FRAMES:
                points_list.append([frame_count+1, cx, cy])

            # Draw the center on the frame
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot

    return frame, center


def normalize_points(points_list, video_width, video_height):
    """Normalizes the x and y coordinates to range [0, 1]."""
    if not points_list:
        return []

    # Extract x and y coordinates
    x_coords = [point[1] for point in points_list]
    y_coords = [point[2] for point in points_list]
    
    # Calculate the min/max for scaling
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    if max_x - min_x == 0:
      print("No Change in x coordinates. Cannot normalize")
    if max_y - min_y == 0:
      print("No Change in y coordinates. Cannot normalize")
    
    normalized_points = []
    for frame, x, y in points_list:
      # Normalize x
      if max_x - min_x != 0:
        normalized_x = (x - min_x) / (max_x - min_x)
      else:
        normalized_x = 0
      
      # Normalize y
      if max_y - min_y !=0:
        normalized_y = (y - min_y) / (max_y - min_y)
      else:
        normalized_y = 0

      normalized_points.append([frame, normalized_x, normalized_y])
      
    return normalized_points


def save_drawing_and_csv(output_folder, canvas, points_list, drawing_id, video_width, video_height):
    """Saves the canvas as a PNG and the points list as a CSV."""
    # Save the canvas
    png_path = f"{output_folder}/wand_path_{drawing_id}.png"
    cv2.imwrite(png_path, canvas)
    print(f"Saved drawing to {png_path}")

    # Normalize points
    normalized_points = normalize_points(points_list, video_width, video_height)
    
    # Save the points to a CSV file
    csv_path = f"{output_folder}/wand_path_{drawing_id}.csv"
    
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "x", "y"])  # Write header
        writer.writerows(normalized_points)  # Write points
    # print(f"Saved normalized coordinates to {csv_path}")
    preds("path_matching_model121212.keras",f"./{csv_path}")


def process_camera(output_folder="output"):
    """Captures frames from the camera, tracks the wand, and allows multiple drawings."""
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    points_list = []
    start_capture = False
    canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)  # Blank canvas
    drawing_id = 1  # Counter for saved drawings
    frame_count = 0  # Counter for captured frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame = cv2.flip(frame, 1)

        # Check for user input
        key = cv2.waitKey(1)
        if key == ord(' '):  # Start/stop capture on spacebar press
            start_capture = not start_capture
            if not start_capture:
                # Reset capture state when stopped manually
                frame_count = 0
                points_list.clear()
        elif key == 27:  # Exit on ESC
            break

        # Process the frame
        processed_frame, center = process_frame(frame, points_list, start_capture, frame_count)

        # If capturing, increment frame count and check for auto-save
        if start_capture:
            frame_count += 1
            if frame_count >= MAX_FRAMES:
                save_drawing_and_csv(output_folder, canvas, points_list, drawing_id, VIDEO_WIDTH, VIDEO_HEIGHT)
                drawing_id += 1
                frame_count = 0
                points_list.clear()
                canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
                start_capture = False

        # Draw the smoothed path on the canvas
        smoothed_points = moving_average([(x, y) for _, x, y in points_list], SMOOTHING_WINDOW)
        for i in range(1, len(smoothed_points)):
            if i < 30:
              cv2.line(canvas, smoothed_points[i - 1], smoothed_points[i], (0, 255, 0), 2)  # Green path
            elif i >70:
              cv2.line(canvas, smoothed_points[i - 1], smoothed_points[i], (0, 0, 255), 2)  # Red path
            else:
              cv2.line(canvas, smoothed_points[i - 1], smoothed_points[i], (255, 255, 255), 2)  # Red path
                
        # Overlay the canvas on the frame for real-time feedback
        combined_frame = cv2.addWeighted(processed_frame, 0.7, canvas, 0.3, 0)

        # Display frame number and coordinates
        if center:
            cv2.putText(combined_frame, f"Frame: {frame_count}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined_frame, f"Tip Position: {center}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the combined feed
        cv2.imshow("Color Adjustments and Feed", combined_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting program.")


if __name__ == "__main__":
    process_camera()