# Project5.py
"""
Color-Based Object Tracking (image or webcam)
- Default uses "test5.jpg" (in same folder). If not found, can use webcam (--webcam).
- Choose color via --color (red, green, blue) or pass custom HSV ranges.
- Saves result image "result_test5.png" (for image mode) or shows live webcam.
"""

import cv2
import numpy as np
import os
import argparse

# Predefined HSV ranges for common colors (H:0-179, S:0-255, V:0-255)
COLOR_RANGES = {
    "red": [([0, 100, 70], [10, 255, 255]), ([170,100,70], [180,255,255])],  # red needs two ranges
    "green": [([35, 60, 40], [85, 255, 255])],
    "blue": [([95, 80, 40], [135, 255, 255])]
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", "-i", default="test5.jpg", help="path to input image (default test5.jpg)")
    p.add_argument("--webcam", "-w", action="store_true", help="use webcam live tracking")
    p.add_argument("--color", "-c", default="red", choices=["red","green","blue","custom"], help="color to track")
    p.add_argument("--min_area", type=int, default=500, help="minimum contour area to consider (pixels)")
    p.add_argument("--show_mask", action="store_true", help="show threshold mask window (helpful for tuning)")
    # optional custom HSV: pass as comma-separated: H1,S1,V1,H2,S2,V2
    p.add_argument("--custom_hsv", type=str, default=None, help="custom HSV lower/upper as H1,S1,V1,H2,S2,V2")
    return p.parse_args()

def build_mask_hsv(hsv, color, custom=None):
    masks = []
    if color == "custom" and custom is not None:
        try:
            vals = [int(x) for x in custom.split(",")]
            if len(vals) != 6:
                raise ValueError
            lower = np.array(vals[:3], dtype=np.uint8)
            upper = np.array(vals[3:], dtype=np.uint8)
            masks.append(cv2.inRange(hsv, lower, upper))
        except Exception:
            raise ValueError("custom_hsv must be H1,S1,V1,H2,S2,V2 (integers)")
    else:
        ranges = COLOR_RANGES[color]
        for (lo, hi) in ranges:
            lower = np.array(lo, dtype=np.uint8)
            upper = np.array(hi, dtype=np.uint8)
            masks.append(cv2.inRange(hsv, lower, upper))
    if len(masks) == 1:
        return masks[0]
    else:
        return cv2.bitwise_or(masks[0], masks[1])

def process_frame(frame, color, min_area=500, show_mask=False, custom_hsv=None):
    blurred = cv2.GaussianBlur(frame, (7,7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = build_mask_hsv(hsv, color, custom=custom_hsv)

    # morphological ops to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        cx = x + w//2
        cy = y + h//2
        detections.append((x,y,w,h,area,(cx,cy)))

    # sort by area desc
    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    # draw results
    out = frame.copy()
    for i,(x,y,w,h,area,centroid) in enumerate(detections, start=1):
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(out, f"#{i} {area:.0f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.circle(out, centroid, 4, (0,0,255), -1)

    if show_mask:
        cv2.imshow("mask", mask)
    return out, mask, detections

def run_image_mode(image_path, color, min_area, show_mask, custom_hsv):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    frame = cv2.imread(image_path)
    out, mask, detections = process_frame(frame, color, min_area=min_area, show_mask=show_mask, custom_hsv=custom_hsv)
    res_name = f"result_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    cv2.imwrite(res_name, out)
    print(f"Saved result image: {res_name}")
    # show final image with matplotlib-like window
    cv2.imshow("Tracking result", out)
    if show_mask:
        cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_webcam_mode(color, min_area, show_mask, custom_hsv):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out, mask, detections = process_frame(frame, color, min_area=min_area, show_mask=show_mask, custom_hsv=custom_hsv)
        cv2.imshow("Webcam Tracking (q to quit)", out)
        if show_mask:
            cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    custom = args.custom_hsv
    if args.webcam:
        print(f"Starting webcam. Tracking color: {args.color}")
        run_webcam_mode(args.color, args.min_area, args.show_mask, custom)
    else:
        print(f"Processing image: {args.image}  (tracking color: {args.color})")
        run_image_mode(args.image, args.color, args.min_area, args.show_mask, custom)

if __name__ == "__main__":
    main()
