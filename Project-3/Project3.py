# Project3.py - improved face detection with upscaling, equalization and adjustable params
import cv2
import os
import matplotlib.pyplot as plt

def detect_and_save(image_path="test3.jpg",
                    upscale=1.5,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(30,30),
                    save_crops=True):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Unable to read image.")
        return

    print("Original image shape:", img.shape)  # (h, w, channels)

    # Optionally upscale (useful when faces are small)
    if upscale and upscale != 1.0:
        new_w = int(img.shape[1] * upscale)
        new_h = int(img.shape[0] * upscale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print("Upscaled image shape:", img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast for better detection
    gray = cv2.equalizeHist(gray)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print(f"Faces detected: {len(faces)}")

    out = img.copy()
    for i, (x,y,w,h) in enumerate(faces, start=1):
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        if save_crops:
            crop = img[y:y+h, x:x+w]
            cv2.imwrite(f"face_crop_{i}.jpg", crop)

    # Save and show
    out_name = "test3_faces_improved.jpg"
    cv2.imwrite(out_name, out)
    print("Saved:", out_name)
    if faces.any() if isinstance(faces, (list,tuple)) else len(faces)>0:
        print(f"Saved {len(faces)} cropped faces as face_crop_*.jpg")

    # display using matplotlib
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Faces detected: {len(faces)}")
    plt.show()


if __name__ == "__main__":
    # try with different settings if zero faces detected
    detect_and_save(image_path="test3.jpg",
                    upscale=1.5,        # try 1.0, 1.5, or 2.0
                    scaleFactor=1.05,   # try 1.05 or 1.1
                    minNeighbors=3,     # try 3..6
                    minSize=(24,24),
                    save_crops=True)
