# Project4.py
# Eye Detection using Haar Cascade (uses test4.jpg in same folder)
import os
import cv2
import matplotlib.pyplot as plt

SAMPLE_URL = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=800&q=80"  # fallback sample

def download_sample(save_path="test4.jpg"):
    import urllib.request
    try:
        print("No local test4.jpg found — downloading a sample image...")
        urllib.request.urlretrieve(SAMPLE_URL, save_path)
        print(f"Saved sample as: {save_path}")
        return save_path
    except Exception as e:
        print("Failed to download sample image:", e)
        return None

def detect_eyes(image_path="test4.jpg", save_output="test4_eyes.jpg", save_crops=True):
    if not os.path.exists(image_path):
        image_path = download_sample(image_path)
        if image_path is None:
            print("No image available. Exiting.")
            return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: could not read image file:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    print("Faces detected:", len(faces))

    eye_crop_count = 0
    for fi, (x,y,w,h) in enumerate(faces, start=1):
        # draw face rectangle (green)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

        # restrict ROI for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # detect eyes inside face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(15,15))
        print(f"  Face {fi}: eyes detected = {len(eyes)}")
        for ei, (ex,ey,ew,eh) in enumerate(eyes, start=1):
            # draw eye rectangle (blue) on the main image (offset by face x,y)
            cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255,0,0), 2)
            if save_crops:
                eye_crop = roi_color[ey:ey+eh, ex:ex+ew]
                if eye_crop.size != 0:
                    eye_crop_count += 1
                    crop_name = f"eye_crop_{fi}_{ei}.jpg"
                    cv2.imwrite(crop_name, eye_crop)

    # save result and show
    cv2.imwrite(save_output, img)
    print(f"Saved output image: {save_output}")
    if save_crops:
        print(f"Saved {eye_crop_count} eye crop(s) as eye_crop_*.jpg")

    # display with matplotlib (works in VS Code)
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Faces: {len(faces)}  (see terminal for eyes per face)")
    plt.show()

if __name__ == "__main__":
    detect_eyes(image_path="test4.jpg", save_output="test4_eyes.jpg", save_crops=True)
