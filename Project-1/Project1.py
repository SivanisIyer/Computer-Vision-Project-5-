# Project1.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import sys

def download_sample(save_path="test1.jpg"):
    """Download a sample image if no local image is found."""
    url = "https://upload.wikimedia.org/wikipedia/commons/7/7d/Dog_face.png"
    try:
        print(f"Downloading sample image from {url} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved sample image as: {save_path}")
        return save_path
    except Exception as e:
        print("Failed to download sample image:", e)
        return None

def find_image(possible_names=None):
    """Return first existing image path or None."""
    if possible_names is None:
        possible_names = ["test1.jpg", "images/test1.jpg"]
    for p in possible_names:
        if os.path.exists(p):
            return p
    return None

def compute_canny_edges(gray_img):
    """Apply Gaussian blur + automatic median-based thresholds for Canny."""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.4)
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)
    return edges, lower, upper

def main():
    # 1) try to locate image in common places
    img_path = find_image(["test1.jpg", "test1.JPG", "images/test1.jpg", "images/test1.JPG"])
    if img_path is None:
        # 2) ask user (best for interactive), but we'll fallback to download sample automatically
        print("No local file 'test1.jpg' found in current folder or images/ subfolder.")
        print("The script will download a sample image automatically so it can run.")
        img_path = download_sample("test1.jpg")
        if img_path is None:
            print("Could not obtain any image. Exiting.")
            sys.exit(1)

    # 3) read the image in grayscale
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"OpenCV could not read the image at: {img_path}")
        sys.exit(1)

    # 4) compute edges
    edges, lower, upper = compute_canny_edges(img_gray)
    print(f"Canny thresholds used -> lower: {lower}, upper: {upper}")

    # 5) save edge-only image
    edge_save_name = f"edges_{os.path.basename(img_path)}"
    cv2.imwrite(edge_save_name, edges)
    print(f"Saved edge image as: {edge_save_name}")

    # 6) create a combined figure and save it
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original (grayscale)")
    plt.axis('off')
    plt.imshow(img_gray, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Canny Edges")
    plt.axis('off')
    plt.imshow(edges, cmap='gray')

    plt.tight_layout()
    result_save = f"result_{os.path.splitext(os.path.basename(img_path))[0]}.png"
    plt.savefig(result_save, bbox_inches='tight', dpi=150)
    print(f"Saved combined result as: {result_save}")

    # Show the figure (this opens a window; press close to continue)
    plt.show()

if __name__ == "__main__":
    main()
