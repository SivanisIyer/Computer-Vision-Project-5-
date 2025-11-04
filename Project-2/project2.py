# Project2.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # your image file
    image_path = "test2.jpg"

    # check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found in the folder.")
        return

    # read color and grayscale
    img_color = cv2.imread(image_path)
    if img_color is None:
        print("Error: Cannot read image. Check file format or path.")
        return
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # convert to binary (Otsu threshold)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert if background is white
    if np.mean(bw) > 127:
        bw = 255 - bw

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_clean, connectivity=8)

    # filter small components
    valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 100]

    print(f"Total objects found (including small): {num_labels - 1}")
    print(f"Objects larger than 100 pixels: {len(valid)}")

    # color the labeled regions
    colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for i in valid:
        mask = labels == i
        color = tuple(rng.integers(50, 255, size=3).tolist())
        colored[mask] = color

    # save output images
    cv2.imwrite("binary_test2.png", bw_clean)
    cv2.imwrite("labeled_test2.png", cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    print("Saved: binary_test2.png, labeled_test2.png")

    # display results
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1); plt.title("Original"); plt.axis('off'); plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.subplot(1,3,2); plt.title("Binary"); plt.axis('off'); plt.imshow(bw_clean, cmap='gray')
    plt.subplot(1,3,3); plt.title("Labeled Objects"); plt.axis('off'); plt.imshow(colored)
    plt.tight_layout()
    plt.savefig("result_test2.png", dpi=150, bbox_inches='tight')
    print("Saved: result_test2.png")
    plt.show()

if __name__ == "__main__":
    main()
