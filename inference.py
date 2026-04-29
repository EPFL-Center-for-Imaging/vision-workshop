"""
Inference script for digit recognition on a webcam or USB camera.
"""

import sys
from pickle import load

import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from skimage.morphology import label
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.segmentation import clear_border

from preprocessing import keep_n_biggest_objects


# Reload the sklearn `pipeline`
with open("pipeline.pkl", "rb") as f:
    pipe = load(f)


def detect_digit_roi(image: np.ndarray, max_objects=12, min_area_px=2000) -> pd.DataFrame:
    # RGB => Grayscale
    gray = rgb2gray(image)
    
    # Otsu threshold
    binary = gray < threshold_otsu(gray)
    
    # Labelling + keep biggest object
    labelled = label(binary)
    
    # Remove object touching the image border (incl. the white background, probably)
    labelled = clear_border(labelled)
    
    # Keep the biggest `max_objects` objects only
    digit_crops = keep_n_biggest_objects(labelled, n=max_objects)

    # Return a dataframe with intensity image and centroids
    df = pd.DataFrame(
        regionprops_table(
            digit_crops,
            intensity_image=gray,
            properties=["area", "bbox"])
    )
    
    # Size filter (ignore small objects)
    df = df[df["area"] > min_area_px]
    
    digit_crops = []
    for _, row in df.iterrows():
        # Crop around the digit
        digit_crop = gray[int(row["bbox-0"]) : int(row["bbox-2"]), int(row["bbox-1"]) : int(row["bbox-3"])]
        
        # Resize to a fixed size
        digit_crop = resize(digit_crop, output_shape=(50, 50))
        
        # Convert to 8-bit
        digit_crop = img_as_ubyte(digit_crop)
        digit_crops.append(digit_crop)
    
    df["roi"] = digit_crops
    
    return df


def classify_digit_roi(digit: np.ndarray) -> str:
    # Normalize intensity values
    digit_normed = rescale_intensity(digit, out_range=(0, 1))
    
    # Flatten pixels
    pixel_features = np.reshape(digit_normed[None], (1, -1))
    
    # Predict
    pred_label = pipe.predict(pixel_features)[0]
    
    return pred_label


def draw_boxes(img, df: pd.DataFrame, font_size=1.5, thickness=2):
    boxes = df[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values
    labels = df["pred"].values
    
    output = img.copy()
    for i, box in enumerate(boxes):
        y1, x1, y2, x2 = map(int, box)

        cv2.rectangle(output, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=thickness)

        text = str(labels[i])
        
        (text_w, text_h), baseline = cv2.getTextSize(
            text=text, 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=font_size, 
            thickness=thickness,
        )

        text_y = y1 + text_h + 14

        cv2.rectangle(
            output,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline),
            color=(0, 255, 0),
            thickness=-1
        )

        cv2.putText(
            img=output,
            text=text,
            org=(x1, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_size,
            color=(0, 0, 0),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return output


def main(camera_idx: int):
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        df = detect_digit_roi(frame)
        
        df["pred"] = df["roi"].apply(classify_digit_roi)
        
        vis_frame = draw_boxes(frame, df)

        cv2.imshow("Camera", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run with the correct camera index, for example: `python inference.py 4`
    _, camera_idx = sys.argv
    
    camera_idx = int(camera_idx)
    
    main(camera_idx)