import sys
from pickle import load

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from skimage.morphology import label
from skimage.transform import rescale, resize
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.segmentation import clear_border

from preprocessing import keep_n_biggest_objects

# Reload the sklearn `pipeline`
with open("pipeline.pkl", "rb") as f:
    pipe = load(f)


def extract_digit_from_digit_roi(image: np.ndarray) -> np.ndarray:
    """Assuming that a digit is drawn inside the ROI, we extract a crop around it and resize it to (50, 50)."""
    image = img_as_ubyte(image)  # Convert to 8-bit
    
    # Downscale by a factor 2
    resized = rescale(image, scale=0.5)
    
    # Otsu threshold
    try:
        binary = resized > threshold_otsu(resized)
    except ValueError:
        return img_as_ubyte(np.zeros((50, 50)))
    
    # Labelling
    labelled = label(binary)

    # The white background is assumed to be the biggest object
    light_square = keep_n_biggest_objects(labelled, n=1)

    # We assume that the digit is the biggest "hole" in that bright background
    # => If the digit is connected to the border it won't be detected
    filled_light_square = ndi.binary_fill_holes(light_square)
    binary_in_light_square = np.logical_and(filled_light_square, light_square == 0)
    labelled_in_light_square = label(binary_in_light_square)
    
    n_objects = labelled_in_light_square.max()
    if n_objects == 0:
        # Case where the ROI is empty
        digit_crop = resized
    else:
        # Digit = biggest object
        digit = keep_n_biggest_objects(labelled_in_light_square, n=1)
        # Extract bounding box
        df = pd.DataFrame(
            regionprops_table(digit, intensity_image=resized, properties=["bbox"])
        )
        digit_row = df.iloc[0]
        # Crop around the digit
        digit_crop = resized[digit_row["bbox-0"] : digit_row["bbox-2"], digit_row["bbox-1"] : digit_row["bbox-3"]]

    # Resize to a fixed size
    digit_crop = resize(digit_crop, output_shape=(50, 50))
    
    # Convert to 8-bit
    digit_crop = img_as_ubyte(digit_crop)
    
    return digit_crop


def detect_digit_roi(image: np.ndarray, max_objects=12) -> pd.DataFrame:
    # RGB => Grayscale
    gray = rgb2gray(image)
    
    # Otsu threshold
    binary = gray > threshold_otsu(gray)
    
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
            properties=["intensity_image", "bbox"],
        )
    )
    
    return df


def classify_digit_square(digit_square_img: np.ndarray) -> str:
    digit = extract_digit_from_digit_roi(digit_square_img)
    
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
        
        df["pred"] = df["intensity_image"].apply(classify_digit_square)
        
        vis_frame = draw_boxes(frame, df)

        cv2.imshow("Camera", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _, camera_idx = sys.argv
    
    camera_idx = int(camera_idx)
    
    main(camera_idx)