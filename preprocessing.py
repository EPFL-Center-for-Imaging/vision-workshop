"""
Script used to extract digit images and the corresponding ground truth labels from images of training grids captured by a camera device. 

The ground truth labels are inferred from the position of the digits in the grid, which is predetermined.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from skimage.morphology import label
from skimage.transform import rescale, resize
from skimage.util import img_as_ubyte


def keep_n_biggest_objects(labelled: np.ndarray, n=1) -> np.ndarray:
    """Remove all but the N biggest objects in a labelled array."""
    uniques, counts = np.unique(labelled, return_counts=1)

    # Ignore the background if it's there
    if uniques[0] == 0:
        uniques = uniques[1:]
        counts = counts[1:]

    # Sort unique values by counts (descending), then extract the N unique values corresponding to the biggest objects
    biggest_labels = uniques[np.argsort(counts)[::-1][:n]]

    biggest_objects_filt = np.isin(labelled, biggest_labels)

    biggest_objects_mask = labelled.copy()
    biggest_objects_mask[~biggest_objects_filt] = 0

    return biggest_objects_mask


def detect_digit_rois_in_training_grid(image: np.ndarray, grid_shape: tuple) -> pd.DataFrame:
    """Extract digits from the image of a training grid."""
    # Convert the image to gray
    gray = rgb2gray(image)

    # Binarize it with Otsu
    binary = gray < threshold_otsu(gray)

    # Label connected components
    labelled = label(binary)

    # Biggest object should be the grid frame
    grid_frame = keep_n_biggest_objects(labelled, n=1)

    # Fill the grid and use the difference to identify squares
    filled_grid_frame = ndi.binary_fill_holes(grid_frame)
    digit_squares = np.logical_and(filled_grid_frame, grid_frame == 0)

    # Label the squares
    digit_squares_labelled = label(digit_squares)

    # Keep N biggest objects (squares)
    n_squares = grid_shape[0] * grid_shape[1]
    digit_squares_cleaned = keep_n_biggest_objects(digit_squares_labelled, n=n_squares)

    # Return a dataframe with intensity image and centroids
    df = pd.DataFrame(
        regionprops_table(
            digit_squares_cleaned,
            intensity_image=gray,
            properties=["intensity_image", "centroid"],
        )
    )
    
    # Sort values
    n_rows, n_cols = grid_shape
    df = df.sort_values(by="centroid-0").reset_index(drop=True)
    row_dfs = []
    for i in range(n_rows):
        row_df = df.iloc[i * n_cols : (i + 1) * n_cols].copy()
        row_df = row_df.sort_values(by="centroid-1").reset_index(drop=True)
        row_dfs.append(row_df)
    df = pd.concat(row_dfs, ignore_index=True)
    
    return df


def extract_digit_from_digit_roi(image: np.ndarray) -> np.ndarray:
    """Note: almost the same as in the inference script, but we print warnings."""
    image = img_as_ubyte(image)
    resized = rescale(image, scale=0.5)
    binary = resized > threshold_otsu(resized)
    labelled = label(binary)
    light_square = keep_n_biggest_objects(labelled, n=1)
    filled_light_square = ndi.binary_fill_holes(light_square)
    binary_in_light_square = np.logical_and(filled_light_square, light_square == 0)
    labelled_in_light_square = label(binary_in_light_square)
    
    n_objects = labelled_in_light_square.max()
    if n_objects == 0:
        print("⚠️ Could not extract a digit in this image!")
        digit_crop = resized
    else:
        digit = keep_n_biggest_objects(labelled_in_light_square, n=1)
        df = pd.DataFrame(
            regionprops_table(digit, intensity_image=resized, properties=["bbox"])
        )
        digit_row = df.iloc[0]
        digit_crop = resized[digit_row["bbox-0"] : digit_row["bbox-2"], digit_row["bbox-1"] : digit_row["bbox-3"]]

    digit_crop = resize(digit_crop, output_shape=(50, 50))
    digit_crop = img_as_ubyte(digit_crop)
    
    return digit_crop


if __name__ == "__main__":
    # Run with, for example: `python preprocessing.py datasets/`
    _, root_img_folder = sys.argv
    
    # We assume that the images contain a (N, M) grid of digits in a predetermined order:
    grid_shape = (3, 4)
    class_labels = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B"]).reshape(grid_shape)

    root = Path(root_img_folder)
    
    training_images_dir = root / "raw_training_images"
    
    dataset_dir = root / "dataset"
    if not dataset_dir.exists():
        os.mkdir(dataset_dir)

    # Create a subdirectory for each class
    for lab in class_labels.ravel():
        dst_dir = dataset_dir / lab
        if not dst_dir.exists():
            os.mkdir(dst_dir)

    # Iterate over the training images
    for image_file in training_images_dir.glob("*.jpg"):
        print("---")
        print(image_file)
        print("---")

        training_img = skimage.io.imread(image_file)

        # Extract `digit squares` crops, sorted by [centroid-0, centroid-1] so we can assume their class label.
        df = detect_digit_rois_in_training_grid(training_img, grid_shape)
        
        # Add the `class` label
        df["class"] = class_labels.ravel()

        for _, row in df.iterrows():
            digit_roi = row["intensity_image"]
            digit_cls = row["class"]
            
            digit_img = extract_digit_from_digit_roi(digit_roi)
            
            # Save the digit image in the corresponding class subfolder
            dst_file = dataset_dir / digit_cls / f"{image_file.stem}.png"

            skimage.io.imsave(dst_file, digit_img)