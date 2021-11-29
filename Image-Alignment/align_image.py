import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import argparse
import os
from utils.rotate import correct_skew
from angle_calculation import *


def main(folder_path):
    """Fix image alignment along horizontal or vertical axis

    Args:
        folder_path (str): path to folders with misaligned images
    """
    # read img directory
    images = os.listdir(folder_path)
    output_path = "aligned_images"
    os.makedirs("./aligned_images/", exist_ok=True)
    for img_name in images:
        print(f"processing {img_name}")
        if not (
            img_name.endswith(".jpg")
            or img_name.endswith(".png")
            or img_name.endswith(".jpeg")
        ):
            print(img_name)
            continue
        image = cv2.imread(os.path.join(folder_path, img_name))
        # correct image skew using img histogram
        _, image_aligned = correct_skew(image)
        img_name = img_name.split(".")[0] + ".jpg"
        cv2.imwrite(f"{output_path}/aligned_{img_name}", image_aligned)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--image_folder",
        required=True,
        help="path to input image folder",
    )
    args = ap.parse_args()
    folder_path = args.image_folder
    main(folder_path)
