import pytest
import os
import tempfile
import shutil
import logging
from typing import Generator

import tensorflow as tf
import numpy as np
from PIL import Image
from src import data

# Configure logging to debug level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def create_tmp_dataset() -> Generator[str, None, None]:
    """
    Pytest fixture to create a temporary dataset directory structure
    with sample images and masks for testing. The temporary dataset includes:
    - A 'good' class with a white image.
    - A 'bad' class with a black image and a corresponding white mask.

    Yields:
        str: Path to the temporary dataset directory.
    """
    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp(prefix="tmp_dataset_")

    try:
        # Define paths for test and mask directories
        test_good_dir = os.path.join(tmp_dir, "test", "good")
        test_bad_dir = os.path.join(tmp_dir, "test", "bad")
        mask_bad_dir = os.path.join(tmp_dir, "mask", "bad")

        # Create directories for 'good' and 'bad' classes, and mask directory
        os.makedirs(test_good_dir)
        os.makedirs(test_bad_dir)
        os.makedirs(mask_bad_dir)

        # Create a white image (255 for each pixel, size 128x128) for the 'good' class
        white_image = np.full((128, 128, 3), 255, dtype=np.uint8)
        white_image_path = os.path.join(test_good_dir, "0.png")
        Image.fromarray(white_image).save(white_image_path)

        # Create a black image (0 for each pixel, size 128x128) for the 'bad' class
        black_image = np.zeros((128, 128, 3), dtype=np.uint8)
        black_image_path = os.path.join(test_bad_dir, "1.png")
        Image.fromarray(black_image).save(black_image_path)

        # Create a white mask image for the 'bad' class (255 for each pixel)
        white_mask = np.full((128, 128), 255, dtype=np.uint8)  # Single-channel mask
        mask_image_path = os.path.join(mask_bad_dir, "1_mask.png")
        Image.fromarray(white_mask).save(mask_image_path)

        # Yield the temp directory for use in the test
        yield tmp_dir

    finally:
        # Cleanup the temporary directory after the test is done
        shutil.rmtree(tmp_dir)


def test_create_dataset(create_tmp_dataset: str) -> None:
    """
    Test the dataset creation and verify the image and mask contents for 'good'
    and 'bad' classes. The test ensures the following:
    - 'good' class images are white (255) and have an empty mask (0).
    - 'bad' class images are black (0) and have a white mask (255).
    
    Args:
        create_tmp_dataset (str): Path to the temporary dataset directory provided by the fixture.
    """
    tmp_dir = create_tmp_dataset

    # Check if the directories and files are created correctly
    assert os.path.exists(os.path.join(tmp_dir, "test", "good", "0.png"))
    assert os.path.exists(os.path.join(tmp_dir, "test", "bad", "1.png"))
    assert os.path.exists(os.path.join(tmp_dir, "mask", "bad", "1_mask.png"))

    # Call the externally defined image_dataset_from_directory function from 'data'
    _, dataset = data.image_dataset_from_directory(
        directory=os.path.join(tmp_dir, 'test'),  # Point to the test directory
        color_mode='rgb',  # Load images in RGB mode
        batch_size=1,  # Process one image per batch
        shuffle=True,  # Shuffle the dataset
        reshuffle=False,  # Do not reshuffle after every epoch
        load_masks=True,  # Load masks associated with the images
        mask_dir=os.path.join(tmp_dir, 'mask'),  # Point to the mask directory
        mask_ext='mask',  # Define the mask file extension
        interpolation='bilinear'  # Use bilinear interpolation for image resizing
    )
    
    # Iterate over the dataset to check image and mask consistency
    for i, inputs in enumerate(dataset):
        image, cls, lbl, id, path, mask = inputs

        # Decode the label and path strings from TensorFlow tensors
        label_str: str = lbl[0].numpy().decode('utf-8')
        path_str: str = path[0].numpy().decode('utf-8')

        # Log the loaded information for each image
        logger.info(
            '%d) Loaded image of shape %s belonging to class %d, with label "%s", '
            'with a mask of shape %s - ID %d - path "%s"',
            i, str(tuple(image.shape)), cls, label_str, str(tuple(mask.shape)), id, path_str
        )
        
        # Perform checks to ensure correct image and mask content based on class
        if cls == 1:  # 'good' class: image should be white (255) and mask should be empty (0)
            assert np.all(image.numpy() == 255), (
                f"Expected all pixels in the image to be 255 for 'good' class, got {np.unique(image.numpy())}"
            )
            assert np.all(mask.numpy() == 0), (
                f"Expected all pixels in the mask to be 0 for 'good' class, got {np.unique(mask.numpy())}"
            )
        elif cls == 0:  # 'bad' class: image should be black (0) and mask should be white (255)
            assert np.all(image.numpy() == 0), (
                f"Expected all pixels in the image to be 0 for 'bad' class, got {np.unique(image.numpy())}"
            )
            assert np.all(mask.numpy() == 255), (
                f"Expected all pixels in the mask to be 255 for 'bad' class, got {np.unique(mask.numpy())}"
            )
