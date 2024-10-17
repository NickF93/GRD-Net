import pytest
import os
import tempfile
import shutil
import logging
from typing import Generator

import tensorflow as tf
import numpy as np
from PIL import Image
from src.data import image_dataset_from_directory
from src.loss import huber_loss, mse_loss, mae_loss, ssim_loss, bce_loss, focal_loss
from src.model import BottleNeckType, build_res_ae

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
    _, dataset = image_dataset_from_directory(
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
            i, str(tuple(image.shape)), cls[0], label_str, str(tuple(mask.shape)), id[0], path_str
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

@pytest.fixture
def y_true():
    return tf.random.normal((2, 16, 16, 3))

@pytest.fixture
def y_pred():
    return tf.random.normal((2, 16, 16, 3))

def test_huber_loss_mean(y_true, y_pred):
    delta = 1.0

    # Custom Huber loss with 'mean' reduction
    custom_loss = huber_loss(y_true, y_pred, delta=delta, reduction='mean')

    # TensorFlow's built-in Huber loss with 'mean' reduction
    tf_loss = tf.reduce_mean(tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)(tf.reshape(y_true, (*(y_true.shape), 1)), tf.reshape(y_pred, (*(y_pred.shape), 1))))

    assert tf.abs(custom_loss - tf_loss).numpy() < 1e-6

def test_mse_loss_mean(y_true, y_pred):
    # Custom MSE loss with 'mean' reduction
    custom_loss = mse_loss(y_true, y_pred, reduction='mean')

    # TensorFlow's built-in MSE loss with 'mean' reduction
    tf_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(tf.reshape(y_true, (*(y_true.shape), 1)), tf.reshape(y_pred, (*(y_pred.shape), 1))))

    assert tf.abs(custom_loss - tf_loss).numpy() < 1e-6

def test_mae_loss_mean(y_true, y_pred):
    # Custom MAE loss with 'mean' reduction
    custom_loss = mae_loss(y_true, y_pred, reduction='mean')

    # TensorFlow's built-in MAE loss with 'mean' reduction
    tf_loss = tf.reduce_mean(tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)(tf.reshape(y_true, (*(y_true.shape), 1)), tf.reshape(y_pred, (*(y_pred.shape), 1))))

    assert tf.abs(custom_loss - tf_loss).numpy() < 1e-6

def test_ssim_loss_mean(y_true, y_pred):
    """
    Test the custom SSIM loss function with 'mean' reduction and compare it with TensorFlow's SSIM implementation.
    """
    # Custom SSIM loss with 'mean' reduction
    custom_loss = ssim_loss(y_true, y_pred, reduction='mean')

    # TensorFlow's SSIM implementation, scaled to loss (1 - SSIM)
    tf_ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    tf_loss = tf.reduce_mean(1 - tf_ssim)

    assert tf.abs(custom_loss - tf_loss).numpy() < 1e-6

def test_bce_loss_mean(y_true, y_pred):
    """
    Test the custom BCE loss function with 'mean' reduction and compare it with TensorFlow's built-in BCE implementation.
    """

    y_true = tf.math.abs(y_true)
    y_true = tf.clip_by_value(((y_true - tf.reduce_min(y_true)) / (tf.reduce_max(y_true) - tf.reduce_min(y_true))), 0.0, 1.0)
    y_true = tf.cast(y_true >= 0.5, dtype=tf.float32)

    # Custom BCE loss with 'mean' reduction
    custom_loss = bce_loss(y_true, y_pred, from_logits=True, reduction='mean')

    # TensorFlow's built-in BCE loss
    tf_loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))

    assert tf.abs(custom_loss - tf_loss).numpy() < 1e-6


def test_focal_loss_mean(y_true, y_pred):
    """
    Test the custom BCE loss function with 'mean' reduction and compare it with TensorFlow's built-in BCE implementation.
    """

    alpha: float = 2.0
    gamma: float = 0.25

    y_true = tf.math.abs(y_true)
    y_true = tf.clip_by_value(((y_true - tf.reduce_min(y_true)) / (tf.reduce_max(y_true) - tf.reduce_min(y_true))), 0.0, 1.0)
    y_true = tf.cast(y_true >= 0.5, dtype=tf.float32)

    # Custom focal loss with 'mean' reduction
    custom_loss = focal_loss(y_true, y_pred, from_logits=True, alpha=alpha, gamma=gamma, reduction='mean')

    # TensorFlow's built-in focal loss
    tf_loss = tf.reduce_mean(tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=True, alpha=alpha, gamma=gamma, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))

    # Compare results step by step
    print("Custom loss:", custom_loss.numpy())
    print("TensorFlow loss:", tf_loss.numpy())

    assert tf.abs(custom_loss - tf_loss).numpy() < 1e-6


def test_focal_bce_loss_mean(y_true, y_pred):
    """
    Test the custom BCE loss function with 'mean' reduction and compare it with TensorFlow's built-in BCE implementation.
    """

    alpha: float = 0.5
    gamma: float = 0.0

    y_true = tf.math.abs(y_true)
    y_true = tf.clip_by_value(((y_true - tf.reduce_min(y_true)) / (tf.reduce_max(y_true) - tf.reduce_min(y_true))), 0.0, 1.0)
    y_true = tf.cast(y_true >= 0.5, dtype=tf.float32)

    # Custom focal loss with 'mean' reduction
    custom_loss = focal_loss(y_true, y_pred, from_logits=True, alpha=alpha, gamma=gamma, apply_class_balancing=False, reduction='mean')

    # TensorFlow's built-in focal loss
    tf_loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))

    # Compare results step by step
    print("Custom loss:", custom_loss.numpy())
    print("TensorFlow loss:", tf_loss.numpy())

    assert tf.abs(custom_loss - tf_loss).numpy() < 1e-6


def test_drae():
    batch_size: int = 8
    img_size: int = 224
    channels: int = 3

    encoder_model: tf.keras.models.Model
    autencoder_model: tf.keras.models.Model
    generator_model: tf.keras.models.Model
    encoder_model, autencoder_model, generator_model = build_res_ae(img_height = img_size, channels = channels, bottleneck_type = BottleNeckType.DENSE, initial_padding=10, initial_padding_filters=64)

    tf.keras.utils.plot_model(
        encoder_model,
        to_file="/tmp/drae_encoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        autencoder_model,
        to_file="/tmp/drae_autencoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        generator_model,
        to_file="/tmp/drae_generator_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    x = tf.random.normal((batch_size, img_size, img_size, channels))
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

    generator_model(x)
    autencoder_model(x)
    encoder_model(x)

    assert tuple(encoder_model.outputs[0].shape) == tuple(autencoder_model.outputs[1].shape)
    assert (None, img_size, img_size, channels) == tuple(autencoder_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[2].shape)
    assert (None, img_size, img_size, channels) == tuple(generator_model.outputs[1].shape)


def test_crae():
    batch_size: int = 8
    img_size: int = 224
    channels: int = 3

    encoder_model: tf.keras.models.Model
    autencoder_model: tf.keras.models.Model
    generator_model: tf.keras.models.Model
    encoder_model, autencoder_model, generator_model = build_res_ae(img_height = img_size, channels = channels, bottleneck_type = BottleNeckType.CONVOLUTIONAL, initial_padding=10, initial_padding_filters=64)

    tf.keras.utils.plot_model(
        encoder_model,
        to_file="/tmp/crae_encoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        autencoder_model,
        to_file="/tmp/crae_autencoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        generator_model,
        to_file="/tmp/crae_generator_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    x = tf.random.normal((batch_size, img_size, img_size, channels))
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

    generator_model(x)
    autencoder_model(x)
    encoder_model(x)

    assert tuple(encoder_model.outputs[0].shape) == tuple(autencoder_model.outputs[1].shape)
    assert (None, img_size, img_size, channels) == tuple(autencoder_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[2].shape)
    assert (None, img_size, img_size, channels) == tuple(generator_model.outputs[1].shape)


def test_wdrae():
    batch_size: int = 8
    img_size: int = 224
    channels: int = 3

    encoder_model: tf.keras.models.Model
    autencoder_model: tf.keras.models.Model
    generator_model: tf.keras.models.Model
    encoder_model, autencoder_model, generator_model = build_res_ae(img_height = img_size, channels = channels, bottleneck_type = BottleNeckType.DENSE, initial_padding=10, initial_padding_filters=64, wide=2)

    tf.keras.utils.plot_model(
        encoder_model,
        to_file="/tmp/drae_encoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        autencoder_model,
        to_file="/tmp/drae_autencoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        generator_model,
        to_file="/tmp/drae_generator_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    x = tf.random.normal((batch_size, img_size, img_size, channels))
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

    generator_model(x)
    autencoder_model(x)
    encoder_model(x)

    assert tuple(encoder_model.outputs[0].shape) == tuple(autencoder_model.outputs[1].shape)
    assert (None, img_size, img_size, channels) == tuple(autencoder_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[2].shape)
    assert (None, img_size, img_size, channels) == tuple(generator_model.outputs[1].shape)


def test_wcrae():
    batch_size: int = 8
    img_size: int = 224
    channels: int = 3

    encoder_model: tf.keras.models.Model
    autencoder_model: tf.keras.models.Model
    generator_model: tf.keras.models.Model
    encoder_model, autencoder_model, generator_model = build_res_ae(img_height = img_size, channels = channels, bottleneck_type = BottleNeckType.CONVOLUTIONAL, initial_padding=10, initial_padding_filters=64, wide=2)

    tf.keras.utils.plot_model(
        encoder_model,
        to_file="/tmp/crae_encoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        autencoder_model,
        to_file="/tmp/crae_autencoder_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    tf.keras.utils.plot_model(
        generator_model,
        to_file="/tmp/crae_generator_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    x = tf.random.normal((batch_size, img_size, img_size, channels))
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

    generator_model(x)
    autencoder_model(x)
    encoder_model(x)

    assert tuple(encoder_model.outputs[0].shape) == tuple(autencoder_model.outputs[1].shape)
    assert (None, img_size, img_size, channels) == tuple(autencoder_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[0].shape)
    assert tuple(encoder_model.outputs[0].shape) == tuple(generator_model.outputs[2].shape)
    assert (None, img_size, img_size, channels) == tuple(generator_model.outputs[1].shape)
