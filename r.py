import tensorflow as tf
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def __rotate_image(self, image: tf.Tensor, theta: float) -> tf.Tensor:
    """
    Rotates an image by an arbitrary angle theta (in radians) around its center using TensorFlow ops.

    Args:
        image (tf.Tensor): 3D tensor representing the image (height, width, channels).
        theta (float): The angle in radians by which to rotate the image.

    Returns:
        tf.Tensor: The rotated image.
    """
    # Get image dimensions
    img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]
    
    # Calculate the cosine and sine of the rotation angle
    cos_theta = tf.math.cos(theta)
    sin_theta = tf.math.sin(theta)
    
    # Get the center of the image
    center_x = tf.cast(img_width, dtype=tf.float32) / 2.0
    center_y = tf.cast(img_height, dtype=tf.float32) / 2.0

    # Construct the transformation matrix with the center offset (8 elements)
    transform = [
        cos_theta, -sin_theta, (1 - cos_theta) * center_x + sin_theta * center_y,
        sin_theta, cos_theta, (1 - cos_theta) * center_y - sin_theta * center_x,
        0.0, 0.0  # The last two elements are for the 3rd row (not used in 2D affine transforms)
    ]
    
    # Apply the transformation using ImageProjectiveTransformV3
    transformed_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, axis=0),
        transforms=tf.convert_to_tensor([transform]),  # Needs to be a batch of transforms
        output_shape=[img_height, img_width],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0
    )
    
    # Remove the added batch dimension
    return tf.squeeze(transformed_image, axis=0)

def __translate_image(self, image: tf.Tensor, h: float, w: float) -> tf.Tensor:
    """
    Translates an image by vertical and horizontal offsets.

    Args:
        image (tf.Tensor): 3D tensor representing the image (height, width, channels).
        h (float): Vertical translation offset.
        w (float): Horizontal translation offset.

    Returns:
        tf.Tensor: The translated image.
    """
    # Get image dimensions
    img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]
    
    # Construct the translation matrix (8 elements)
    transform = [
        1.0, 0.0, w,  # No scaling or rotation, just horizontal translation
        0.0, 1.0, h,  # No scaling or rotation, just vertical translation
        0.0, 0.0  # The last two elements are for the 3rd row (not used in 2D affine transforms)
    ]
    
    # Apply the translation using ImageProjectiveTransformV3
    translated_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, axis=0),
        transforms=tf.convert_to_tensor([transform]),  # Needs to be a batch of transforms
        output_shape=[img_height, img_width],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0
    )
    
    # Remove the added batch dimension
    return tf.squeeze(translated_image, axis=0)

def __zoom_image(self, image: tf.Tensor, z: float) -> tf.Tensor:
    """
    Zooms an image by a factor of z. Positive values zoom in, negative values zoom out.
    The result image has the same size as the original image.

    Args:
        image (tf.Tensor): 3D tensor representing the image (height, width, channels).
        z (float): Zoom factor. Positive values zoom in, negative values zoom out.

    Returns:
        tf.Tensor: The zoomed image with the same size as the original image.
    """
    img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]
    
    # Cast height and width to float for multiplication
    img_height_float = tf.cast(img_height, tf.float32)
    img_width_float = tf.cast(img_width, tf.float32)

    # Calculate new size based on zoom factor
    if z > 0:  # Zoom in
        new_height = tf.cast(img_height_float * (1 + z), tf.int32)
        new_width = tf.cast(img_width_float * (1 + z), tf.int32)
        resized_image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        
        # Crop the center of the resized image to match original size
        cropped_image = tf.image.resize_with_crop_or_pad(resized_image, img_height, img_width)
        return cropped_image
    
    elif z < 0:  # Zoom out (dezoom)
        new_height = tf.cast(img_height_float * (1 + z), tf.int32)
        new_width = tf.cast(img_width_float * (1 + z), tf.int32)
        resized_image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        
        # Pad the resized image to match original size
        padded_image = tf.image.resize_with_crop_or_pad(resized_image, img_height, img_width)
        return padded_image

    else:  # No zoom (z == 0)
        return image

def load_image(image_path: str) -> tf.Tensor:
    """
    Loads an image from the given path and returns it as a TensorFlow tensor.

    Args:
        image_path (str): The file path of the image.

    Returns:
        tf.Tensor: The image tensor.
    """
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure it's in RGB mode
    img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def show_image(image_tensor: tf.Tensor):
    """
    Displays a TensorFlow tensor as an image.

    Args:
        image_tensor (tf.Tensor): The image tensor to display.
    """
    image_tensor = tf.clip_by_value(image_tensor, 0.0, 1.0)  # Ensure the values are in [0, 1] range
    img_array = image_tensor.numpy()  # Convert tensor to numpy array
    plt.imshow(img_array)
    plt.axis('off')  # Hide axes
    plt.show()

def load_image(image_path: str) -> tf.Tensor:
    """
    Loads an image from the given path and returns it as a TensorFlow tensor.

    Args:
        image_path (str): The file path of the image.

    Returns:
        tf.Tensor: The image tensor.
    """
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure it's in RGB mode
    img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def show_image(image_tensor: tf.Tensor):
    """
    Displays a TensorFlow tensor as an image.

    Args:
        image_tensor (tf.Tensor): The image tensor to display.
    """
    image_tensor = tf.clip_by_value(image_tensor, 0.0, 1.0)  # Ensure the values are in [0, 1] range
    img_array = image_tensor.numpy()  # Convert tensor to numpy array
    plt.imshow(img_array)
    plt.axis('off')  # Hide axes
    plt.show()

# Example usage
image_path = 'resources/000.png'
image = load_image(image_path)

# Rotate the image by theta = pi/4 radians (45 degrees)
theta = math.pi / 4
rotated_image = __rotate_image(None, image, theta)

# Visualize the rotated image
show_image(rotated_image)

# Rotate the image by theta = 2pi/4 radians (45 degrees)
theta = 2 * math.pi / 4
rotated_image = __rotate_image(None, image, theta)

# Visualize the rotated image
show_image(rotated_image)

# Rotate the image by theta = 3pi/4 radians (45 degrees)
theta = 3 * math.pi / 4
rotated_image = __rotate_image(None, image, theta)

# Visualize the rotated image
show_image(rotated_image)

# Rotate the image by theta = 4pi/4 radians (45 degrees)
theta = 4 * math.pi / 4
rotated_image = __rotate_image(None, image, theta)

# Visualize the rotated image
show_image(rotated_image)

# Example usage
image_path = 'resources/000.png'
image = load_image(image_path)

# Define vertical (h) and horizontal (w) translation values
h = 250  # Vertical translation (positive = down, negative = up)
w = -100  # Horizontal translation (positive = right, negative = left)

# Apply the translation
translated_image = __translate_image(None, image, h, w)

# Visualize the translated image
show_image(translated_image)

# Example usage
image_path = 'resources/000.png'
image = load_image(image_path)

# Zoom factor (positive for zoom in, negative for zoom out)
zoom_factor = 0.5  # Example for zoom in (positive)
# zoom_factor = -0.5  # Example for zoom out (negative)

# Apply the zoom
zoomed_image = __zoom_image(None, image, zoom_factor)

# Visualize the zoomed image
show_image(zoomed_image)

# Example usage
image_path = 'resources/000.png'
image = load_image(image_path)

# Zoom factor (positive for zoom in, negative for zoom out)
# zoom_factor = 0.5  # Example for zoom in (positive)
zoom_factor = -0.5  # Example for zoom out (negative)

# Apply the zoom
zoomed_image = __zoom_image(None, image, zoom_factor)

# Visualize the zoomed image
show_image(zoomed_image)

plt.close('all')
