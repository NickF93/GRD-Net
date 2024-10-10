"""
This module defines the AugmentPipe class for applying random data augmentations to images and masks,
including rotations, translations, zooms, and flips. The augmentations are applied based on the parameters
set during initialization.

The class supports applying augmentations with specific probabilities and ranges,
making it suitable for data augmentation in deep learning pipelines.
"""

import random
from typing import Union, Optional, Tuple
import tensorflow as tf


class AugmentPipe:
    """
    A data augmentation pipeline for applying random transformations such as rotation, translation,
    zoom, and flipping to images and masks. Each transformation is controlled by parameters set during
    initialization.

    Args:
        random_90_rotation (Optional[int]): The number of 90-degree rotations to apply, None or < 1 disables this rotation.
        rotation_angle (Optional[Union[float, Tuple[float, float]]]): A single angle or a range of angles (in radians) for arbitrary rotations. None disables this rotation.
        flip_mode (Optional[str]): Specifies the flip mode, one of ["none", "both", "vertical", "horizontal"]. None disables flipping.
        translation_range (Union[float, Tuple[float, float]]): A single value or a range of values for random translations. None disables translation.
        zoom_range (Union[float, Tuple[float, float]]): A single value or a range of values for random zooming. None disables zooming.

    Attributes:
        __random_90_rotation (bool): Flag to enable or disable random 90-degree rotations.
        __random_90_rotation_fact (int): Number of possible 90-degree rotations.
        __rotate (bool): Flag to enable or disable arbitrary rotation.
        __rotate_thetas (Optional[Tuple[float, float]]): Range of angles for arbitrary rotation.
        __flip_v (bool): Flag to enable or disable vertical flipping.
        __flip_h (bool): Flag to enable or disable horizontal flipping.
        __translate (bool): Flag to enable or disable translation.
        __translate_range (Optional[Tuple[float, float]]): Range of values for random translation.
        __zoom (bool): Flag to enable or disable zooming.
        __zoom_range (Optional[Tuple[float, float]]): Range of values for random zooming.
    """

    def __init__(self,
                 random_90_rotation: Optional[int],
                 rotation_angle: Optional[Union[float, Tuple[float, float]]],
                 flip_mode: Optional[str],
                 translation_range: Union[float, Tuple[float, float]],
                 zoom_range: Union[float, Tuple[float, float]]):
        """
        Initializes the AugmentPipe class with the desired augmentation parameters.

        Args:
            random_90_rotation (Optional[int]): Number of 90-degree rotations to apply (0, 90, 180, or 270 degrees). None or < 1 disables the rotation.
            rotation_angle (Optional[Union[float, Tuple[float, float]]]): A single angle (float) or a range (tuple) in radians for arbitrary rotation. None disables the rotation.
            flip_mode (Optional[str]): Mode of flipping, one of ["none", "both", "vertical", "horizontal"]. None disables flipping.
            translation_range (Union[float, Tuple[float, float]]): Translation range for both x and y axes. A single float applies the same translation in both directions; a tuple allows specifying a range for random translation.
            zoom_range (Union[float, Tuple[float, float]]): Zoom range for scaling the image. A single float applies a fixed zoom, while a tuple allows random zooming within the range.

        Raises:
            ValueError: If flip_mode is not in ["none", "both", "vertical", "horizontal"].
        """
        # Configure random 90-degree rotation
        self.__random_90_rotation: bool
        self.__random_90_rotation_fact: int
        if random_90_rotation is None or random_90_rotation < 1:
            self.__random_90_rotation = False
            self.__random_90_rotation_fact = 0
        else:
            self.__random_90_rotation = True
            self.__random_90_rotation_fact = random_90_rotation

        # Configure arbitrary rotation
        self.__rotate: bool
        self.__rotate_thetas: Optional[Tuple[float, float]]
        if rotation_angle is None:
            self.__rotate = False
            self.__rotate_thetas = None
        else:
            self.__rotate = True
            self.__rotate_thetas = [rotation_angle, rotation_angle] if isinstance(rotation_angle, float) \
                                   else tuple(rotation_angle) if isinstance(rotation_angle, tuple) else tuple(rotation_angle)

        # Configure flipping mode
        self.__flip_v: bool
        self.__flip_h: bool
        if flip_mode is None or flip_mode.lower().strip() == 'none':
            self.__flip_v = False
            self.__flip_h = False
        elif flip_mode.lower().strip() == 'both':
            self.__flip_v = True
            self.__flip_h = True
        elif flip_mode.lower().strip() == 'vertical':
            self.__flip_v = True
            self.__flip_h = False
        elif flip_mode.lower().strip() == 'horizontal':
            self.__flip_v = False
            self.__flip_h = True
        else:
            raise ValueError('`flip_mode` must be one of [None, "none", "both", "vertical", "horizontal"]')

        # Configure translation
        self.__translate: bool
        self.__translate_range: Optional[Tuple[float, float]]
        if translation_range is None:
            self.__translate = False
            self.__translate_range = None
        else:
            self.__translate = True
            self.__translate_range = [translation_range, translation_range] if isinstance(translation_range, float) \
                                     else tuple(translation_range)

        # Configure zoom
        self.__zoom: bool
        self.__zoom_range: Optional[Tuple[float, float]]
        if zoom_range is None:
            self.__zoom = False
            self.__zoom_range = None
        else:
            self.__zoom = True
            self.__zoom_range = [zoom_range, zoom_range] if isinstance(zoom_range, float) \
                                else tuple(zoom_range)

    def __rotate_image(self, image: tf.Tensor, theta: float, mask: bool = False) -> tf.Tensor:
        """
        Rotates an image by an arbitrary angle theta (in radians) around its center using TensorFlow ops.

        Args:
            image (tf.Tensor): 3D tensor representing the image (height, width, channels).
            theta (float): The angle in radians by which to rotate the image.
            mask (bool): If True, treats the image as a mask and uses 'CONSTANT' fill mode.

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

        # Construct the transformation matrix for rotation
        transform = [
            cos_theta, -sin_theta, (1 - cos_theta) * center_x + sin_theta * center_y,
            sin_theta, cos_theta, (1 - cos_theta) * center_y - sin_theta * center_x,
            0.0, 0.0  # The last two elements are placeholders for 2D affine transforms
        ]

        # Apply the transformation using ImageProjectiveTransformV3
        transformed_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, axis=0),
            transforms=tf.convert_to_tensor([transform]),  # Must be a batch of transforms
            output_shape=[img_height, img_width],
            interpolation="BILINEAR",
            fill_mode="NEAREST" if not mask else "CONSTANT",
            fill_value=0.0
        )

        return tf.squeeze(transformed_image, axis=0)

    def __translate_image(self, image: tf.Tensor, h: float, w: float, mask: bool = False) -> tf.Tensor:
        """
        Translates an image by vertical and horizontal offsets.

        Args:
            image (tf.Tensor): 3D tensor representing the image (height, width, channels).
            h (float): Vertical translation offset.
            w (float): Horizontal translation offset.
            mask (bool): If True, treats the image as a mask and uses 'CONSTANT' fill mode.

        Returns:
            tf.Tensor: The translated image.
        """
        # Get image dimensions
        img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]

        # Construct the translation matrix
        transform = [
            1.0, 0.0, w,  # Horizontal translation
            0.0, 1.0, h,  # Vertical translation
            0.0, 0.0  # Placeholder for 2D affine transforms
        ]

        # Apply the translation using ImageProjectiveTransformV3
        translated_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, axis=0),
            transforms=tf.convert_to_tensor([transform]),  # Must be a batch of transforms
            output_shape=[img_height, img_width],
            interpolation="BILINEAR",
            fill_mode="NEAREST" if not mask else "CONSTANT",
            fill_value=0.0
        )

        return tf.squeeze(translated_image, axis=0)

    def __rotate_90(self, image: tf.Tensor, k: int) -> tf.Tensor:
        """
        Rotates an image by multiples of 90 degrees.

        Args:
            image (tf.Tensor): 3D tensor representing the image.
            k (int): The number of 90-degree rotations (clockwise).

        Returns:
            tf.Tensor: The rotated image.
        """
        return tf.image.rot90(image, k=k)

    def __zoom_image(self, image: tf.Tensor, z: float) -> tf.Tensor:
        """
        Zooms an image by a factor of z. Positive values zoom in, negative values zoom out.

        Args:
            image (tf.Tensor): 3D tensor representing the image (height, width, channels).
            z (float): Zoom factor. Positive values zoom in, negative values zoom out.

        Returns:
            tf.Tensor: The zoomed image, resized to the original dimensions.
        """
        img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]
        img_height_float = tf.cast(img_height, tf.float32)
        img_width_float = tf.cast(img_width, tf.float32)

        if z > 0:  # Zoom in
            new_height = tf.cast(img_height_float * (1 + z), tf.int32)
            new_width = tf.cast(img_width_float * (1 + z), tf.int32)
            resized_image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
            return tf.image.resize_with_crop_or_pad(resized_image, img_height, img_width)
        elif z < 0:  # Zoom out (dezoom)
            new_height = tf.cast(img_height_float * (1 + z), tf.int32)
            new_width = tf.cast(img_width_float * (1 + z), tf.int32)
            resized_image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
            return tf.image.resize_with_crop_or_pad(resized_image, img_height, img_width)
        else:  # No zoom
            return image

    def __h_flip(self, image: tf.Tensor) -> tf.Tensor:
        """
        Performs a horizontal flip on the image.

        Args:
            image (tf.Tensor): 3D tensor representing the image.

        Returns:
            tf.Tensor: The horizontally flipped image.
        """
        return tf.image.flip_left_right(image)

    def __v_flip(self, image: tf.Tensor) -> tf.Tensor:
        """
        Performs a vertical flip on the image.

        Args:
            image (tf.Tensor): 3D tensor representing the image.

        Returns:
            tf.Tensor: The vertically flipped image.
        """
        return tf.image.flip_up_down(image)

    def __call__(self, image: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Applies the specified augmentations to the input image (and mask, if provided) in sequence.

        Args:
            image (tf.Tensor): The image to augment (3D tensor).
            mask (Optional[tf.Tensor]): The mask to augment (3D tensor), if provided.

        Returns:
            Tuple[tf.Tensor, Optional[tf.Tensor]]: The augmented image and mask (if provided).
        """
        aug_image = image
        aug_mask = mask

        if self.__random_90_rotation:
            k = random.randint(0, self.__random_90_rotation_fact)
            aug_image = self.__rotate_90(aug_image, k)
            if aug_mask is not None:
                aug_mask = self.__rotate_90(aug_mask, k)

        if self.__rotate:
            theta = random.uniform(self.__rotate_thetas[0], self.__rotate_thetas[1])
            aug_image = self.__rotate_image(aug_image, theta)
            if aug_mask is not None:
                aug_mask = self.__rotate_image(aug_mask, theta, mask=True)

        if self.__translate:
            h_range = random.uniform(-self.__translate_range[0], self.__translate_range[0])
            w_range = random.uniform(-self.__translate_range[1], self.__translate_range[1])
            aug_image = self.__translate_image(aug_image, h_range, w_range)
            if aug_mask is not None:
                aug_mask = self.__translate_image(aug_mask, h_range, w_range, mask=False)

        if self.__zoom:
            z_range = random.uniform(-self.__zoom_range[0], self.__zoom_range[1])
            aug_image = self.__zoom_image(aug_image, z_range)
            if aug_mask is not None:
                aug_mask = self.__zoom_image(aug_mask, z_range)

        if self.__flip_h:
            if random.random() < 0.5:
                aug_image = self.__h_flip(aug_image)
                if aug_mask is not None:
                    aug_mask = self.__h_flip(aug_mask)

        if self.__flip_v:
            if random.random() < 0.5:
                aug_image = self.__v_flip(aug_image)
                if aug_mask is not None:
                    aug_mask = self.__v_flip(aug_mask)

        return aug_image, aug_mask
