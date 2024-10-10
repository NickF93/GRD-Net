import os
import time
from typing import Tuple, Optional, Union
from enum import Enum

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ..data import image_dataset_from_directory
from ..augment import AugmentPipe
from ..loss import mae_loss, mse_loss, huber_loss, bce_loss, focal_loss, ssim_loss, ssim_rgb_loss, cosine_similarity_loss

class NetType(Enum):
    GRD = 0
    ResGAN = 1

class Trainer:
    def __init__(self,
                name: str,
                net_type: NetType,
                batch_size: int,
                channels: int,
                train_and_validation_path: str,
                train_and_validation_roi_path: str,
                validation_split: float,
                test_path: str,
                mask_path: str,
                patch_size: Tuple[int, int], 
                patches_row: int,
                patches_col: int, 
                stride: Tuple[int, int],
                padding: str = 'VALID',
                mask_suffix: str = 'mask',
                random_90_rotation: Optional[int] = 3,
                rotation_angle: Optional[Union[float, Tuple[float, float]]] = [-np.pi, np.pi],
                flip_mode: Optional[str] = 'both',
                translation_range: Union[float, Tuple[float, float]] = None,
                zoom_range: Union[float, Tuple[float, float]] = [-0.1, 0.1]
            ):
        self.name = str(name)
        self.net_type: NetType = net_type
        self.batch_size: int = batch_size
        self.patch_size: Tuple[int, int] = patch_size
        self.patches_row: int = patches_row
        self.patches_col: int = patches_col
        self.stride: Tuple[int, int] = stride
        self.padding: str = padding
        self.target_size: Tuple[int, int] = self.calculate_target_size(patch_size, patches_row, patches_col, stride, padding)
        self.channels: int = channels
        self.validation_split: float = validation_split
        self.mask_suffix: str = mask_suffix
        self.aug_pipe: AugmentPipe = AugmentPipe(random_90_rotation=random_90_rotation, rotation_angle=rotation_angle, flip_mode=flip_mode, translation_range=translation_range, zoom_range=zoom_range)

        self.train_and_validation_path: str = os.path.realpath(train_and_validation_path)
        assert os.path.exists(self.train_and_validation_path) and os.path.isdir(self.train_and_validation_path), 'train and validation path must exist and must be a directory'

        self.train_and_validation_roi_path: str = os.path.realpath(train_and_validation_roi_path)
        assert os.path.exists(self.train_and_validation_roi_path) and os.path.isdir(self.train_and_validation_roi_path), 'train and validation ROI path must exist and must be a directory'

        self.test_path: str = os.path.realpath(test_path)
        assert os.path.exists(self.test_path) and os.path.isdir(self.test_path), 'test path must exist and must be a directory'

        self.mask_path: str = os.path.realpath(mask_path)
        assert os.path.exists(self.mask_path) and os.path.isdir(self.mask_path), 'mask path must exist and must be a directory'

        _, self.ds_training_path, _, self.ds_validation_path = image_dataset_from_directory(
                                        directory=self.train_and_validation_path,
                                        color_mode=self.get_image_type(),
                                        batch_size=self.batch_size,
                                        image_size=self.target_size,
                                        shuffle=True,
                                        reshuffle=True,
                                        seed=round(time.time() / 100.),
                                        validation_split=self.validation_split if self.validation_split > 0 else None,
                                        subset='both' if self.validation_split > 0 else None,
                                        load_masks=True,
                                        mask_type='roi',
                                        mask_dir=self.train_and_validation_roi_path,
                                        mask_ext=self.mask_suffix,
                                        samples=None)

        _, self.ds_test_path = image_dataset_from_directory(
                                        self.test_path,
                                        color_mode=self.get_image_type(),
                                        batch_size=self.batch_size,
                                        image_size=self.target_size,
                                        shuffle=False,
                                        reshuffle=False,
                                        load_masks=True,
                                        mask_type='mask',
                                        mask_dir=self.mask_path,
                                        mask_ext=self.mask_suffix)
        
        normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        self.ds_training_path = self.ds_training_path.map(lambda x, y, l, i, p, m: (normalization_layer(x), normalization_layer(m)))
        normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        self.ds_validation_path = self.ds_validation_path.map(lambda x, y, l, i, p, m: (normalization_layer(x), normalization_layer(m)))
        normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        self.ds_test_path = self.ds_test_path.map(lambda x, y, l, i, p, m: (normalization_layer(x), normalization_layer(m)))


    def show_first_batch_images_and_masks(self, train: bool = True, augment: bool = False):
        """
        Displays the first batch of images and corresponding masks from the dataset.
        
        Args:
            ds_test_path: A TensorFlow dataset that returns a batch of images and masks.
        """
        # Get the first batch of images and masks
        first_batch = next(iter(self.ds_training_path if train else self.ds_test_path))
        
        # Assuming the dataset returns a tuple of (images, masks)
        images, masks = first_batch
        
        # Batch size (number of images in the batch)
        batch_size = images.shape[0]
        
        # Create a figure to plot images and masks side by side
        fig, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 3))
        
        for i in range(batch_size):
            # Get the image and mask
            img = images[i]
            mask = masks[i]
            
            # Apply augmentation
            augmented_img, augmented_mask = (self.aug_pipe(img, mask) if augment else (img, mask))
            
            # Display the augmented image in column 0
            axes[i, 0].imshow(augmented_img.numpy())
            axes[i, 0].set_title(f"Augmented Image {i + 1}")
            axes[i, 0].axis('off')
            
            # Display the corresponding augmented mask in column 1
            axes[i, 1].imshow(tf.image.grayscale_to_rgb(augmented_mask).numpy())
            axes[i, 1].set_title(f"Augmented Mask {i + 1}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        plt.cla()
        plt.clf()
        plt.close(fig)
        plt.close()
        plt.close('all')


    def get_adv_loss_fn(self, w_adv=1.0):
        if self.net_type == NetType.GRD:
            return lambda f_real, f_fake : tf.math.multiply_no_nan(mse_loss(f_real, f_fake, reduction='mean'), w_adv)
        elif self.net_type == NetType.ResGAN:
            return lambda f_real, f_fake : tf.math.multiply_no_nan(mse_loss(f_real, f_fake, reduction='mean'), w_adv)
        else:
            raise ValueError('NetType error')


    def get_con_loss_fn(self, channels, w_con=5.0, w_1=10.0, w_2=1.0):
        ssim_fn = ssim_loss if channels == 1 else ssim_rgb_loss
        if self.net_type == NetType.GRD:
            return lambda xr, xf : tf.math.multiply_no_nan(tf.math.add(tf.math.multiply_no_nan(mae_loss(xr, xf, reduction='mean'), w_1), tf.math.multiply_no_nan(ssim_fn(xr, xf, reduction='mean'), w_2)), w_con)
        elif self.net_type == NetType.ResGAN:
            return lambda xr, xf : tf.math.multiply_no_nan(tf.math.add(tf.math.multiply_no_nan(huber_loss(xr, xf, reduction='mean'), w_1), tf.math.multiply_no_nan(ssim_fn(xr, xf, reduction='mean'), w_2)), w_con)
        else:
            raise ValueError('NetType error')


    def get_lat_loss_fn(self, w_lat=1.0):
        if self.net_type == NetType.GRD:
            return lambda zr, zf : tf.math.multiply_no_nan(mse_loss(zr, zf, reduction='mean'), w_lat)
        elif self.net_type == NetType.ResGAN:
            return lambda zr, zf : tf.math.multiply_no_nan(cosine_similarity_loss(zr, zf, reduction='mean'), w_lat)
        else:
            raise ValueError('NetType error')


    def get_disc_loss_fn(self):
        return lambda pr, pf : tf.math.divide_no_nan(tf.math.add((bce_loss(tf.ones_like(pr), pr, from_logits=False, reduction='mean'), bce_loss(tf.zeros_like(pf), pf, from_logits=False, reduction='mean'))), 2.0)


    def get_image_type(self):
        if self.channels == 1:
            return 'grayscale'
        elif self.channels == 3:
            return 'rgb'
        else:
            raise ValueError('Image type not supported')


    def calculate_target_size(self,
                            patch_size: Tuple[int, int], 
                            patches_row: int, patches_col: int, 
                            stride: Tuple[int, int],
                            padding: str = 'VALID') -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calculate the target size of the original image based on patch size, number of patches per row/col,
        stride, and padding type.

        Args:
            patch_size (Tuple[int, int]): The size of the patch (height, width).
            patches_row (int): The number of patches per column (height-wise).
            patches_col (int): The number of patches per row (width-wise).
            stride (Tuple[int, int]): The stride (height, width) for patch extraction.
            padding (str): Padding type, either 'VALID' or 'SAME'.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The calculated target height and width of the original image.
        """
        ph, pw = patch_size
        stride_h, stride_w = stride
        
        if padding == 'VALID':
            # No padding is added, so the formula is straightforward
            target_height = (patches_row - 1) * stride_h + ph
            target_width = (patches_col - 1) * stride_w + pw

        elif padding == 'SAME':
            # Padding is added to keep the output size same, the formula is adjusted to include padding
            target_height = patches_row * stride_h
            target_width = patches_col * stride_w

        else:
            raise ValueError("Padding must be either 'VALID' or 'SAME'")

        return target_height, target_width
