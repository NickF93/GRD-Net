import os
import datetime
from typing import Tuple, Optional, Union
from enum import Enum
import tempfile
import logging

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import colorama
from tqdm.auto import tqdm

from ..data import image_dataset_from_directory
from ..augment import AugmentPipe
from ..loss import mae_loss, mse_loss, huber_loss, bce_loss, focal_loss, ssim_loss, ssim_rgb_loss, cosine_similarity_loss
from ..util import config_gpu, clear_session, set_seed, LevelNameFormatter, model_logger
from ..perlin import Perlin
from ..experiment_manager import ExperimentManager
from ..model import BottleNeckType, build_res_ae, build_res_disc, build_res_unet

colorama.init()

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Choose the custom formatter
ch.setFormatter(LevelNameFormatter())

# Add the handler to the logger
logger.addHandler(ch)

logger.info('Logger configured')

clear_session()
logger.info('Cleared session')

config_gpu()
logger.info('GPU configured for TensorFlow memory growth')

class NetType(Enum):
    GRD = 0
    ResGAN = 1

class Trainer:
    def __init__(self,
                name: str,
                net_type: NetType,
                batch_size: int,
                channels: int,
                epochs: int,
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
                zoom_range: Union[float, Tuple[float, float]] = [-0.1, 0.1],
                initial_learning_rate: float = 1e-4,
                first_decay_steps: int = 1000,
                t_mul: float = 2.0,
                m_mul: float = 1.0,
                alpha: float = 1e-6,
                log_path: str = tempfile.gettempdir(),
                mlflow_uri: str = 'localhost:5000'
            ):
        self.name = str(name)
        self.net_type: NetType = net_type
        self.batch_size: int = batch_size
        self.epochs: int = epochs
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
        self.log_path: str = log_path

        self.train_and_validation_path: str = os.path.realpath(train_and_validation_path)
        assert os.path.exists(self.train_and_validation_path) and os.path.isdir(self.train_and_validation_path), 'train and validation path must exist and must be a directory'

        self.train_and_validation_roi_path: str = os.path.realpath(train_and_validation_roi_path)
        assert os.path.exists(self.train_and_validation_roi_path) and os.path.isdir(self.train_and_validation_roi_path), 'train and validation ROI path must exist and must be a directory'

        self.test_path: str = os.path.realpath(test_path)
        assert os.path.exists(self.test_path) and os.path.isdir(self.test_path), 'test path must exist and must be a directory'

        self.mask_path: str = os.path.realpath(mask_path)
        assert os.path.exists(self.mask_path) and os.path.isdir(self.mask_path), 'mask path must exist and must be a directory'

        self.seed = set_seed()

        _, self.ds_training_path, _, self.ds_validation_path = image_dataset_from_directory(
                                        directory=self.train_and_validation_path,
                                        color_mode=self.get_image_type(),
                                        batch_size=self.batch_size,
                                        image_size=self.target_size,
                                        shuffle=True,
                                        reshuffle=True,
                                        seed=self.seed,
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

        max_size = max(self.target_size)
        size = int((2 ** np.ceil(np.log2(max_size))).astype(np.int64) * 2)

        _, self.ds_reference_dataset = image_dataset_from_directory(
                                        directory=self.train_and_validation_path,
                                        color_mode=self.get_image_type(),
                                        batch_size=self.batch_size,
                                        image_size=(size, size),
                                        shuffle=True,
                                        reshuffle=True,
                                        seed=self.seed,
                                        load_masks=True,
                                        mask_type='mask',
                                        mask_dir=self.train_and_validation_roi_path,
                                        mask_ext=self.mask_suffix,
                                        samples=None)

        _, self.ds_real_defect_dataset = image_dataset_from_directory(
                                        self.test_path,
                                        color_mode=self.get_image_type(),
                                        batch_size=self.batch_size,
                                        image_size=(size, size),
                                        shuffle=True,
                                        reshuffle=True,
                                        seed=self.seed,
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

        normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        self.ds_reference_dataset = self.ds_reference_dataset.map(lambda x, y, l, i, p, m: (normalization_layer(x), normalization_layer(m)))
        normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        self.ds_real_defect_dataset = self.ds_real_defect_dataset.map(lambda x, y, l, i, p, m: (normalization_layer(x), normalization_layer(m)))

        self.perlin: Perlin = Perlin(size=max_size, target_size=self.target_size, reference_dataset=self.ds_reference_dataset, real_defect_dataset=self.ds_real_defect_dataset, fraction=0.75, choice=0.10, def_choice=0.90, perlin_queue_max=100, perlin_queue_min=100)

        generator_lr_policy = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                            initial_learning_rate   = initial_learning_rate,
                                                            first_decay_steps       = first_decay_steps,
                                                            t_mul                   = t_mul,
                                                            m_mul                   = m_mul,
                                                            alpha                   = alpha
                                                        )
        
        discriminator_lr_policy = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                            initial_learning_rate   = initial_learning_rate,
                                                            first_decay_steps       = first_decay_steps,
                                                            t_mul                   = t_mul,
                                                            m_mul                   = m_mul,
                                                            alpha                   = alpha
                                                        )
        
        segmentator_lr_policy = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                            initial_learning_rate   = initial_learning_rate,
                                                            first_decay_steps       = first_decay_steps,
                                                            t_mul                   = t_mul,
                                                            m_mul                   = m_mul,
                                                            alpha                   = alpha
                                                        )
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr_policy, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr_policy, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        self.segmentator_optimizer = tf.keras.optimizers.Adam(learning_rate=segmentator_lr_policy, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

        self.logdir = self.log_path + '/' + str(self.name) + '_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f_%A-%W-%B"))
        self.manager: ExperimentManager = ExperimentManager(mlflow_uri=mlflow_uri, experiment_name=self.name, tensorboard_logdir=self.logdir + os.sep + 'tensorboard', mlflow_alt_logdir=self.logdir + os.sep + 'mlflow')

        logger.info('Building models...')
        self.encoder_model: tf.keras.models.Model = None
        self.autencoder_model: tf.keras.models.Model = None
        self.generator_model: tf.keras.models.Model = None
        self.discriminator_model: tf.keras.models.Model = None
        self.unet_model: tf.keras.models.Model = None

        self.encoder_model, self.autencoder_model, self.generator_model = build_res_ae(bottleneck_type = BottleNeckType.DENSE, initial_padding=10, initial_padding_filters=64)
        logger.debug('Encoder structure:')
        model_logger(model=self.encoder_model, logger=logger, save_path=tempfile.gettempdir(), print_visualkeras=False)
        logger.debug('Autoencoder structure:')
        model_logger(model=self.autencoder_model, logger=logger, save_path=tempfile.gettempdir(), print_visualkeras=False)
        logger.debug('Generator structure:')
        model_logger(model=self.generator_model, logger=logger, save_path=tempfile.gettempdir(), print_visualkeras=False)

        self.discriminator_model = build_res_disc(initial_padding=10, initial_padding_filters=64)
        logger.debug('Discriminator structure:')
        model_logger(model=self.discriminator_model, logger=logger, save_path=tempfile.gettempdir(), print_visualkeras=False)

        self.unet_model = build_res_unet(skips=4, initial_padding=10, initial_padding_filters=64)
        logger.debug('U-Net structure:')
        model_logger(model=self.unet_model, logger=logger, save_path=tempfile.gettempdir(), print_visualkeras=False)
        
        self.contextual_loss = self.get_con_loss_fn(channels=self.channels, w_con=5.0, w_1=10.0, w_2=1.0)
        self.adversarial_loss = self.get_adv_loss_fn(w_adv=1.0)
        self.latent_loss = self.get_lat_loss_fn(w_lat=1.0)
        self.discriminator_loss = self.get_disc_loss_fn()
        self.segmentator_loss = self.get_seg_loss_fn(w_seg=1.0, alpha=0.25, gamma=2.0)
    

    @tf.function(autograph=True, reduce_retracing=True)
    def augment_inputs(self, inputs):
        image, roi = inputs

        # Use tf.map_fn to apply augmentations to each element in the batch
        augmented_images, augmented_rois = tf.map_fn(
            lambda x: self.aug_pipe.apply(image=x[0], mask=x[1]),
            (image, roi),  # Provide the image and roi pairs
            parallel_iterations=12,
            fn_output_signature=(tf.TensorSpec(shape=None, dtype=image.dtype),
                                tf.TensorSpec(shape=None, dtype=roi.dtype))  # Specify output types and shapes
        )
        augmented_rois = tf.where(augmented_rois > 0.5, 1.0, 0.0)
        del image, roi

        return (tf.stop_gradient(augmented_images), tf.stop_gradient(augmented_rois))


    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True)
    def train_step(self, inputs):
        xr, xn, n, mr, r = inputs

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape, tf.GradientTape() as segmentator_tape:
            # Get generator output for training
            zr, xf, zf = self.generator_model(xn, training=True)

            # Get discriminator output for training
            fr, yr = self.discriminator_model(xr, training=True)
            ff, yf = self.discriminator_model(xf, training=True)

            # Get segmentator output for training
            mf = self.unet_model((xr, tf.stop_gradient(xf)), training=True)

            contextual_loss = self.contextual_loss(xr, xf)
            adversarial_loss = self.adversarial_loss(fr, ff)
            latent_loss = self.latent_loss(zr, zf)
            generator_loss = tf.math.add_n([adversarial_loss, contextual_loss, latent_loss])

            discriminator_loss = self.discriminator_loss(yr, yf)

            segmentator_loss = self.segmentator_loss(mr, mf, r)

        generator_grads = generator_tape.gradient(generator_loss, self.discriminator_model.trainable_weights)
        discriminator_grads = discriminator_tape.gradient(discriminator_loss, self.generator_model.trainable_weights)
        segmentator_grads = segmentator_tape.gradient(segmentator_loss, self.generator_model.trainable_weights)

        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator_model.trainable_weights))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator_model.trainable_weights))
        self.segmentator_optimizer.apply_gradients(zip(segmentator_grads, self.segmentator_model.trainable_weights))

        return {'Xf': xf,
                'Mf': mf,
                'Zr': zr,
                'Zf': zf,
                'contextual_loss': contextual_loss,
                'adversarial_loss': adversarial_loss,
                'latent_loss': latent_loss,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss,
                'segmentator_loss': segmentator_loss}


    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True)
    def test_step(self, inputs):
        xr, xn, n, mr, r = inputs

        # Get generator output for validation
        zr, xf, zf = self.generator_model(xn, training=False)

        # Get discriminator output for validation
        fr, yr = self.discriminator_model(xr, training=False)
        ff, yf = self.discriminator_model(xf, training=False)

        # Get segmentator output for validation
        mf = self.unet_model((xr, tf.stop_gradient(xf)), training=False)

        contextual_loss = self.contextual_loss(xr, xf)
        adversarial_loss = self.adversarial_loss(fr, ff)
        latent_loss = self.latent_loss(zr, zf)
        generator_loss = tf.math.add_n([adversarial_loss, contextual_loss, latent_loss])

        discriminator_loss = self.discriminator_loss(yr, yf)

        segmentator_loss = self.segmentator_loss(mr, mf, r)

        return {'Xf': xf,
                'Mf': mf,
                'Zr': zr,
                'Zf': zf,
                'contextual_loss': contextual_loss,
                'adversarial_loss': adversarial_loss,
                'latent_loss': latent_loss,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss,
                'segmentator_loss': segmentator_loss}
    

    def train_loop(self):
        with tqdm(iterable=self.ds_training_path, leave=True, desc='Train', unit='batch') as pbar:
            for idx, inputs in enumerate(pbar):
                image, roi = self.augment_inputs(inputs)
                xa, xn, n, m = self.perlin.perlin_noise_batch(image)
                self.log_inputs((image, roi))
                self.log_inputs((xn, m))
                self.log_inputs((xa, n))


    def train(self):
        for epoch in range(self.epochs):
            logger.info('Epoch %d / %d', epoch + 1, self.epochs)

            self.perlin.pre_generate_noise(epoch=epoch, min_area=20)

            self.train_loop()
    
    
    def log_inputs(self, inputs):
        images, masks = inputs
        
        # Batch size (number of images in the batch)
        batch_size = images.shape[0]
        
        # Create a figure to plot images and masks side by side
        fig, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 3))
        
        for i in range(batch_size):
            # Get the image and mask
            img = images[i]
            mask = masks[i]
            
            # Apply augmentation
            augmented_img, augmented_mask = (img, mask)
            
            # Display the augmented image in column 0
            axes[i, 0].imshow(augmented_img.numpy())
            axes[i, 0].axis('off')
            
            # Display the corresponding augmented mask in column 1
            axes[i, 1].imshow(tf.image.grayscale_to_rgb(augmented_mask).numpy())
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        plt.cla()
        plt.clf()
        plt.close(fig)
        plt.close()
        plt.close('all')


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


    def get_seg_loss_fn(self, w_seg: float = 1.0, alpha: float = 0.25, gamma: float = 2.0,):
        return lambda mr, mf, roi : tf.math.multiply_no_nan(focal_loss(y_true=tf.math.multiply_no_nan(mr, roi), y_pred=mf, alpha=alpha, gamma=gamma, from_logits=False, reduction='mean'), w_seg)


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
