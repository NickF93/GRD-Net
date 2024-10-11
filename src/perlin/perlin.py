import math
import random

from typing import Tuple, Union

import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
import imgaug.augmenters as iaa

from ..augment import AugmentPipe

class Perlin:
    def __init__(self,
                 size,
                 target_size,
                 reference_dataset: tf.data.Dataset = None,
                 real_defect_dataset: tf.data.Dataset = None,
                 fraction=0.75,
                 choice=0.25,
                 def_choice=0.25,
                 use_perlin_noise=True,
                 use_gaussian_noise=True,
                 generate_perlin_each_time=False,
                 perlin_queue_max:int = 1000,
                 perlin_queue_min: int=500,
                 perlin_generation_every_n_epochs:int = 1,
                 perlin_generate_m_perturbations: int = 10
                 ):
        self.size       = size
        self.target_size = target_size

        self.reference_dataset = reference_dataset
        self.real_defect_dataset = real_defect_dataset

        self.fraction = fraction
        self.choice     = choice
        self.def_choice = def_choice

        if self.reference_dataset is not None:
            self.r_iter = iter(self.reference_dataset)
        else:
            self.r_iter = None
        
        if self.real_defect_dataset is not None:
            self.d_iter = iter(self.real_defect_dataset)
        else:
            self.d_iter = None

        self.pipe = AugmentPipe(random_90_rotation=3,
                                rotation_angle=np.pi,
                                flip_mode='both',
                                translation_range=round(max(target_size) / 200),
                                zoom_range=0.1)

        self.use_perlin_noise = use_perlin_noise
        self.use_gaussian_noise = use_gaussian_noise
        self.generate_perlin_each_time        = generate_perlin_each_time
        self.perlin_queue_max                 = perlin_queue_max
        self.perlin_queue_min                 = perlin_queue_min
        self.perlin_generation_every_n_epochs = perlin_generation_every_n_epochs
        self.perlin_generate_m_perturbations  = perlin_generate_m_perturbations

        self.perlin_noise_array               = []
        self.perlin_mask_array                = []

    def get_image(self, regularize=True):
        try:
            batch = next(self.r_iter)
        except StopIteration:
            self.r_iter = iter(self.reference_dataset)
            batch = next(self.r_iter)

        x, *_ = batch
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[random.randint(0, len(x) - 1)]
        elif isinstance(x, tf.Tensor):
            if len(x.shape) == 4:
                x = x[random.randint(0, x.shape[0] - 1)]

            
        x = tf.image.rot90                  (x, random.randint(-3, 3))
        x = tf.image.random_flip_left_right (x)
        x = tf.image.random_flip_up_down    (x)
        x = tf.image.random_hue             (x, 0.5)
        x = tf.image.random_saturation      (x, 0.5, 1.5)
        x = tf.image.random_brightness      (x, 0.2)

        # Regularization
        if regularize:
            xmax = tf.math.reduce_max           (x)
            xmin = tf.math.reduce_min           (x)
            a = tf.math.maximum                 (xmax, 1.0)
            b = tf.math.minimum                 (xmin, 0.0)
            x = tf.math.divide_no_nan           (tf.math.subtract(x, b), tf.math.subtract(a, b))
        x = tf.clip_by_value                (x, clip_value_min=0.0, clip_value_max=1.0)

        x = tf.image.central_crop(x, central_fraction=self.fraction)

        x = tf.image.random_crop            (x, size=(self.size, self.size, x.shape[-1]))
        return x

    def lerp_np(self, x, y, w):
        """Helper function."""
        fin_out = (y - x) * w + x
        return fin_out

    def _rand_perlin_2d_np(self, shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
        """Generate a random image containing Perlin noise. Numpy version."""
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        def tile_grads(slice1, slice2):
            g = gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
            a = np.repeat(g, d[0], axis=0)
            b = np.repeat(a, d[1], axis=1)
            return b

        def dot(grad, shift):
            return (
                    np.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), axis=-1)
                    * grad[: shape[0], : shape[1]]
            ).sum(axis=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[: shape[0], : shape[1]])
        return math.sqrt(2) * self.lerp_np(self.lerp_np(n00, n10, t[..., 0]), self.lerp_np(n01, n11, t[..., 0]), t[..., 1])

    def random_2d_perlin(self, shape: Tuple, res: Tuple[Union[int, tf.Tensor], Union[int, tf.Tensor]], fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3) -> Union[np.ndarray, tf.Tensor]:
        """Returns a random 2d perlin noise array.
        Args:
                shape (Tuple): Shape of the 2d map.
                res (Tuple[Union[int, Tensor]]): Tuple of scales for perlin noise for height and width dimension.
                fade (_type_, optional): Function used for fading the resulting 2d map.
                        Defaults to equation 6*t**5-15*t**4+10*t**3.
        Returns:
                Union[np.ndarray, Tensor]: Random 2d-array/tensor generated using perlin noise.
        """
        if isinstance(res[0], int):
            result = self._rand_perlin_2d_np(shape, res, fade)
        else:
            raise TypeError(f"got scales of type {type(res[0])}")
        return result

    def perlin_perturbation(self):
        size = int(self.size)
        if size & (size - 1) == 0:
            new_size = int(size)
        else:
            new_size = int(2 ** int(math.ceil(math.log(size) / math.log(2))))

        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]
        rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        def rand_augmenter() -> iaa.Sequential:
                """Selects 3 random transforms that will be applied to the anomaly source images.
                Returns:
                        A selection of 3 transforms.
                """
                aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
                aug = iaa.Sequential([augmenters[aug_ind[0]], augmenters[aug_ind[1]], augmenters[aug_ind[2]]])
                return aug

        perlin_scale = 6
        min_perlin_scale = 0

        perlin_scalex = 2 ** random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** random.randint(min_perlin_scale, perlin_scale)

        perlin_noise = self.random_2d_perlin((new_size, new_size), (perlin_scalex, perlin_scaley))
        perlin_noise = rot(image=perlin_noise)

        # Create mask from perlin noise
        mask = np.where(perlin_noise > 0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        mask = np.expand_dims(mask, axis=2).astype(np.float32)

        anomaly_source_img = np.expand_dims(perlin_noise, 2).repeat(3, 2)
        anomaly_source_img = (anomaly_source_img * 255).astype(np.uint8)

        # Augment anomaly source image
        aug = rand_augmenter()
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Create anomalous perturbation that we will apply to the image
        perturbation = anomaly_img_augmented.astype(np.float32) * mask / 255.0

        perturbation = tf.image.resize(tf.convert_to_tensor(perturbation), self.target_size, 'bilinear', antialias=True)
        mask = tf.image.resize(tf.convert_to_tensor(mask), self.target_size, 'nearest', antialias=True)

        if self.reference_dataset is not None and random.random() > self.choice:
            x = self.get_image()
            perturbation = tf.math.multiply_no_nan(x, mask)
        elif self.real_defect_dataset is not None and random.random() > self.def_choice:
            try:
                perturbation, mask = next(self.d_iter)
            except StopIteration:
                self.d_iter = iter(self.real_defect_dataset)
                perturbation, mask = next(self.d_iter)
                
            if (len(perturbation.shape) == 4):
                r = random.randint(0, perturbation.shape[0] - 1)
                perturbation = perturbation[r]
                mask = mask[r]
                perturbation = tf.math.multiply_no_nan(perturbation, mask)
                self.pipe(perturbation, mask)
        else:
            pass

        return perturbation, mask

    #region [perlin_noise_batch]
    def generate_perlin_perturbation_greater_than(self, area_min):
        """
        @brief Generate Perlin noise perturbation greater than a specified area.
        
        @param area_min Minimum area for the Perlin noise perturbation.
        
        @return Tuple of noise and noise mask.
        """
        # Check if the minimum area is greater than 0
        if area_min > 0:
            # Initialize noise and noise mask
            noise = None
            noise_mask = None
            redm = 0
            # The noise area should be at least `area_min` pxl
            redo_count = 3
            # Loop until noise and noise mask are not None and redm is less than area_min
            while (((noise is None) or (noise_mask is None)) or redm < area_min):
                # Generate perlin perturbation
                noise, noise_mask = self.perlin_perturbation()
                # Calculate the sum of the noise mask
                redm = tf.math.reduce_sum(noise_mask)
                # Increment the redo count
                redo_count += 1
                # If redo count is greater than 3, reset the seed
                if redo_count > 3:
                    self.perlin.set_seed()
                    print('Reset seed...')
            # Delete redm and redo_count to free up memory
            del redm, redo_count
        else:
            # Generate perlin perturbation
            noise, noise_mask = self.perlin_perturbation()

        return noise, noise_mask

    def generate_perlin_noise(self, area_min=25):
        """
        @brief Generate Perlin noise.
        
        @param area_min Minimum area for the Perlin noise.
        
        @return Tuple of noise and noise mask.
        """
        # Check if perlin noise should be generated each time
        if self.generate_perlin_each_time:
            # Generate perlin perturbation greater than area_min
            noise, noise_mask = self.generate_perlin_perturbation_greater_than(area_min=area_min)
        else:
            # Get the length of the perlin noise array
            l = len(self.perlin_noise_array)
            # Check if the lengths of the perlin noise array and the perlin mask array match
            assert l == len(self.perlin_mask_array), "Perlin noise arrays size mismatch"
            # Generate a random integer between 0 and l - 1
            p = random.randint(0, (l - 1))
            # Get the noise and noise mask from the perlin noise array and the perlin mask array
            noise       = self.perlin_noise_array[p]
            noise_mask  = self.perlin_mask_array[p]

        return noise, noise_mask

    def perlin_noise_tensors(self, n, tsize=224, channels=3, p=1.0):
        """
        @brief Generate Perlin noise tensors.
        
        @param n Number of tensors to generate.
        @param tsize Size of the tensor.
        @param channels Number of channels in the tensor.
        @param p Probability of generating Perlin noise.
        
        @return Tuple of noise mask and noise.
        """
        # Initialize noise mask and noise tensors
        N : tf.Tensor = None
        M : tf.Tensor = None
        # Loop for n times
        for _ in range(n):
            # If random number is less than or equal to p, generate perlin noise
            if random.random() <= p:
                noise, noise_mask = self.generate_perlin_noise(area_min=100)
            else:
                # Otherwise, generate zero tensors
                noise, noise_mask = (tf.zeros((tsize, tsize, channels)), tf.zeros((tsize, tsize, 1)))
            
            # If channels is 1, convert noise to grayscale
            if self.nc == 1:
                noise = tf.image.rgb_to_grayscale(noise)
                noise_mask_tmp = noise_mask
            else:
                # Otherwise, convert noise mask to rgb
                noise_mask_tmp = tf.image.grayscale_to_rgb(noise_mask)

            # If M is None, initialize M and N
            if M is None:
                M = noise_mask_tmp[tf.newaxis, ...]
                N = noise[tf.newaxis, ...]
            else:
                # Otherwise, concatenate new noise and noise mask to N and M
                N =   tf.concat([N  , noise[tf.newaxis, ...]]     , axis=0)
                M =   tf.concat([M  , noise_mask_tmp[tf.newaxis, ...]]  , axis=0)

            # Delete noise, noise mask, and noise mask tmp to free up memory
            del noise, noise_mask, noise_mask_tmp

        # Convert M to grayscale
        M = tf.image.rgb_to_grayscale(M)
        
        return M, N

    def perlin_noise_batch(self, X, channels=3, p=1.0, area_min=50):
        """
        @brief Generate a batch of Perlin noise.
        
        @param X Input tensor.
        @param p Probability of generating Perlin noise.
        
        @return Tuple of original tensor, noisy tensor, noise, and noise mask.
        """
        # Initialize noisy tensor, noise, and noise mask
        Xn : tf.Tensor = None
        N : tf.Tensor = None
        M : tf.Tensor = None
        # Loop for the length of X
        for _ in range(len(X)):
            # If random number is less than p, generate perlin noise
            if random.random() < p:
                noise, noise_mask = self.generate_perlin_noise(area_min=area_min)
            else:
                # Otherwise, generate zero tensors
                _s = list()
                _s.extend(X.shape[1:-1])
                _s.append(1)
                noise, noise_mask = (tf.zeros(tuple(X.shape[1:])), tf.zeros(tuple(_s)))
            
            # If channels is 1, convert noise to grayscale
            if channels == 1:
                noise = tf.image.rgb_to_grayscale(noise)
                noise_mask_tmp = noise_mask
            else:
                # Otherwise, convert noise mask to rgb
                noise_mask_tmp = tf.image.grayscale_to_rgb(noise_mask)

            # If M is None, initialize M and N
            if M is None:
                M = noise_mask_tmp[tf.newaxis, ...]
                N = noise[tf.newaxis, ...]
            else:
                # Otherwise, concatenate new noise and noise mask to N and M
                N =   tf.concat([N  , noise[tf.newaxis, ...]]     , axis=0)
                M =   tf.concat([M  , noise_mask_tmp[tf.newaxis, ...]]  , axis=0)

            # Delete noise, noise mask, and noise mask tmp to free up memory
            del noise, noise_mask, noise_mask_tmp
        
        # Calculate X_thr, Xa, and Xn
        X_thr = tf.math.multiply_no_nan(X, M)
        Xa    = tf.math.multiply_no_nan(X, (1 - M)) + tf.math.multiply_no_nan((1 - self.beta), X_thr) + tf.math.multiply_no_nan(tf.math.multiply_no_nan(self.beta, X), (M))
        Xn    = tf.math.multiply_no_nan(M, Xa) + tf.math.multiply_no_nan((1. - M), X)
        Xn    = tf.clip_by_value((Xn - M), 0., 1.)
        Xn    = (Xn + N)

        # Convert M to grayscale
        M = tf.image.rgb_to_grayscale(M)

        # Replace NaN values with 0 in X, Xn, N, and M
        X  = tf.where(tf.math.is_nan(X),  0., X )
        Xn = tf.where(tf.math.is_nan(Xn), 0., Xn)
        N  = tf.where(tf.math.is_nan(N),  0., N )
        M  = tf.where(tf.math.is_nan(M),  0., M )
        
        # Clip the values of X, Xn, N, and M between 0 and 1
        X  = tf.clip_by_value(X,  clip_value_min=0., clip_value_max=1.)
        Xn = tf.clip_by_value(Xn, clip_value_min=0., clip_value_max=1.)
        N  = tf.clip_by_value(N,  clip_value_min=0., clip_value_max=1.)
        M  = tf.clip_by_value(M,  clip_value_min=0., clip_value_max=1.)
        return X, Xn, N, M

    def pre_generate_noise(self, epoch):
        # region [PERLIN NOISE GENERATION]
        if self.use_perlin_noise and not self.generate_perlin_each_time:
            # Check if Perlin noise should be generated
            if ((epoch == 0) or ((epoch % self.perlin_generation_every_n_epochs) == 0)):
                # Check if it's the first epoch
                if epoch == 0:
                    print('Generate Perlin perturbations for first time. This could take a while...')
                    # Initialize progress bar
                    with tqdm(total=self.perlin_queue_min, leave=True) as pbar:
                        # Loop for perlin_queue_min times
                        for i in range(self.perlin_queue_min):
                            # Generate perlin perturbation
                            noise, mask = self.perlin_perturbation()
                            # Append noise and mask to perlin noise array and perlin mask array
                            self.perlin_noise_array.append(noise)
                            self.perlin_mask_array.append(mask)
                            # Delete noise and mask to free up memory
                            del noise, mask
                            # Update progress bar
                            pbar.update(1)
                else:
                    print('Generate Perlin perturbations...')
                    # Initialize progress bar
                    with tqdm(total=self.perlin_generate_m_perturbations, leave=True) as pbar:
                        # Loop for perlin_generate_m_perturbations times
                        for i in range(self.perlin_generate_m_perturbations):
                            # Generate perlin perturbation
                            noise, mask = self.perlin.perlin_perturbation()
                            # Append noise and mask to perlin noise array and perlin mask array
                            self.perlin_noise_array.append(noise)
                            self.perlin_mask_array.append(mask)
                            # Delete noise and mask to free up memory
                            del noise, mask
                            # Update progress bar
                            pbar.update(1)
                    
                    # Check if the length of the perlin noise array is greater than perlin_queue_max
                    while len(self.perlin_noise_array) > self.perlin_queue_max:
                        # Remove the first element from the perlin noise array and the perlin mask array
                        self.perlin_noise_array.pop(0)
                        self.perlin_mask_array.pop(0)
        # endregion [PERLIN NOISE GENERATION]

    #endregion [perlin_noise_batch]

