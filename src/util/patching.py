"""
This module provides utility functions for extracting patches from an image tensor
and reconstructing the image from those patches using TensorFlow. The main functions 
included are:

1. unfold: Extracts square patches from an image.
2. fold: Reconstructs an image from extracted patches.
3. tf_unfold: Extracts patches from an image tensor in NHWC format with customizable kernel sizes and strides.
4. tf_fold: Reconstructs an output tensor from patches with handling for padding and stride.
5. recontruct_from_patches_grp: Reconstructs an image from patches using a gradient-based method.

These functions are useful for tasks such as image processing, data augmentation, 
and convolutional neural network operations that require patch-wise manipulation of images.
"""

from typing import Tuple, Union
import tensorflow as tf

def unfold(
        image: tf.Tensor,
        patch_size: int,
        strides: int
    ) -> tf.Tensor:
    """
    Extracts patches from an image tensor.

    Args:
    image (tf.Tensor): Input tensor of shape [batch_size, height, width, channels].
    patch_size (int): Size of each patch (patch_height, patch_width).
    strides (int): Stride of the sliding window for extracting patches.

    Returns:
    tf.Tensor: A tensor of shape [batch_size, num_patches, patch_height, patch_width, channels],
               where num_patches is the number of patches extracted from the image.
    """
    # Extract patches using TensorFlow's extract_patches function.
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, strides, strides, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'  # Only extract patches that fit fully within the image.
    )
    
    # Get the batch size from the patches tensor.
    batch_size = tf.shape(patches)[0]
    
    # Reshape the patches tensor to have the shape [batch_size, num_patches, patch_height, patch_width, channels].
    patches = tf.reshape(patches, [batch_size, -1, patch_size, patch_size, image.shape[-1]])
    
    return patches

def fold(
        patches: tf.Tensor,
        image_shape: tf.TensorShape,
        patch_size: int,
        strides: int
    ) -> tf.Tensor:
    """
    Reconstructs the image from patches by placing them at the correct positions.

    Args:
    patches (tf.Tensor): Tensor of extracted patches of shape [batch_size, num_patches, patch_height, patch_width, channels].
    image_shape (tf.TensorShape): Shape of the original image as [batch_size, height, width, channels].
    patch_size (int): Size of each patch (patch_height, patch_width).
    strides (int): Stride of the sliding window used during patch extraction.

    Returns:
    tf.Tensor: The reconstructed image tensor of shape [batch_size, height, width, channels].
    """
    batch_size = image_shape[0]
    height = image_shape[1]
    width = image_shape[2]
    channels = image_shape[3]

    # Initialize an empty tensor to hold the reconstructed image.
    reconstructed_image = tf.zeros(image_shape, dtype=patches.dtype)
    
    # Initialize a tensor to keep track of how many patches overlap at each pixel.
    overlap_count = tf.zeros(image_shape, dtype=patches.dtype)

    # Calculate the number of patches along the height and width of the image.
    num_patches_h = (height - patch_size) // strides + 1
    num_patches_w = (width - patch_size) // strides + 1

    # Initialize a patch index counter.
    patch_idx = 0
    
    # Iterate over the grid of patches.
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extract the current patch.
            patch = patches[:, patch_idx, :, :, :]
            patch = tf.reshape(patch, [batch_size, patch_size, patch_size, channels])

            # Calculate the top-left corner position where the patch should be placed.
            h_start = i * strides
            w_start = j * strides

            # Generate the indices for placing the patch in the reconstructed image.
            batch_indices = tf.range(batch_size)
            h_indices = tf.range(h_start, h_start + patch_size)
            w_indices = tf.range(w_start, w_start + patch_size)
            c_indices = tf.range(channels)

            # Create a meshgrid for the indices.
            b, h, w, c = tf.meshgrid(batch_indices, h_indices, w_indices, c_indices, indexing='ij')
            indices = tf.stack([b, h, w, c], axis=-1)

            # Reshape indices and updates to match the patch shape.
            indices = tf.reshape(indices, [-1, 4])
            updates = tf.reshape(patch, [-1])

            # Add the patch values to the reconstructed image.
            reconstructed_image = tf.tensor_scatter_nd_add(reconstructed_image, indices, updates)

            # Increment the overlap count for normalization.
            overlap_patch = tf.ones_like(patch)
            overlap_updates = tf.reshape(overlap_patch, [-1])
            overlap_count = tf.tensor_scatter_nd_add(overlap_count, indices, overlap_updates)

            # Move to the next patch index.
            patch_idx += 1

    # Normalize the reconstructed image to account for overlapping patches.
    reconstructed_image = tf.math.divide_no_nan(reconstructed_image, overlap_count)

    return reconstructed_image

def tf_unfold(
        input_tensor: tf.Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int,
        padding: str = 'VALID'
    ) -> tf.Tensor:
    """
    Extracts patches from an input tensor in NHWC format.

    Args:
    input_tensor (tf.Tensor): Input tensor in NHWC format [batch_size, height, width, channels].
    kernel_size (Union[int, Tuple[int, int]]): Size of the extraction kernel as an integer or a tuple (height, width).
    stride (int): Stride of the sliding window for extracting patches.
    padding (str): Padding mode, either 'VALID' or 'SAME'.

    Returns:
    tf.Tensor: A tensor of patches of shape [batch_size, num_patches, patch_height, patch_width, channels].
    """
    # Determine the dimensions of the input tensor and kernel.
    batch_size, height, width, in_channels = input_tensor.shape
    kernel_height, kernel_width = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)

    # Extract patches using TensorFlow's extract_patches function.
    patches = tf.image.extract_patches(
        images=input_tensor,
        sizes=[1, kernel_height, kernel_width, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding=padding.upper()  # Use TensorFlow's padding format.
    )

    # Reshape patches to the desired output shape [batch_size, num_patches, patch_height, patch_width, channels].
    patches = tf.reshape(patches, [batch_size, -1, kernel_height, kernel_width, in_channels])

    return patches

def tf_fold(
        patches: tf.Tensor,
        output_size: Tuple[int, int],
        kernel_size: Union[int, Tuple[int, int]],
        stride: int,
        padding: str = 'SAME'
    ) -> tf.Tensor:
    """
    Reconstructs an output tensor from patches by placing them at the correct positions.

    Args:
    patches (tf.Tensor): Tensor of extracted patches of shape [batch_size, num_patches, patch_height, patch_width, channels].
    output_size (Tuple[int, int]): The size of the output tensor as (height, width).
    kernel_size (Union[int, Tuple[int, int]]): Size of the kernel as an integer or a tuple (kernel_height, kernel_width).
    stride (int): Stride of the sliding window used during patch extraction.
    padding (str): Padding mode, either 'VALID' or 'SAME'.

    Returns:
    tf.Tensor: The reconstructed tensor of shape [batch_size, output_height, output_width, channels].
    """
    batch_size, num_patches, patch_height, patch_width, in_channels = patches.shape
    output_height, output_width = output_size
    kernel_height, kernel_width = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    
    # Calculate the total padding required if using 'SAME' padding.
    if padding == 'SAME':
        pad_total_height = int(max((tf.math.ceil(output_height / stride) - 1) * stride + kernel_height - output_height, 0))
        pad_total_width = int(max((tf.math.ceil(output_width / stride) - 1) * stride + kernel_width - output_width, 0))

        pad_top = pad_total_height // 2
        pad_bottom = pad_total_height - pad_top
        pad_left = pad_total_width // 2
        pad_right = pad_total_width - pad_left
    else:
        pad_top = pad_left = pad_bottom = pad_right = pad_total_height = pad_total_width = 0

    # Initialize the output tensor and a count tensor to keep track of overlapping regions.
    output_tensor = tf.zeros([batch_size, output_height + pad_total_height, output_width + pad_total_width, in_channels], dtype=patches.dtype)
    count_tensor = tf.zeros([batch_size, output_height + pad_total_height, output_width + pad_total_width, in_channels], dtype=patches.dtype)

    # Initialize the patch index counter.
    idx = 0
    
    # Iterate over the grid of possible patch positions.
    for i in range(0, output_height - kernel_height + 1 + pad_top + pad_bottom, stride):
        for j in range(0, output_width - kernel_width + 1 + pad_left + pad_right, stride):
            # Extract the current patch.
            patch = patches[:, idx, :, :, :]

            # Calculate the indices for placing the patch in the output tensor.
            batch_indices = tf.range(batch_size)[:, None, None, None]
            height_indices = tf.range(i, i + patch_height)[None, :, None, None]
            width_indices = tf.range(j, j + patch_width)[None, None, :, None]
            channel_indices = tf.range(in_channels)[None, None, None, :]

            # Create a meshgrid for the indices.
            indices = tf.stack(tf.meshgrid(batch_indices, height_indices, width_indices, channel_indices, indexing='ij'), axis=-1)
            indices = tf.reshape(indices, [-1, 4])

            # Flatten the patch to match the flattened indices.
            flat_patch = tf.reshape(patch, [-1])

            # Scatter add the patch values into the output tensor.
            output_tensor = tf.tensor_scatter_nd_add(output_tensor, indices, flat_patch)
            count_tensor = tf.tensor_scatter_nd_add(count_tensor, indices, tf.ones_like(flat_patch))

            # Move to the next patch index.
            idx += 1

    # Normalize the output tensor by the count tensor to handle overlapping areas.
    output_tensor = tf.math.divide_no_nan(output_tensor, count_tensor)
    
    # Remove any padding from the output tensor.
    output_tensor = output_tensor[:, pad_top:pad_top + output_height, pad_left:pad_left + output_width, :]

    return output_tensor

@tf.function(autograph=True, reduce_retracing=True)
def recontruct_from_patches_grp(
        ref_X: tf.Tensor,
        patched_X: tf.Tensor,
        patch_size: int,
        strides: Tuple[int, int],
        patch_alg: str = 'VALID'
    ) -> tf.Tensor:
    """
    Reconstructs an image from patches using a gradient-based method.

    This function calculates the gradient of the patched tensor with respect to 
    a reference tensor and then uses this gradient to reconstruct the original 
    image from the patches.

    Args:
    ref_X (tf.Tensor): The reference tensor of shape [batch_size, height, width, channels].
    patched_X (tf.Tensor): The tensor containing patches to be reconstructed.
    patch_size (int): The size of the patches to be reconstructed.
    strides (Tuple[int, int]): The strides used in both height and width dimensions.
    patch_alg (str): The patch extraction algorithm to use, either 'VALID' or 'SAME'.

    Returns:
    tf.Tensor: The reconstructed image tensor of the same shape as ref_X.
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(ref_X)  # Watch the reference tensor for gradients.
        patches = patched_X  # Patches to be reconstructed.
        _x = tf.zeros_like(ref_X)  # A zero tensor of the same shape as the reference tensor.
        _y = tf.image.extract_patches(images=_x, sizes=[1, patch_size, patch_size, 1], strides=[1, strides[0], strides[1], 1], rates=[1, 1, 1, 1], padding=patch_alg.upper())
    
    # Compute the gradient of _y with respect to _x.
    grad = tape.gradient(_y, _x)
    
    # Calculate the inverse gradient to reconstruct the image.
    inv = tape.gradient(_y, _x, output_gradients=patches) / grad
    
    # Clean up the tape to free memory.
    del tape, grad
    
    return inv