from typing import Union, Optional, Tuple
import tensorflow as tf


def _apply_reduction(
    loss: tf.Tensor, 
    reduction: str = 'mean', 
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Private function to apply reduction to the computed loss values.

    Parameters:
    - loss (tf.Tensor): The computed loss values (element-wise).
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - tf.Tensor: The reduced loss value, according to the 'reduction' parameter.
    """
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return tf.reduce_sum(loss, axis=axis)
    else:  # 'mean'
        return tf.reduce_mean(loss, axis=axis)


def _binary_crossentropy(y_true, y_pred, from_logits=False):
    epsilon = tf.keras.backend.epsilon()
    
    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)  # Apply sigmoid if working with logits
    
    # Clip the predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Compute the element-wise binary cross-entropy
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    
    return bce  # No reduction is applied here


@tf.function(reduce_retracing=True)
def huber_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    delta: float = 1.0,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Compute the Huber loss between true and predicted values.

    Parameters:
    - y_true (tf.Tensor): Ground truth values.
    - y_pred (tf.Tensor): Predicted values.
    - delta (float): Threshold at which to switch between squared and linear loss. Default is 1.0.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - tf.Tensor: The Huber loss value, optionally reduced according to the 'reduction' parameter.
    """
    # Calculate the error between true and predicted values
    error = y_true - y_pred
    abs_error = tf.abs(error)
    
    # Quadratic loss for small errors, linear for large errors
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear

    # Apply reduction to the final loss
    return _apply_reduction(loss, reduction=reduction, axis=axis)


@tf.function(reduce_retracing=True)
def mse_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    - y_true (tf.Tensor): Ground truth values.
    - y_pred (tf.Tensor): Predicted values.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - tf.Tensor: The MSE loss value, optionally reduced according to the 'reduction' parameter.
    """
    # Calculate the squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Apply reduction to the squared error
    return _apply_reduction(squared_error, reduction=reduction, axis=axis)


@tf.function(reduce_retracing=True)
def mae_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Compute the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    - y_true (tf.Tensor): Ground truth values.
    - y_pred (tf.Tensor): Predicted values.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - tf.Tensor: The MAE loss value, optionally reduced according to the 'reduction' parameter.
    """
    # Calculate the absolute error
    abs_error = tf.abs(y_true - y_pred)
    
    # Apply reduction to the absolute error
    return _apply_reduction(abs_error, reduction=reduction, axis=axis)


@tf.function(reduce_retracing=True)
def ssim_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    max_val: float = 1.0,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Compute the SSIM loss between the true and predicted images, with support for batch processing
    and flexible reduction methods.

    Parameters:
    - y_true (tf.Tensor): Ground truth images tensor. Expected shape is [batch_size, ...].
    - y_pred (tf.Tensor): Predicted images tensor. Expected shape is [batch_size, ...].
    - max_val (float): Maximum possible value for input images (e.g., 1.0 for normalized images).
    - reduction (str): Specifies the reduction to apply to the SSIM loss: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - tf.Tensor: The computed SSIM loss, reduced according to the specified reduction type.

    Raises:
    - ValueError: If an invalid reduction type is provided.
    - tf.errors.InvalidArgumentError: If the shapes of y_true or y_pred don't match.

    Example:
        model.compile(optimizer='adam', loss=lambda y_true, y_pred: ssim_loss(y_true, y_pred, reduction='mean'))

    Notes:
        - The SSIM value is computed using `tf.image.ssim`, which measures the similarity between two images.
        - If `reduction='none'`, the SSIM result is expanded by dividing by the product of the target shape (excluding batch size).
    """
    # Ensure that y_true and y_pred have the same shape
    tf.debugging.assert_equal(
        tf.shape(y_true), tf.shape(y_pred),
        message="y_true and y_pred must have the same shape."
    )

    # Infer target shape (excluding the batch axis)
    target_shape = tf.shape(y_true)[1:]
    target_size = tf.reduce_prod(target_shape)

    # Compute SSIM for each image in the batch
    ssim_value: tf.Tensor = tf.image.ssim(y_true, y_pred, max_val=max_val)

    # Convert SSIM to a loss value (1 - SSIM)
    loss: tf.Tensor = 1 - ssim_value

    # If reduction is 'none', expand the SSIM result and divide by the product of the target shape
    if reduction == 'none':
        loss = tf.expand_dims(loss, axis=-1) / tf.cast(target_size, loss.dtype)
    
    return _apply_reduction(loss, reduction=reduction, axis=axis)


@tf.function(reduce_retracing=True)
def bce_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    from_logits: bool = False,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Compute the Binary Cross-Entropy (BCE) loss between true and predicted values.

    Parameters:
    - y_true (tf.Tensor): Ground truth binary labels (0 or 1).
    - y_pred (tf.Tensor): Predicted values (probabilities).
    - from_logits (bool): If True, interpret y_pred as logits (pre-sigmoid activations). Default is False.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - tf.Tensor: The BCE loss value, optionally reduced according to the 'reduction' parameter.
    """
    # Compute BCE loss, optionally from logits
    loss: tf.Tensor = _binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # Apply reduction to the BCE loss
    return _apply_reduction(loss, reduction=reduction, axis=axis)


@tf.function(reduce_retracing=True)
def focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    bce = _binary_crossentropy(
        y_true=y_true,
        y_pred=y_pred,
        from_logits=False,
    )

    # Calculate focal factor
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_factor = tf.pow(1.0 - p_t, gamma)

    focal_bce = focal_factor * bce

    weight = y_true * alpha + (1 - y_true) * (1 - alpha)
    loss: tf.Tensor = weight * focal_bce

    # Apply the private reduction function
    return _apply_reduction(loss, reduction=reduction, axis=axis)
