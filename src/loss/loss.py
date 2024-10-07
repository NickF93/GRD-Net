from typing import Union, Optional
import tensorflow as tf

def _apply_reduction(
    loss: tf.Tensor, 
    reduction: str = 'mean', 
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Private function to apply reduction to the computed loss values.

    Parameters:
    - loss (Tensor): The computed loss values (element-wise).
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - Tensor: The reduced loss value, according to the 'reduction' parameter.
    """
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return tf.reduce_sum(loss, axis=axis)
    else:  # 'mean'
        return tf.reduce_mean(loss, axis=axis)

@tf.function
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
    - y_true (Tensor): Ground truth values.
    - y_pred (Tensor): Predicted values.
    - delta (float): Threshold at which to switch between squared and linear loss. Default is 1.0.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - Tensor: The Huber loss value, optionally reduced according to the 'reduction' parameter.
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear

    return _apply_reduction(loss, reduction=reduction, axis=axis)

@tf.function
def mse_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.
    
    Parameters:
    - y_true (Tensor): Ground truth values.
    - y_pred (Tensor): Predicted values.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - Tensor: The MSE loss value, optionally reduced according to the 'reduction' parameter.
    """
    squared_error = tf.square(y_true - y_pred)
    
    return _apply_reduction(squared_error, reduction=reduction, axis=axis)

@tf.function
def mae_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    reduction: str = 'mean',
    axis: Optional[Union[int, list, tuple]] = None
) -> tf.Tensor:
    """
    Compute the Mean Absolute Error (MAE) between true and predicted values.
    
    Parameters:
    - y_true (Tensor): Ground truth values.
    - y_pred (Tensor): Predicted values.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 'sum', or 'mean'. Default is 'mean'.
    - axis (Optional[Union[int, list, tuple]]): The dimensions to reduce. Default is None, which reduces all dimensions.

    Returns:
    - Tensor: The MAE loss value, optionally reduced according to the 'reduction' parameter.
    """
    abs_error = tf.abs(y_true - y_pred)
    
    return _apply_reduction(abs_error, reduction=reduction, axis=axis)
