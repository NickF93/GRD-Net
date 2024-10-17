import tensorflow as tf
import src.loss as loss

y_true = tf.random.uniform(shape=(2, 64, 64, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
y_true = tf.where(y_true > 0.5, 1.0, 0.0)
y_pred = tf.random.normal(shape=(2, 64, 64, 3), dtype=tf.float32)

custom_loss = loss.focal_loss(y_true, y_pred, from_logits=True, alpha=0.5, gamma=0.0, apply_class_balancing=False, reduction='mean')
custom_loss1 = 2 * loss.focal_loss(y_true, y_pred, from_logits=True, alpha=0.5, gamma=0.0, apply_class_balancing=True, reduction='mean')
tf_loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))
print(abs(float((custom_loss - tf_loss).numpy())))
print(abs(float((custom_loss1 - tf_loss).numpy())))

y_true = tf.random.normal((2, 16, 16, 3))
y_pred = tf.random.normal((2, 16, 16, 3))

alpha: float = 0.5
gamma: float = 0.0

y_true = tf.math.abs(y_true)
y_true = tf.clip_by_value(((y_true - tf.reduce_min(y_true)) / (tf.reduce_max(y_true) - tf.reduce_min(y_true))), 0.0, 1.0)
y_true = tf.cast(y_true >= 0.5, dtype=tf.float32)

# Custom focal loss with 'mean' reduction
custom_loss = loss.focal_loss(y_true, y_pred, from_logits=True, alpha=alpha, gamma=gamma, apply_class_balancing=False, reduction='mean')

# TensorFlow's built-in focal loss
tf_loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))

# Compare results step by step
print("Custom loss:", custom_loss.numpy())
print("TensorFlow loss:", tf_loss.numpy())

loss.ssim_rgb_loss(y_true, y_pred, reduction='none')

t=0
