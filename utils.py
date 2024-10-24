import tensorflow as tf
from tensorflow.keras import backend as K

def dice(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Sorensen Dice coeffient.
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask

    return: Dice coeff value ranging between [0-1]
    '''
    smooth = 1e-15

    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    y_mask = tf.cast(tf.math.greater(y_mask, 0.5), tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2, 3))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth
    dice = tf.reduce_mean(((2*intersection+smooth) / union))

    return dice

def iou(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Intersection over Union measure
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask

    return: IoU measure value ranging between [0-1]
    '''
    smooth = 1e-15

    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2))

    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2)) + smooth

    iou = tf.reduce_mean((intersection)/(union-intersection))

    return iou

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 1)), tf.float32))
    false_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 0), tf.equal(tf.round(y_pred), 1)), tf.float32))
    
    smooth = 1e-5
    precision = true_positives / (true_positives + false_positives + smooth)
    
    return precision

def recall(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 1)), tf.float32))
    false_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 0)), tf.float32))

    smooth = 1e-5
    recall = true_positives / (true_positives + false_negatives + smooth)

    return recall

def f1(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 1)), tf.float32))
    false_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 0), tf.equal(tf.round(y_pred), 1)), tf.float32))
    false_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 0)), tf.float32))

    smooth = 1e-5
    precision = true_positives / (true_positives + false_positives + smooth)
    recall = true_positives / (true_positives + false_negatives + smooth)

    f1 = 2.0 * (precision * recall) / (precision + recall + smooth)

    return f1

def mae(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mae

def binary_cross_entropy(y_true, y_pred):
    # Clip the predicted values to avoid log(0) and log(1) which are undefined
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Compute binary cross-entropy
    bce = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return bce

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1.0 - dice_coeff

def combine_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + binary_cross_entropy(y_true, y_pred)
