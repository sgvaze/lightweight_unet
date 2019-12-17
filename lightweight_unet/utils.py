import tensorflow as tf
import keras.backend as K
import os
import numpy as np

from keras import losses

def regularized_dice_loss(y_true, y_pred):
    # FUNCTION EXPECTS GROUND TRUTH TO HAVE BORDERS WEIGHTED SUCH THAT INTENDED ZEROES HAVE VALUE 2, AND INTENDED ONES (ON BORDER)
    # HAVE VALUE 3

    smooth_val = 0.01  # Define how much smoothing is required in dice loss
    lam = 1e-3  # Define relative weighting of regularizer

    regularizer = boundary_regularizer(y_true, y_pred)

    # Set weighted borders to correct value
    y_true = set_value_to(y_true, 2, 0)
    y_true = set_value_to(y_true, 3, 1)

    # Compute dice
    truth_vec = K.flatten(y_true)
    pred_vec = K.flatten(y_pred)
    intersection = tf.reduce_sum(pred_vec * truth_vec)
    dice_coef = -(2. * intersection + smooth_val) / (
            tf.reduce_sum(pred_vec) + tf.reduce_sum(truth_vec) + smooth_val)

    return dice_coef + lam * regularizer

def boundary_regularizer(y_true, y_pred):
    # Set smoothing value
    smooth = 0.01

    # Set weighted borders to correct value
    y_true = set_value_to(y_true, 1, 0)
    y_true = set_value_to(y_true, 3, 1)
    y_true = set_value_to(y_true, 2, -1)

    truth_vec = K.flatten(y_true)
    pred_vec = K.flatten(y_pred)

    numerator = tf.reduce_sum(pred_vec * truth_vec)
    den_squared = tf.reduce_sum(tf.square(truth_vec)) + tf.reduce_sum(tf.square(pred_vec))
    denomoninator = tf.sqrt(den_squared)

    return -(numerator) / (denomoninator + smooth)


def dice_loss(y_true, y_pred):
    # FUNCTION EXPECTS GROUND TRUTH TO HAVE BORDERS WEIGHTED SUCH THAT INTENDED ZEROES HAVE VALUE 2, AND INTENDED ONES (ON BORDER)
    # HAVE VALUE 3

    smooth_val = 0.01  # Define how much smoothing is required in dice loss

    # Set weighted borders to correct value
    y_true = set_value_to(y_true, 2, 0)
    y_true = set_value_to(y_true, 3, 1)

    # Compute dice
    truth_vec = K.flatten(y_true)
    pred_vec = K.flatten(y_pred)
    intersection = tf.reduce_sum(pred_vec * truth_vec)
    dice_coef = -(2. * intersection + smooth_val) / (
            tf.reduce_sum(pred_vec) + tf.reduce_sum(truth_vec) + smooth_val)

    return dice_coef


# If value of tensorflow tensor 'x' is equal to 'a', set it equal to 'b'
def set_value_to(x, a, b):
    # Find locations in x where the element is equal to a
    cond = tf.equal(x, a * tf.ones(tf.shape(x)))  # Boolean mask, 1 where element in x and a are equal
    out = tf.where(cond, b * tf.ones(tf.shape(x)),
                   x)  # Assign value of 'b' to places where 'cond' is true, otherwise assign value of 'x' element

    return out

def find_distillation_teacher_path(trainer):

    teacher_root_dir = trainer.trainer_cfg['teacher_root_folder']

    for folder in trainer.trainer_cfg['teacher_folders']:

        path = os.path.join(teacher_root_dir, folder, "log.txt")
        with open(path, "r") as f:
            for line in f.readlines():
                if 'num_blocks' in line:
                    if str(trainer.model_manager.model_cfg['num_blocks']) in line:

                        return path


def get_loss_list_and_weights(indices, total_weight, layers, boundary_regularizer = True):

    # Function returns a "loss_list" and "loss_weights" for distillation supervision
    #   "loss_list" is a list of losses (function pointers) for each layer in the student.
    #               In this case, all intermediate activations are trained with "mean_absolute_error" (mean L1 distance), final prediction is trained with Dice loss
    #   "loss_weights" is a scalar weighting the various losses. Weights are defined to be inversely proportional to size of intermediate feature map

    num_supervised_layers = len(indices)

    # Create list of losses
    loss_list = [losses.mean_absolute_error]*num_supervised_layers

    if boundary_regularizer:
        loss_list.append(regularized_dice_loss)
    else:
        loss_list.append(dice_loss)

    # Get loss weights inversely proportional to feature map area
    loss_weights = np.ndarray([0])
    for index in indices:
        out_shape = layers[index].output_shape
        weighting = 1/(out_shape[1]*out_shape[2])
        loss_weights = np.append(loss_weights, weighting)

    # Normalize loss weights as defined by "total_weight", which describes the contribution of supervised losses relative to L2 weight regularisation
    loss_weights *= total_weight/np.sum(loss_weights)
    loss_weights = np.ndarray.tolist(loss_weights)
    loss_weights.append(1)


    return loss_list, loss_weights


def get_distillation_indices(model):
    indices = []
    for i, layer in enumerate(model.layers):
        if "conv2d" in model.layers[i - 1].name and "batch_norm" in model.layers[i + 1].name:
            indices.append(i)
    return indices