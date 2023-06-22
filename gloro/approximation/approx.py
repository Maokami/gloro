# Source : https://github.com/snu-ccl/approxCNN/blob/main/models/utils_approx.py

import tensorflow as tf
import numpy as np

import os
import warnings
import math

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
deg_dir = os.path.join(current_dir_path, "degreeResult")
coeff_dir = os.path.join(current_dir_path, "coeffResult")


class rangeException(Exception):
    def __init__(self, type, val, B):
        self.type = type
        self.val = val
        self.B = B
        print(
            "STOP! There is an input value",
            self.val,
            "But the B is",
            self.B,
            "for the",
            self.type,
            "function.",
        )


def poly_eval(x, coeff):
    coeff = tf.constant(coeff, dtype=tf.float64)
    if len(x.shape) == 2:
        range_tensor = tf.range(tf.shape(coeff)[0], dtype=tf.float64)[None, None, :]
        x_expanded = tf.expand_dims(x, -1)
        result = x_expanded**range_tensor * coeff
        return tf.reduce_sum(result, axis=-1)
    elif len(x.shape) == 4:
        range_tensor = tf.range(tf.shape(coeff)[0], dtype=tf.float64)[
            None, None, None, None, :
        ]
        x_expanded = tf.expand_dims(x, -1)
        result = x_expanded**range_tensor * coeff
        return tf.reduce_sum(result, axis=-1)


def sgn_approx(x, relu_dict):
    alpha = relu_dict["alpha"]
    B = relu_dict["B"]

    # Get degrees
    f = open(deg_dir + "/" + "deg_" + str(alpha) + ".txt")
    readed = f.readlines()
    comp_deg = [int(i) for i in readed]

    # Get coefficients
    f = open(coeff_dir + "/" + "coeff_" + str(alpha) + ".txt")
    coeffs_all_str = f.readlines()
    coeffs_all = [np.float64(i) for i in coeffs_all_str]
    i = 0

    if tf.executing_eagerly():
        condition = tf.reduce_sum(tf.cast(tf.abs(x) > B, tf.int32)) != 0
        max_val = tf.norm(x, ord=np.inf)
        if condition:
            warnings.warn(f"ReLU : max_val ({max_val}) exceeds B ({B})")
            # raise rangeException("relu", max_val, B)
    x = x / B

    for deg in comp_deg:
        coeffs_part = coeffs_all[i : (i + deg + 1)]
        x = poly_eval(x, coeffs_part)
        i += deg + 1

    return x


def ReLU_approx(x, relu_dict):
    x = tf.cast(x, dtype=tf.float64)
    sgnx = sgn_approx(x, relu_dict)
    output = x * (tf.constant(1.0, dtype=tf.float64) + sgnx) / 2

    # condition = (
    #     tf.reduce_sum(
    #         tf.cast(tf.reduce_max(tf.abs(x)) < tf.reduce_max(tf.abs(output)), tf.int32)
    #     )
    #     != 0
    # )
    # tf.print(
    #     tf.where(
    #         condition,
    #         f"ReLU_input : {tf.reduce_max(tf.abs(x))}\nReLU_output : {tf.reduce_max(tf.abs(output))}",
    #         "",
    #     )
    # )

    return output
