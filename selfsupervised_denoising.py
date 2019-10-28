# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import argparse
import os
import sys
import time
import numpy as np
import imageio

import h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import PIL.Image
import math
import glob
import pickle
import re

import dnnlib
import dnnlib.tflib
import dnnlib.tflib.tfutil as tfutil
from dnnlib.tflib.autosummary import autosummary
import dnnlib.submission.submit as submit

#----------------------------------------------------------------------------
# Misc helpers.

def init_tf(seed=None):
    config_dict = {'graph_options.place_pruned_graph': True, 'gpu_options.allow_growth': True}
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31) if (seed is None) else seed)
        tfutil.create_session(config_dict, force_as_default=True)

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def convert_to_pil_image(image, drange=[0,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0, 255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)

def save_image(image, filename, drange=[0,1], quality=95):
    img = convert_to_pil_image(image, drange)
    if '.jpg' in filename:
        img.save(filename,"JPEG", quality=quality, optimize=True)
    else:
        img.save(filename)

def save_snapshot(submit_config, net, fname_postfix):
    dump_fname = os.path.join(submit_config.run_dir, "network-%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)

def compute_ramped_lrate(i, iteration_count, ramp_up_fraction, ramp_down_fraction, learning_rate):
    if ramp_up_fraction > 0.0:
        ramp_up_end_iter = iteration_count * ramp_up_fraction
        if i <= ramp_up_end_iter:
            t = (i / ramp_up_fraction) / iteration_count
            learning_rate = learning_rate * (0.5 - np.cos(t * np.pi)/2)

    if ramp_down_fraction > 0.0:
        ramp_down_start_iter = iteration_count * (1 - ramp_down_fraction)
        if i >= ramp_down_start_iter:
            t = ((i - ramp_down_start_iter) / ramp_down_fraction) / iteration_count
            learning_rate = learning_rate * (0.5 + np.cos(t * np.pi)/2)**2

    return learning_rate

def clip_to_uint8(arr):
    if isinstance(arr, np.ndarray):
        return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    x = tf.clip_by_value(arr * 255.0 + 0.5, 0, 255)
    return tf.cast(x, tf.uint8)

def calculate_psnr(a, b, axis=None):
    a, b = [clip_to_uint8(x) for x in [a, b]]
    if isinstance(a, np.ndarray):
        a, b = [x.astype(np.float32) for x in [a, b]]
        x = np.mean((a - b)**2, axis=axis)
        return np.log10((255 * 255) / x) * 10.0
    a, b = [tf.cast(x, tf.float32) for x in [a, b]]
    x = tf.reduce_mean((a - b)**2, axis=axis)
    return tf.log((255 * 255) / x) * (10.0 / math.log(10))

#----------------------------------------------------------------------------

def poisson(x, lam):
    if lam > 0.0:
        return np.random.poisson(x * lam) / lam
    return 0.0 * x

#----------------------------------------------------------------------------

# Number of channels enforcer while retaining dtype.
def set_color_channels(x, num_channels):
    assert x.shape[0] in [1, 3, 4]
    x = x[:min(x.shape[0], 3)] # drop possible alpha channel
    if x.shape[0] == num_channels:
        return x
    elif x.shape[0] == 1:
        return np.tile(x, [3, 1, 1])
    y = np.mean(x, axis=0, keepdims=True)
    if np.issubdtype(x.dtype, np.integer):
        y = np.round(y).astype(x.dtype)
    return y

#----------------------------------------------------------------------------

def load_datasets(num_channels, dataset_dir, train_dataset, validation_dataset, prune_dataset=None):
    # Training set.

    if train_dataset is None:
        print("Not loading training data.")
        train_images = []
    else:
        fn = submit.get_path_from_template(train_dataset)
        print("Loading training dataset from '%s'." % fn)

        h5file = h5py.File(fn, 'r')
        num = h5file['images'].shape[0]
        print("Dataset contains %d images." % num)

        if prune_dataset is not None:
            num = prune_dataset
            print("Pruned down to %d first images." % num)

        # Load the images.
        train_images = [None] * num
        bs = 1024
        for i in range(0, num, bs):
            sys.stdout.write("\r%d / %d .." % (i, num))
            n = min(bs, num - i)
            img = h5file['images'][i : i+n]
            shp = h5file['shapes'][i : i+n]
            for j in range(n):
                train_images[i+j] = set_color_channels(np.reshape(img[j], shp[j]), num_channels)

        print("\nLoading done.")
        h5file.close()

    if validation_dataset in ['kodak', 'bsd300', 'set14']:
        paths = { 'kodak':  os.path.join(dataset_dir, 'kodak', '*.png'),
                  'bsd300': os.path.join(dataset_dir, 'BSDS300', 'images/test/*.jpg'),   # Just the 100 test images
                  'set14':  os.path.join(dataset_dir, 'Set14', '*.png')}
        fn = submit.get_path_from_template(paths[validation_dataset])
        print("Loading validation dataset from '%s'." % fn)
        validation_images = [imageio.imread(x) for x in glob.glob(fn)]
        validation_images = [x[..., np.newaxis] if len(x.shape) == 2 else x for x in validation_images] # Add channel axis to grayscale images.
        validation_images = [x.transpose([2, 0, 1]) for x in validation_images]
        validation_images = [set_color_channels(x, num_channels) for x in validation_images] # Enforce RGB/grayscale mode.
        print("Loaded %d images." % len(validation_images))

    # Pad the validation images to size.
    validation_image_size = [max([x.shape[axis] for x in validation_images]) for axis in [1, 2]]
    validation_image_size = [(x + 31) // 32 * 32 for x in validation_image_size] # Round up to a multiple of 32.
    validation_image_size = [max(validation_image_size) for x in validation_image_size] # Square it up for the rotators.
    print("Validation image padded size = [%d, %d]." % (validation_image_size[0], validation_image_size[1]))

    return train_images, validation_images, validation_image_size

#----------------------------------------------------------------------------
# Backbone autoencoder network, optional blind spot.

def analysis_network(image, num_output_components, blindspot, zero_last=False):

    def conv(n, name, n_out, size=3, gain=np.sqrt(2), zero_weights=False):
        if blindspot: assert (size % 2) == 1
        ofs = 0 if (not blindspot) else size // 2

        with tf.variable_scope(name):
            wshape = [size, size, n.shape[1].value, n_out]
            wstd = gain / np.sqrt(np.prod(wshape[:-1])) # He init.
            W = tf.get_variable('W', shape=wshape, initializer=(tf.initializers.zeros() if zero_weights else tf.initializers.random_normal(0., wstd)))
            b = tf.get_variable('b', shape=[n_out], initializer=tf.initializers.zeros())
            if ofs > 0: n = tf.pad(n, [[0, 0], [0, 0], [ofs, 0], [0, 0]])
            n = tf.nn.conv2d(n, W, strides=[1]*4, padding='SAME', data_format='NCHW') + tf.reshape(b, [1, -1, 1, 1])
            if ofs > 0: n = n[:, :, :-ofs, :]
        return n

    def up(n, name):
        with tf.name_scope(name):
            s = tf.shape(n)
            s = [-1, n.shape[1], s[2], s[3]]
            n = tf.reshape(n, [s[0], s[1], s[2], 1, s[3], 1])
            n = tf.tile(n, [1, 1, 1, 2, 1, 2])
            n = tf.reshape(n, [s[0], s[1], s[2] * 2, s[3] * 2])
        return n

    def down(n, name):
        with tf.name_scope(name):
            if blindspot: # Shift and pad if blindspot.
                n = tf.pad(n[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])
            n = tf.nn.max_pool(n, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
        return n

    def rotate(x, angle):
        if   angle == 0:   return x
        elif angle == 90:  return tf.transpose(x[:, :, :, ::-1], [0, 1, 3, 2])
        elif angle == 180: return x[:, :, ::-1, ::-1]
        elif angle == 270: return tf.transpose(x[:, :, ::-1, :], [0, 1, 3, 2])

    def concat(name, layers):
        return tf.concat(layers, axis=1, name=name)

    def LR(n, alpha=0.1):
        return tf.nn.leaky_relu(n, alpha=alpha, name='lrelu')

    # Input stage.

    if not blindspot:
        x = image
    else:
        x = tf.concat([rotate(image, a) for a in [0, 90, 180, 270]], axis=0)

    # Encoder part.

    pool0 = x
    x = LR(conv(x, 'enc_conv0', 48))
    x = LR(conv(x, 'enc_conv1', 48))
    x = down(x, 'pool1'); pool1 = x

    x = LR(conv(x, 'enc_conv2', 48))
    x = down(x, 'pool2'); pool2 = x

    x = LR(conv(x, 'enc_conv3', 48))
    x = down(x, 'pool3'); pool3 = x

    x = LR(conv(x, 'enc_conv4', 48))
    x = down(x, 'pool4'); pool4 = x

    x = LR(conv(x, 'enc_conv5', 48))
    x = down(x, 'pool5')

    x = LR(conv(x, 'enc_conv6', 48))

    # Decoder part.

    x = up(x, 'upsample5')
    x = concat('concat5', [x, pool4])
    x = LR(conv(x, 'dec_conv5a', 96))
    x = LR(conv(x, 'dec_conv5b', 96))

    x = up(x, 'upsample4')
    x = concat('concat4', [x, pool3])
    x = LR(conv(x, 'dec_conv4a', 96))
    x = LR(conv(x, 'dec_conv4b', 96))

    x = up(x, 'upsample3')
    x = concat('concat3', [x, pool2])
    x = LR(conv(x, 'dec_conv3a', 96))
    x = LR(conv(x, 'dec_conv3b', 96))

    x = up(x, 'upsample2')
    x = concat('concat2', [x, pool1])
    x = LR(conv(x, 'dec_conv2a', 96))
    x = LR(conv(x, 'dec_conv2b', 96))

    x = up(x, 'upsample1')
    x = concat('concat1', [x, pool0])

    # Output stages.

    if blindspot:
        # Blind-spot output stages.
        x = LR(conv(x, 'dec_conv1a', 96))
        x = LR(conv(x, 'dec_conv1b', 96))
        x = tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])   # Shift and pad.
        x = tf.split(x, 4, axis=0)                                      # Split into rotations.
        x = [rotate(y, a) for y, a in zip(x, [0, 270, 180, 90])]        # Counterrotate.
        x = tf.concat(x, axis=1)                                        # Combine on channel axis.
        x = LR(conv(x, 'nin_a', 96*4, size=1))
        x = LR(conv(x, 'nin_b', 96, size=1))
        x = conv(x, 'nin_c', num_output_components, size=1, gain=1.0, zero_weights=zero_last)
    else:
        # Baseline network with postprocessing layers -- keep feature maps and distill with 1x1 convolutions.
        x = LR(conv(x, 'dec_conv1a', 96))
        x = LR(conv(x, 'dec_conv1b', 96))
        x = LR(conv(x, 'nin_a', 96, size=1))
        x = LR(conv(x, 'nin_b', 96, size=1))
        x = conv(x, 'nin_c', num_output_components, size=1, gain=1.0, zero_weights=zero_last)

    # Return results.

    return x

#----------------------------------------------------------------------------

def blindspot_pipeline(noisy_in,
                       noise_params_in,
                       diagonal_covariance  = False,
                       input_shape          = None,
                       noise_style          = None,
                       noise_params         = None,
                       **_kwargs):

    num_channels = input_shape[1]
    assert num_channels in [1, 3]
    assert noise_style in ['gauss', 'poisson', 'impulse']
    assert noise_params in ['known', 'global', 'per_image']

    # Shapes.
    noisy_in.set_shape(input_shape)
    noise_params_in.set_shape(input_shape[:1] + [1, 1, 1])

    # Clean data distribution.
    num_output_components = num_channels + (num_channels * (num_channels + 1)) // 2 # Means, triangular A.
    if diagonal_covariance:
        num_output_components = num_channels * 2 # Means, diagonal of A.
    net_out = analysis_network(noisy_in, num_output_components, blindspot=True)
    net_out = tf.cast(net_out, tf.float64)
    mu_x = net_out[:, 0:num_channels, ...] # Means (NCHW).
    A_c = net_out[:, num_channels:num_output_components, ...] # Components ot triangular A.
    if num_channels == 1:
        sigma_x = A_c ** 2 # N1HW
    elif num_channels == 3:
        A_c = tf.transpose(A_c, [0, 2, 3, 1]) # NHWC
        if diagonal_covariance:
            c00 = A_c[..., 0]**2
            c11 = A_c[..., 1]**2
            c22 = A_c[..., 2]**2
            zro = tf.zeros_like(c00)
            c0 = tf.stack([c00, zro, zro], axis=-1) # NHW3
            c1 = tf.stack([zro, c11, zro], axis=-1) # NHW3
            c2 = tf.stack([zro, zro, c22], axis=-1) # NHW3
        else:
            # Calculate A^T * A
            c00 = A_c[..., 0]**2 + A_c[..., 1]**2 + A_c[..., 2]**2 # NHW
            c01 = A_c[..., 1]*A_c[..., 3] + A_c[..., 2]*A_c[..., 4]
            c02 = A_c[..., 2]*A_c[..., 5]
            c11 = A_c[..., 3]**2 + A_c[..., 4]**2
            c12 = A_c[..., 4]*A_c[..., 5]
            c22 = A_c[..., 5]**2
            c0 = tf.stack([c00, c01, c02], axis=-1) # NHW3
            c1 = tf.stack([c01, c11, c12], axis=-1) # NHW3
            c2 = tf.stack([c02, c12, c22], axis=-1) # NHW3
        sigma_x = tf.stack([c0, c1, c2], axis=-1) # NHW33

    # Data on which noise parameter estimation is based.
    if noise_params == 'global':
        # Global constant over the entire dataset.
        noise_est_out = tf.get_variable('noise_data', shape=[1, 1, 1, 1], initializer=tf.initializers.constant(0.0)) # 1111
        noise_est_out = tf.cast(noise_est_out, tf.float64)
    elif noise_params == 'per_image':
        # Separate analysis network.
        with tf.variable_scope('param_estimation_net'):
            noise_est_out = analysis_network(noisy_in, 1, blindspot=False, zero_last=True) # N1HW
        noise_est_out = tf.reduce_mean(noise_est_out, axis=[2, 3], keepdims=True) # N111
        noise_est_out = tf.cast(noise_est_out, tf.float64)

    # Cast remaining data into float64.
    noisy_in = tf.cast(noisy_in, tf.float64)
    noise_params_in = tf.cast(noise_params_in, tf.float64)

    # Remap noise estimate to ensure it is always positive and starts near zero.
    if noise_params != 'known':
        noise_est_out = tf.nn.softplus(noise_est_out - 4.0) + 1e-3

    # Distill noise parameters from learned/known data.
    if noise_style == 'gauss':
        if noise_params == 'known':
            noise_std = tf.maximum(noise_params_in, 1e-3) # N111
        else:
            noise_std = noise_est_out
    elif noise_style == 'poisson': # Simple signal-dependent Poisson approximation [Hasinoff 2012].
        if noise_params == 'known':
            noise_std = (tf.maximum(mu_x, tf.constant(1e-3, tf.float64)) / noise_params_in) ** 0.5 # NCHW
        else:
            noise_std = (tf.maximum(mu_x, tf.constant(1e-3, tf.float64)) * noise_est_out) ** 0.5 # NCHW
    elif noise_style == 'impulse':
        if noise_params == 'known':
            noise_std = noise_params_in # N111, actually the alpha.
        else:
            noise_std = noise_est_out

    # Casts and vars.
    noise_std = tf.cast(noise_std, tf.float64)
    I = tf.eye(num_channels, batch_shape=[1, 1, 1], dtype=tf.float64)
    Ieps = I * tf.constant(1e-6, dtype=tf.float64)
    zero64 = tf.constant(0.0, dtype=tf.float64)

    # Helpers.
    def batch_mvmul(m, v): # Batched (M * v).
        return tf.reduce_sum(m * v[..., tf.newaxis, :], axis=-1)
    def batch_vtmv(v, m): # Batched (v^T * M * v).
        return tf.reduce_sum(v[..., :, tf.newaxis] * v[..., tf.newaxis, :] * m, axis=[-2, -1])
    def batch_vvt(v): # Batched (v * v^T).
        return v[..., :, tf.newaxis] * v[..., tf.newaxis, :]

    # Negative log-likelihood loss and posterior mean estimation.
    if noise_style in ['gauss', 'poisson']:
        if num_channels == 1:
            sigma_n = noise_std**2 # N111 / N1HW
            sigma_y = sigma_x + sigma_n # N1HW. Total variance.
            loss_out = ((noisy_in - mu_x) ** 2) / sigma_y + tf.log(sigma_y) # N1HW
            pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (sigma_x + sigma_n) # N1HW
            net_std_out = (sigma_x**0.5)[:, 0, ...] # NHW
            noise_std_out = noise_std[:, 0, ...] # N11 / NHW
            if noise_params != 'known':
                loss_out = loss_out - 0.1 * noise_std # Balance regularization.
        else:
            # Training loss.
            sigma_n = tf.transpose(noise_std**2, [0, 2, 3, 1])[..., tf.newaxis] * I # NHWC1 * NHWCC = NHWCC
            sigma_y = sigma_x + sigma_n # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
            sigma_y_inv = tf.linalg.inv(sigma_y) # NHWCC
            mu_x2 = tf.transpose(mu_x, [0, 2, 3, 1]) # NHWC
            noisy_in2 = tf.transpose(noisy_in, [0, 2, 3, 1]) # NHWC
            diff = (noisy_in2 - mu_x2) # NHWC
            diff = -0.5 * batch_vtmv(diff, sigma_y_inv) # NHW
            dets = tf.linalg.det(sigma_y) # NHW
            dets = tf.maximum(zero64, dets) # NHW. Avoid division by zero and negative square roots.
            loss_out = 0.5 * tf.log(dets) - diff # NHW
            if noise_params != 'known':
                loss_out = loss_out - 0.1 * tf.reduce_mean(noise_std, axis=1) # Balance regularization.

            # Posterior mean estimate.
            sigma_x_inv = tf.linalg.inv(sigma_x + Ieps) # NHWCC
            sigma_n_inv = tf.linalg.inv(sigma_n + Ieps) # NHWCC
            pme_c1 = tf.linalg.inv(sigma_x_inv + sigma_n_inv + Ieps) # NHWCC
            pme_c2 = batch_mvmul(sigma_x_inv, mu_x2) # NHWCC * NHWC -> NHWC
            pme_c2 = pme_c2 + batch_mvmul(sigma_n_inv, noisy_in2) # NHWC
            pme_out = batch_mvmul(pme_c1, pme_c2) # NHWC
            pme_out = tf.transpose(pme_out, [0, 3, 1, 2]) # NCHW

            # Summary statistics.
            net_std_out = tf.maximum(zero64, tf.linalg.det(sigma_x))**(1.0/6.0) # NHW
            noise_std_out = tf.maximum(zero64, tf.linalg.det(sigma_n))**(1.0/6.0) # N11 / NHW
    elif noise_style == 'impulse':
        alpha = noise_std # N111.
        if num_channels == 1:
            raise NotImplementedError
        else:
            # Preliminaries.
            sigma_x = sigma_x + Ieps # NHWCC. Inflate by epsilon.
            sigma_x_inv = tf.linalg.inv(sigma_x) # NHWCC
            mu_x2 = tf.transpose(mu_x, [0, 2, 3, 1]) # NHWC
            noisy_in2 = tf.transpose(noisy_in, [0, 2, 3, 1]) # NHWC
            diff = (noisy_in2 - mu_x2) # NHWC
            diff = batch_vtmv(diff, sigma_x_inv) # NHW
            dets = tf.linalg.det(sigma_x) # NHW
            dets = tf.maximum(tf.constant(1e-9, dtype=tf.float64), dets) # NHW. Avoid division by zero and negative square roots.
            g = tf.exp(-0.5 * diff) / ((2.0 * np.pi)**num_channels * dets)**0.5 # NHW
            g = g[..., tf.newaxis] # NHW1

            # Posterior mean estimate.
            h = (1.0 - alpha) * g # NHW1
            pme_out = (alpha * mu_x2 + h * noisy_in2) / (alpha + h)
            pme_out = tf.transpose(pme_out, [0, 3, 1, 2]) # NCHW

            # Training loss with the modified stats.
            mu_y2 = alpha * .5 + (1.0 - alpha) * mu_x2 # NHWC
            alpha = alpha[..., tf.newaxis] # n1111
            sigma_y = alpha * (1.0/4.0 + I/12.0) + (1.0 - alpha) * (sigma_x + batch_vvt(mu_x2)) - batch_vvt(mu_y2) # NHWCC
            sigma_y_inv = tf.linalg.inv(sigma_y) # NHWCC
            diff = (noisy_in2 - mu_y2) # NHWC
            diff = batch_vtmv(diff, sigma_y_inv) # NHW
            dets = tf.linalg.det(sigma_y) # NHW
            dets = tf.maximum(tf.constant(1e-9, dtype=tf.float64), dets) # NHW
            loss_out = diff + tf.log(dets) # NHW

            # Summary statistics.
            net_std_out = tf.maximum(zero64, tf.linalg.det(sigma_x))**(1.0/6.0) # NHW. Cube root of volumetric scaling factor.
            noise_std_out = alpha[..., 0, 0] / 255.0 * 100.0 # N11 / NHW. Shows as percentage in output.

    return mu_x, pme_out, loss_out, net_std_out, noise_std_out

#----------------------------------------------------------------------------

def simple_pipeline(clean_in,
                    noisy_in,
                    L_exponent_in,
                    noise_style = None,
                    input_shape = None,
                    blindspot = False,
                    noisy_targets = False,
                    **_kwargs):

    clean_in.set_shape(input_shape)
    noisy_in.set_shape(input_shape)
    L_exponent_in.set_shape([])

    x = analysis_network(noisy_in, input_shape[1], blindspot=blindspot)

    if noise_style == 'impulse' and noisy_targets: # Cannot use L2 loss because mean changes
        loss_out = (tf.abs(x - clean_in) + 1e-8) ** L_exponent_in
    else:
        loss_out = (x - clean_in) ** 2.0

    net_std_out, noise_std_out = [tf.zeros_like(noisy_in) for x in range(2)]
    return x, x, loss_out, net_std_out, noise_std_out

#----------------------------------------------------------------------------

def get_scrambled_indices(num, bs):
    assert num > 0
    i, x = 0, []
    while True:
        res = x[i : i + bs]
        i += bs
        while len(res) < bs:
            x = list(np.arange(num))
            np.random.shuffle(x)
            i = bs - len(res)
            res += x[:i]
        yield res

#----------------------------------------------------------------------------

def random_crop_numpy(img, crop_size):
    y = np.random.randint(img.shape[1] - crop_size + 1)
    x = np.random.randint(img.shape[2] - crop_size + 1)
    return img[:, y : y+crop_size, x : x+crop_size]

#----------------------------------------------------------------------------
# Noise implementations.
#----------------------------------------------------------------------------

operation_seed_counter = 0
def noisify(x, style):
    def get_seed():
        global operation_seed_counter
        operation_seed_counter += 1
        return operation_seed_counter

    if style.startswith('gauss'): # Gaussian noise with constant/variable std.dev.
        params = [float(p) / 255.0 for p in style.replace('gauss', '', 1).split('_')]
        if len(params) == 1:
            std = params[0]
        elif len(params) == 2:
            min_std, max_std = params
            std = tf.random_uniform(shape=[tf.shape(x)[0], 1, 1, 1], minval=min_std, maxval=max_std, seed=get_seed())
        return x + tf.random_normal(shape=tf.shape(x), seed=get_seed()) * std, std
    elif style.startswith('poisson'): # Poisson noise with constant/variable lambda.
        params = [float(p) for p in style.replace('poisson', '', 1).split('_')]
        if len(params) == 1:
            lam = params[0]
        elif len(params) == 2:
            min_lam, max_lam = params
            lam = tf.random_uniform(shape=[tf.shape(x)[0], 1, 1, 1], minval=min_lam, maxval=max_lam, seed=get_seed())
        x = x * lam
        with tf.device("/cpu:0"):
            x = tf.random_poisson(x, [1], seed=get_seed())
        return x[0] / lam, lam
    elif style.startswith('impulse'): # Random replacement with constant/variable alpha.
        params = [float(p) * 0.01 for p in style.replace('impulse', '', 1).split('_')]
        msh = tf.shape(x[:, :1, ...])
        if len(params) == 1:
            alpha = params[0]
            keep_mask = tf.where(tf.random_uniform(shape=msh, seed=get_seed()) >= alpha, tf.ones(shape=msh), tf.zeros(shape=msh))
        elif len(params) == 2:
            min_alpha, max_alpha = params
            alpha = tf.random_uniform(shape=[tf.shape(x)[0], 1, 1, 1], minval=min_alpha, maxval=max_alpha, seed=get_seed())
            keep_mask = tf.where(tf.random_uniform(shape=msh, seed=get_seed()) >= tf.ones(shape=msh) * alpha, tf.ones(shape=msh), tf.zeros(shape=msh))
        noise = tf.random_uniform(shape=tf.shape(x), seed=get_seed())
        return x * keep_mask + noise * (1.0 - keep_mask), alpha

#----------------------------------------------------------------------------
# Training loop.
#----------------------------------------------------------------------------

def train(submit_config,
          num_iter              = 1000000,
          train_resolution      = 256,
          minibatch_size        = 4,
          learning_rate         = 3e-4,
          rampup_fraction       = 0.1,
          rampdown_fraction     = 0.3,
          snapshot_every        = 0,        # Export network snapshot every n images (must be divisible by minibatch).
          pipeline              = None,
          diagonal_covariance   = False,    # Force non-diagonal covariances to zero (per-channel univariate).
          noise_style           = None,
          noise_params          = None,     # 'known', 'global', 'per_image'
          train_dataset         = None,
          validation_dataset    = None,
          validation_repeats    = 1,
          prune_dataset         = None,
          num_channels          = None,
          print_interval        = 1000,
          eval_interval         = 10000,
          eval_network          = None,
          config_name           = None,
          dataset_dir           = None):

    # Are we in evaluation mode?
    eval_mode = eval_network is not None

    # Initialize Tensorflow.
    if eval_mode:
        init_tf(0) # Use fixed seeds if evaluating a network.
        np.random.seed(0)
    else:
        init_tf() # Use a random random seed.

    # Get going.
    ctx = dnnlib.RunContext(submit_config)
    run_dir = submit_config.run_dir
    img_dir = os.path.join(run_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Load the data.
    train_images, validation_images, validation_image_size = load_datasets(num_channels, dataset_dir, None if eval_mode else train_dataset, validation_dataset, prune_dataset)

    # Repeat validation set if asked to.
    original_validation_image_count = len(validation_images) # Avoid exporting the duplicate images.
    if validation_repeats > 1:
        print("Repeating the validation set %d times." % validation_repeats)
        validation_images = validation_images * validation_repeats
        validation_image_size = validation_image_size * validation_repeats

    # Construct the network.
    input_shape = [None, num_channels, None, None]
    with tf.device("/gpu:0"):
        if eval_mode:
            print("Evaluating network '%s'." % eval_network)
            with open(eval_network, 'rb') as f:
                net = pickle.load(f)
        else:
            if noise_style.startswith('gauss'):   net_noise_style = 'gauss'
            if noise_style.startswith('poisson'): net_noise_style = 'poisson'
            if noise_style.startswith('impulse'): net_noise_style = 'impulse'

            if pipeline == 'blindspot':
                net = dnnlib.tflib.Network('net', 'selfsupervised_denoising.blindspot_pipeline', input_shape=input_shape, noise_style=net_noise_style, noise_params=noise_params, diagonal_covariance=diagonal_covariance)
            elif pipeline == 'blindspot_mean':
                net = dnnlib.tflib.Network('net', 'selfsupervised_denoising.simple_pipeline', input_shape=input_shape, noise_style=net_noise_style, blindspot=True, noisy_targets=True)
            elif pipeline == 'n2c':
                net = dnnlib.tflib.Network('net', 'selfsupervised_denoising.simple_pipeline', input_shape=input_shape, noise_style=net_noise_style, blindspot=False, noisy_targets=False)
            elif pipeline == 'n2n':
                net = dnnlib.tflib.Network('net', 'selfsupervised_denoising.simple_pipeline', input_shape=input_shape, noise_style=net_noise_style, blindspot=False, noisy_targets=True)

    # Data splits.
    with tf.name_scope('Inputs'), tf.device("/cpu:0"):
        learning_rate_in = tf.placeholder(tf.float32, name='learning_rate_in', shape=[])
        L_exponent_in = tf.placeholder(tf.float32, name='L_exponent', shape=[])
        clean_in = tf.placeholder(tf.float32, shape=input_shape)
        clean_in_split = tf.split(clean_in, submit_config.num_gpus)

    # Optimizer.
    opt = dnnlib.tflib.Optimizer(tf_optimizer='tf.train.AdamOptimizer', learning_rate=learning_rate_in, beta1=0.9, beta2=0.99)

    # Per-gpu stuff.
    train_loss = 0.
    train_psnr = 0.
    train_psnr_pme = 0.
    gpu_outputs = []
    for gpu in range(submit_config.num_gpus):
        with tf.device("/gpu:%d" % gpu):
            net_gpu = net if gpu == 0 else net.clone()
            clean_in_gpu = clean_in_split[gpu]
            noisy_in_gpu, noise_coeff = noisify(clean_in_gpu, noise_style)

            if pipeline == 'blindspot_mean':
                reference_in_gpu = noisy_in_gpu
            elif pipeline == 'n2n':
                reference_in_gpu, _ = noisify(clean_in_gpu, noise_style) # Another noise instantiation.
            else:
                reference_in_gpu = clean_in_gpu

            noise_coeff = tf.zeros([tf.shape(noisy_in_gpu)[0], 1, 1, 1]) + noise_coeff # Broadcast to [n, 1, 1, 1] shape.

            # Support for networks that were exported from an older version of code and loaded for evaluation purposes.
            if net.num_inputs == 5:
                mu_x, pme_out, loss_out, net_std_out, noise_std_out, _ = net_gpu.get_output_for(reference_in_gpu, noisy_in_gpu, noise_coeff, tf.constant(1e-6, dtype=tf.float32), tf.constant(1e-1, dtype=tf.float32))
            else:
                if pipeline == 'blindspot':
                   if net.num_inputs == 3:
                        mu_x, pme_out, loss_out, net_std_out, noise_std_out, _ = net_gpu.get_output_for(noisy_in_gpu, noise_coeff, L_exponent_in) # Previous version.
                   else:
                        mu_x, pme_out, loss_out, net_std_out, noise_std_out = net_gpu.get_output_for(noisy_in_gpu, noise_coeff)
                else:
                    if net.num_inputs == 4:
                        mu_x, pme_out, loss_out, net_std_out, noise_std_out, _ = net_gpu.get_output_for(reference_in_gpu, noisy_in_gpu, noise_coeff, L_exponent_in) # Previous version.
                    else:
                        mu_x, pme_out, loss_out, net_std_out, noise_std_out = net_gpu.get_output_for(reference_in_gpu, noisy_in_gpu, L_exponent_in)

            gpu_outputs.append([mu_x, pme_out, loss_out, net_std_out, noise_std_out, noisy_in_gpu])

            # Loss.
            loss = tf.reduce_mean(loss_out)

             # PSNR during training.
            psnr = tf.reduce_mean(calculate_psnr(mu_x, clean_in_gpu, axis=[1, 2, 3]))
            psnr_pme = tf.reduce_mean(calculate_psnr(pme_out, clean_in_gpu, axis=[1, 2, 3]))
            with tf.control_dependencies([autosummary("train_loss", loss), autosummary("train_psnr", psnr), autosummary("train_psnr_pme", psnr_pme)]):
                opt.register_gradients(loss, net_gpu.trainables)

        # Accumulation not on the GPU.
        train_loss += loss / submit_config.num_gpus
        train_psnr += psnr / submit_config.num_gpus
        train_psnr_pme += psnr_pme / submit_config.num_gpus

    # Total outputs.
    mu_x_out, pme_out, loss_out, net_std_out, noise_std_out, noisy_out = [tf.concat(x, axis=0) for x in zip(*gpu_outputs)]

    # Train step op.
    train_step = opt.apply_updates()

    # Create a log file for Tensorboard.
    if not eval_mode:
        summary_log = tf.summary.FileWriter(run_dir)
        summary_log.add_graph(tf.get_default_graph())

    # Training image index generator.
    index_generator = get_scrambled_indices(len(train_images), minibatch_size)

    # Init stats.
    print_last, eval_last = 0, 0
    loss_acc, loss_n = 0., 0.
    psnr_acc, psnr_pme_acc = 0., 0.
    std_net_acc, std_noise_acc = 0., 0.
    valid_psnr_mu, valid_psnr_pme = 0., 0.
    t_start = time.time()

    # Train.
    if eval_mode:
        print('Evaluating network with %d images.' % len(validation_images))
    else:
        print('Training for %d images.' % num_iter)

    for n in range(0, num_iter + minibatch_size, minibatch_size):
        if ctx.should_stop():
            break

        # Save snapshot.
        if (n > 0) and (snapshot_every > 0) and (n % snapshot_every == 0):
            save_snapshot(submit_config, net, '%08d' % n)

        # Set up training step.
        lr = compute_ramped_lrate(n, num_iter, rampup_fraction, rampdown_fraction, learning_rate)
        L_exponent = 0.5 if eval_mode else max(0.5, 2.0 - 2.0 * n / num_iter)

        # Training step unless in evaluation mode.
        if not eval_mode:
            # Get clean images from training set.
            clean = np.zeros([minibatch_size, num_channels, train_resolution, train_resolution], dtype=np.uint8)
            for i, j in enumerate(next(index_generator)):
                clean[i] = random_crop_numpy(train_images[j], train_resolution)
            clean = adjust_dynamic_range(clean, [0, 255], [0.0, 1.0])

            # Run training step.
            feed_dict = {clean_in: clean, learning_rate_in: lr, L_exponent_in: L_exponent}
            loss_val, psnr_val, psnr_pme_val, net_std_val, noise_std_val, _ = tfutil.run([train_loss, train_psnr, train_psnr_pme, net_std_out, noise_std_out, train_step], feed_dict)

            # Accumulate stats.
            loss_acc += loss_val
            psnr_acc += psnr_val
            psnr_pme_acc += psnr_pme_val
            std_net_acc += np.mean(net_std_val)
            std_noise_acc += np.mean(noise_std_val)
            loss_n += 1.0

        # Print.
        if n == 0 or n >= print_last + print_interval:
            loss_n = max(loss_n, 1.0)
            loss_acc /= loss_n
            psnr_acc /= loss_n
            psnr_pme_acc /= loss_n
            std_net_acc = std_net_acc / loss_n * 255.0
            std_noise_acc = std_noise_acc / loss_n * 255.0
            t_iter = time.time() - t_start
            print("%8d: time=%6.2f, loss=%8.4f, train_psnr=%8.4f, train_psnr_pme=%8.4f, std_net=%8.4f, std_noise=%8.4f" % (n, t_iter, loss_acc, psnr_acc, psnr_pme_acc, autosummary('std_net', std_net_acc), autosummary('std_noise', std_noise_acc)), end='')
            ctx.update(loss='%.2f %.2f' % (psnr_pme_acc, valid_psnr_pme), cur_epoch=n, max_epoch=num_iter)
            print_last += print_interval if (n > 0) else 0
            loss_acc, loss_n = 0., 0.
            psnr_acc, psnr_pme_acc = 0., 0.
            std_net_acc, std_noise_acc = 0., 0.
            t_start = time.time()

            # Measure and export validation images.
            if n == 0 or n >= eval_last + eval_interval or n == num_iter:
                valid_psnr_mu = 0.
                valid_psnr_pme = 0.
                bs = submit_config.num_gpus # Validation batch size.
                for idx0 in range(0, len(validation_images), bs):
                    num = min(bs, len(validation_images) - idx0)
                    idx = list(range(idx0, idx0 + bs))
                    idx = [min(x, len(validation_images) - 1) for x in idx]
                    val_input = []
                    val_sz = []
                    for i in idx:
                        img = validation_images[i][np.newaxis, ...]
                        img = adjust_dynamic_range(img, [0, 255], [0.0, 1.0])
                        sz = img.shape[2:]
                        img = np.pad(img, [[0, 0], [0, 0], [0, validation_image_size[0] - sz[0]], [0, validation_image_size[1] - sz[1]]], 'reflect')
                        val_input.append(img)
                        val_sz.append(sz)
                    val_input = np.concatenate(val_input, axis=0) # Batch of validation images.

                    # Run the actual step.
                    feed_dict = {clean_in: val_input}
                    mu_x, net_std, pme, noisy = tfutil.run([mu_x_out, net_std_out, pme_out, noisy_out], feed_dict)

                    # Process the result images.
                    for i, j in enumerate(idx[:num]):
                        crop_val_input, crop_mu_x, crop_pme, crop_noisy = [x[i, :, :val_sz[i][0], :val_sz[i][1]] for x in [val_input, mu_x, pme, noisy]]
                        crop_net_std = net_std[i, :val_sz[i][0], :val_sz[i][1]] # HW grayscale
                        crop_net_std /= 10.0 / 255.0 # white = 10 ULPs in U8.
                        valid_psnr_mu += calculate_psnr(crop_mu_x, crop_val_input) / len(validation_images)
                        valid_psnr_pme += calculate_psnr(crop_pme, crop_val_input) / len(validation_images)

                        if (eval_mode and (j < original_validation_image_count)) or ((not eval_mode) and (j == len(validation_images) - 1)): # Export last image, or all if evaluating.
                            k, ext = (j, 'png') if eval_mode else (n, 'jpg')
                            def save_img(name, img): save_image(img, os.path.join(img_dir, 'img-%07d-%s.%s' % (k, name, ext)), [0.0, 1.0])
                            save_img('a_nsy',  crop_noisy)      # Noisy input
                            save_img('b_out',  crop_mu_x)       # Predicted mean
                            save_img('b_out2', crop_pme)        # Posterior mean estimate (actual output)
                            save_img('b_std',  crop_net_std)    # Predicted std. dev
                            save_img('c_cln',  crop_val_input)  # Clean reference image

                    # Validation pass completed.

                print(", valid_psnr_mu=%8.4f, valid_psnr_pme=%8.4f" % (valid_psnr_mu, valid_psnr_pme), end='')
                eval_last += eval_interval if (n > 0) else 0

            # Exit if evaluation mode.
            if eval_mode:
                print("\nEvaluation done, exiting.")
                print("RESULT %8.4f" % valid_psnr_pme)
                ctx.close()
                return

            # Finish printing.
            autosummary('valid_psnr_mu', valid_psnr_mu)
            autosummary('valid_psnr_pme', valid_psnr_pme)
            dnnlib.tflib.autosummary.save_summaries(summary_log, n)
            print("")

    # Save the result.
    save_snapshot(submit_config, net, 'final-'+config_name)

    # Done.
    summary_log.close()
    ctx.close()


#----------------------------------------------------------------------------
config_lst = [
    dict(eval_id = '00011', noise_style='gauss25', num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00012', noise_style='gauss25', num_iter=2000000, pipeline='n2n'),
    dict(eval_id = '00013', noise_style='gauss25', num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00014', noise_style='gauss25', num_iter=2000000, pipeline='blindspot', noise_params='global'),
    dict(eval_id = '00015', noise_style='gauss25', num_iter=2000000, pipeline='blindspot', noise_params='known',  diagonal_covariance=True),
    dict(eval_id = '00016', noise_style='gauss25', num_iter=2000000, pipeline='blindspot', noise_params='global', diagonal_covariance=True),
    dict(eval_id = '00017', noise_style='gauss25', num_iter=2000000, pipeline='blindspot_mean'),
    dict(eval_id = '00018', noise_style='gauss5_50', num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00019', noise_style='gauss5_50', num_iter=2000000, pipeline='n2n'),
    dict(eval_id = '00020', noise_style='gauss5_50', num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00021', noise_style='gauss5_50', num_iter=2000000, pipeline='blindspot', noise_params='per_image'),
    dict(eval_id = '00022', noise_style='gauss5_50', num_iter=2000000, pipeline='blindspot', noise_params='known',     diagonal_covariance=True),
    dict(eval_id = '00023', noise_style='gauss5_50', num_iter=2000000, pipeline='blindspot', noise_params='per_image', diagonal_covariance=True),
    dict(eval_id = '00024', noise_style='gauss5_50', num_iter=2000000, pipeline='blindspot_mean'),
    dict(eval_id = '00030', noise_style='poisson30', num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00031', noise_style='poisson30', num_iter=2000000, pipeline='n2n'),
    dict(eval_id = '00032', noise_style='poisson30', num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00033', noise_style='poisson30', num_iter=2000000, pipeline='blindspot', noise_params='global'),
    dict(eval_id = '00034', noise_style='poisson30', num_iter=2000000, pipeline='blindspot_mean'),
    dict(eval_id = '00035', noise_style='poisson5_50', num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00036', noise_style='poisson5_50', num_iter=2000000, pipeline='n2n'),
    dict(eval_id = '00037', noise_style='poisson5_50', num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00038', noise_style='poisson5_50', num_iter=2000000, pipeline='blindspot', noise_params='per_image'),
    dict(eval_id = '00039', noise_style='poisson5_50', num_iter=2000000, pipeline='blindspot_mean'),
    dict(eval_id = '00050', noise_style='impulse50', pipeline='n2c', num_iter=16000000),
    dict(eval_id = '00051', noise_style='impulse50', pipeline='n2n', num_iter=16000000),
    dict(eval_id = '00052', noise_style='impulse50', pipeline='blindspot', noise_params='known',  num_iter=4000000),
    dict(eval_id = '00053', noise_style='impulse50', pipeline='blindspot', noise_params='global', num_iter=4000000),
    dict(eval_id = '00054', noise_style='impulse50', pipeline='blindspot_mean', num_iter=8000000),
    dict(eval_id = '00055', noise_style='impulse0_100', pipeline='n2c', num_iter=16000000),
    dict(eval_id = '00056', noise_style='impulse0_100', pipeline='n2n', num_iter=16000000),
    dict(eval_id = '00057', noise_style='impulse0_100', pipeline='blindspot', noise_params='known',     num_iter=4000000),
    dict(eval_id = '00058', noise_style='impulse0_100', pipeline='blindspot', noise_params='per_image', num_iter=4000000),
    dict(eval_id = '00059', noise_style='impulse0_100', pipeline='blindspot_mean',                      num_iter=8000000),
    dict(eval_id = '00180', noise_style='gauss25', num_channels=1, num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00181', noise_style='gauss25', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00182', noise_style='gauss25', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='global'),
    dict(eval_id = '00183', noise_style='gauss5_50', num_channels=1, num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00184', noise_style='gauss5_50', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00185', noise_style='gauss5_50', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='per_image'),
    dict(eval_id = '00188', noise_style='poisson30', num_channels=1, num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00189', noise_style='poisson30', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00190', noise_style='poisson30', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='global'),
    dict(eval_id = '00191', noise_style='poisson5_50', num_channels=1, num_iter=2000000, pipeline='n2c'),
    dict(eval_id = '00192', noise_style='poisson5_50', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='known'),
    dict(eval_id = '00193', noise_style='poisson5_50', num_channels=1, num_iter=2000000, pipeline='blindspot', noise_params='per_image', snapshot_every=100000), # A bit unstable.
]


def make_config_name(c):
    num_channels = c.get('num_channels', 3)
    diag = c.get('diagonal_covariance', False)
    is_blindspot = c['pipeline'] == 'blindspot'
    sigma = '-sigma_'+c['noise_params'] if is_blindspot else ''
    return c['noise_style']+'-'+c['pipeline']+('_diag' if diag else '')+sigma+('-mono' if num_channels == 1 else '')

# ------------------------------------------------------------------------------------------
def cli_examples(configs):
    return '''examples:
  # Train a network with gauss25-blindspot-sigma_global configuration
  python %(prog)s --train=gauss25-blindspot-sigma_global --dataset-dir=$HOME/datasets --validation-set=kodak --train-h5=imagenet_val_raw.h5

  # Evaluate a network using the BSD300 dataset:
  python %(prog)s --eval=$HOME/pretrained/network-00012-gauss25-n2n.pickle --dataset-dir=$HOME/datasets --validation-set=kodak

  List of all configs:

  ''' + '\n  '.join(configs)

def main():
    sc = dnnlib.SubmitConfig()
    sc.run_dir_root = 'results'
    sc.run_dir_ignore += ['datasets', 'results']

    config_map = {}
    selected_config = None
    config_names = []
    for c in config_lst:
        cfg_name = make_config_name(c)
        assert cfg_name not in config_map
        config_map[cfg_name] = c
        config_names.append(cfg_name)

    parser = argparse.ArgumentParser(
        description='Train or evaluate.',
        epilog=cli_examples(config_names),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset-dir', help='Path to validation set data')
    parser.add_argument('--train-h5', help='Specify training set .h5 filename')
    parser.add_argument('--validation-set', help='Evaluation dataset', default='kodak')
    parser.add_argument('--eval', help='Evaluate validation set with the given network pickle')
    parser.add_argument('--train', help='Train for the given config')
    args = parser.parse_args()

    eval_sets = {
        'kodak':  dict(validation_repeats=10),
        'bsd300': dict(validation_repeats=3),
        'set14':  dict(validation_repeats=20)
    }
    if args.validation_set not in eval_sets:
        print ('Validation set specified with --validation-set not in one of: ' + ', '.join(eval_sets))
        sys.exit(1)

    if args.dataset_dir is None:
        print ('Must specify validation dataset path with --dataset-dir')
        sys.exit(1)
    if not os.path.isdir(args.dataset_dir):
        print ('Directory specified with --dataset-dir does not seem to exist.')
        sys.exit(1)

    config_name = None
    if args.train:
        if args.eval is not None:
            print ('Use either --train or --eval')
            sys.exit(1)
        if args.train_h5 is None:
            print ('Must specify training dataset with --train-h5 when training')
            sys.exit(1)
        config_name = args.train
    elif args.eval:
        pickle_name = args.eval
        pickle_re = re.compile('^network-(?:[0-9]+|final)-(.+)\\.pickle')
        m = pickle_re.match(os.path.basename(pickle_name))
        if m is None:
            print ('network pickle name must contain network config string')
            sys.exit(1)
        config_name = m.group(1)
    else:
        print ('Must use either --train or --eval')
        sys.exit(1)


    if config_name not in config_map:
        print ('unknown config', config_name)
        sys.exit(1)

    validation_repeats = eval_sets[args.validation_set]['validation_repeats'] if args.eval else 1

    # Common configuration for all runs.
    config = dnnlib.EasyDict(
        train_dataset       = args.train_h5,            # Training set.
        validation_dataset  = args.validation_set,      # Dataset used to monitor validation convergence during training.
        validation_repeats  = validation_repeats,
        num_channels        = 3,                        # RGB.
        train_resolution    = 256,
        minibatch_size      = 4,
        learning_rate       = 3e-4,
        config_name         = config_name,
        dataset_dir         = args.dataset_dir
    )

    selected_config = config_map[config_name]
    config.update(**selected_config)
    if args.eval is not None:
        config['eval_network'] = args.eval
    del config['eval_id']

    #----------------------------------------------------------------------------

    # Execute.
    sc.run_desc = 'eval' if config.get('eval_network') else 'train'

    # Decorate run_desc.
    sc.run_desc += '-ilsvrc'
    if config.get('prune_dataset'): sc.run_desc += '_%d' % config.prune_dataset
    sc.run_desc += '-%s' % config.validation_dataset
    sc.run_desc += '-%dc' % config.num_channels
    sc.run_desc += '-%s' % config.noise_style
    if config.minibatch_size != 4:      sc.run_desc += '-mb%d' % config.minibatch_size
    if config.learning_rate != 3e-4:    sc.run_desc += '-lr%g' % config.learning_rate
    if config.num_iter >= 1000000:
        sc.run_desc += '-iter%dm' % (config.num_iter // 1000000)
    elif config.num_iter >= 1000:
        sc.run_desc += '-iter%dk' % (config.num_iter // 1000)
    else:
        sc.run_desc += '-iter%d' % config.num_iter
    sc.run_desc += '-%s' % config.pipeline
    if config.get('diagonal_covariance'): sc.run_desc += 'Diag'
    if config.pipeline == 'blindspot':
        sc.run_desc += '-%s' % config.noise_params
    if config.train_resolution != 256:  sc.run_desc += '-res%d' % config.train_resolution

    if config.get('eval_network'): sc.run_desc += '-EVAL_%s' % config_name
    if config.get('eval_network'): sc.run_dir_root += '/_eval'

    # Submit.
    submit.submit_run(sc, 'selfsupervised_denoising.train', **config)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
