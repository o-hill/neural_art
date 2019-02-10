'''

    Loss Functions and corresponding utilities for training the VGG network.

'''

import numpy as np
from keras import backend as K

from ipdb import set_trace as debug


def content_loss(original, generated) -> float:
    '''Content loss as described in the paper.

        @original: the tensor corresponding to the responses in the filters for the original image.
        @generated: same as original, but for the white noise image.

        Returns the squared error loss between the two.
    '''

    return 0.5 * K.sum(K.square(generated - original))


def gram_matrix(responses):
    '''Computes the Gram matrix for the given filter responses.

        @responses: the filter responses for the current image.
            Size: (Number of Layers) x (Number of Filters in Layer i) x (Size of Filter j).
    '''

    assert K.ndim(responses) == 3

    if K.image_data_format() == 'channels_first':
        F = K.batch_flatten(responses)
    else:
        F = K.batch_flatten(K.permute_dimensions(responses, (2, 0, 1)))

    # Now return the Gram matrix.
    return K.dot(F, K.transpose(F))


def style_loss(original, generated, n_rows, n_cols) -> float:
    '''Style loss as described by the paper.

        @original: tensor of the responses in the filters for the original image.
        @generated: same as original, but for the white noise image.

        Return the MSE between the two.
    '''

    assert K.ndim(original) == 3 and K.ndim(generated) == 3
    channels = 3

    # Find the coefficients for the loss.
    Nl = (n_rows * n_cols) ** 2
    Ml = channels ** 2

    return K.sum(K.square(gram_matrix(generated) - gram_matrix(original))) / (4.0 * Nl * Ml)


def total_variation_loss(x, img_nrows: int = 0, img_ncols: int = 0) -> float:
    '''Return the total variational loss to preserve spatial coherency.'''
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))






