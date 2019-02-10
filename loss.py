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

    # Find the shape of the 3D matrix.
    shape = K.shape(responses)

    # Reshape the filters into the F matrix: (width*height, num_filters)
    F = K.reshape(responses, (shape[0] * shape[1], shape[2]))

    debug()

    # Now return the Gram matrix.
    return K.dot(F, K.transpose(F))


def style_loss(original, generated) -> float:
    '''Style loss as described by the paper.

        @original: tensor of the responses in the filters for the original image.
        @generated: same as original, but for the white noise image.

        Return the MSE between the two.
    '''

    shape = K.shape(original)

    # Find the coefficients for the loss.
    Nl = (shape[0] * shape[1]) ** 2
    Ml = shape[2] ** 2

    factor = 1 / (4 * Nl * Ml)
    return factor * K.sum(K.square(gram_matrix(generated) - gram_matrix(original)))


def total_variation_loss(image):
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






