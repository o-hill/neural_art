'''

    Keras implementation of style transfer.

    Based off of 'A Neural Algorithm for Artistic Style'.

'''

import numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img, save_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from keras.applications import vgg19
import os
from tqdm import tqdm

from keras.utils import plot_model
from k_loss import *


# Define some constants.
FILE_PATH = './images'
STYLE_PATH = f'{FILE_PATH}/style'
CONTENT_PATH = f'{FILE_PATH}/content'
WRITE_PATH = f'{FILE_PATH}/results'

# Layers we want to use to extract filter data from.
feature_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
]

# The layer we want to use to extract the content information from.
content_layer = 'block5_conv2'

alpha = 0.025   # Content loss weight.
beta = 1.0      # Style loss weight.
gamma = 1.0     # Total variation loss weight.


class StyleTransfer:

    def __init__(self, iterations: int = 10) -> None:
        self.base_path = FILE_PATH
        self.style_path = STYLE_PATH
        self.content_path = CONTENT_PATH
        self.write_path = WRITE_PATH
        self.iterations = iterations

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.content = ''
        self.style = ''

        self.rows = 0
        self.cols = 0

        self.evaluator = None
        self.loss = None
        self.gradients = None


    def create_image(self, style: str = '', content: str = '') -> None:
        '''Create an image using style transfer.

            @style: a filename of the image to use for style.
            @content: filename of the image to use for content.

            Writes a new image to the write path.
        '''

        self.style = f'{self.style_path}/{style}'
        self.content = f'{self.content_path}/{content}'

        # Find the dimensions of the generated image.
        width, height = load_img(self.content).size
        self.rows = 400
        self.cols = int(width * self.rows / height)
        print(f'Creating images of size ({self.rows}, {self.cols})')

        base, style, generated, input_tensor = self.create_tensors()

        # Load the network each time due to the variable size of images.
        network = vgg19.VGG19(input_tensor = input_tensor,
                    weights = 'imagenet',
                    include_top = False)

        outputs = { layer.name: layer.output for layer in network.layers }

        # Get the loss and the gradient of the generated image with respect to the loss.
        self.loss_tensor(outputs, generated)
        self.gradients = K.gradients(self.loss, generated)

        # Define the output of the network and how to get the gradients and loss.
        network_output = self.network_outputs(generated)
        self.evaluator = Evaluator(network_output, self.rows, self.cols)

        image = self.get_image(self.content)
        print('Generated all data succesfully, starting to run...')

        plot_model(network, show_shapes=True)
        self._run(image, content)


    def _run(self, image, image_filename: str = '') -> None:
        '''Run the style transfer algorithm on the image.

            Writes images to the write path.
        '''

        write_path = f'{self.write_path}/{image_filename.split(".")[0]}'
        file_type = image_filename.split('.')[-1]
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        for iteration in tqdm(range(self.iterations)):

            image, loss, info = fmin_l_bfgs_b(
                    self.evaluator.loss_eval,
                    image.flatten(),
                    fprime = self.evaluator.gradients_eval,
                    maxfun = 20)

            print(f'Loss at iteration {iteration}: {loss}')
            save_img(f'{write_path}/iteration_{iteration}.{file_type}',
                    self.deprocess_image(image.copy()))

            print(f'Image saved at iteration {iteration}')



    def get_image(self, img_path: str = '') -> K.variable:
        '''Load an image.'''
        img = img_to_array(load_img(img_path, target_size=(self.rows, self.cols)))
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img


    def network_outputs(self, generated_t):
        '''Return a list of the outputs of the network.'''
        out = [self.loss]

        if isinstance(self.gradients, (list, tuple)):
            out += self.gradients
        else:
            out.append(self.gradients)

        return K.function([generated_t], out)


    def loss_tensor(self, outputs: dict = { }, generated_t = None) -> K.variable:
        '''Define and return the Keras tensor for the total loss.

            @generated_t: Keras tensor for the generated image.
        '''

        self.loss = K.variable(0.0)

        layer_features = outputs[content_layer]
        original = layer_features[0, :, : ,:]
        generated = layer_features[2, :, :, :]

        self.loss += alpha * content_loss(original, generated)

        style_weight = beta / len(feature_layers)
        for layer in feature_layers:
            layer_features = outputs[layer]
            style = layer_features[1, :, :, :]
            generated = layer_features[2, :, :, :]

            self.loss += style_weight * style_loss(style, generated, self.rows, self.cols)

        self.loss += gamma * total_variation_loss(generated_t, self.rows, self.cols)


    def create_tensors(self) -> tuple:
        '''Create the tensors that will be used for the Keras network.'''
        base = K.variable(self.get_image(self.content))
        style = K.variable(self.get_image(self.style))

        if K.image_data_format() == 'channels_first':
            generated = K.placeholder((1, 3, self.rows, self.cols))
        else:
            generated = K.placeholder((1, self.rows, self.cols, 3))

        input_tensor = K.concatenate([base, style, generated], axis=0)

        return base, style, generated, input_tensor



    def deprocess_image(self, x) -> np.ndarray:
        '''Converts an image from a Keras tensor to a viewable format. From Keras docs.'''
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, self.rows, self.cols))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((img_nrows, img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x



class Evaluator:

    def __init__(self, network, rows: int = 0, cols: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.network = network

        self.loss = None
        self.gradient = None


    def eval_loss_and_grads(self, x) -> None:
        '''Evaluate the loss and gradient of the image. From the Keras docs.'''

        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, self.rows, self.cols))
        else:
            x = x.reshape((1, self.rows, self.cols, 3))

        outs = self.network([x])
        print('found gradients')
        self.loss = outs[0]

        if len(outs[1:]) == 1:
            self.gradient = outs[1].flatten().astype('float64')
        else:
            self.gradient = np.array(outs[1:]).flatten().astype('float64')

        print('Found loss and gradient...')


    def loss_eval(self, x) -> float:
        '''Evaluate the loss and gradient and store them.'''

        assert self.loss == None

        self.eval_loss_and_grads(x)
        return self.loss


    def gradients_eval(self, x):
        '''Return the gradient.'''

        assert self.loss != None

        gradient = np.copy(self.gradient)
        self.loss = None
        self.gradient = None

        return gradient


if __name__ == '__main__':

    content = 'small_tuebingen.jpg'
    style = 'small_starry_night.jpg'

    transfer = StyleTransfer()
    transfer.create_image(style, content)



















