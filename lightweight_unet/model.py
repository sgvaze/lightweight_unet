import os; os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, concatenate, \
    SeparableConv2D
from keras.regularizers import l2
from keras.models import Model
from lightweight_unet.utils import get_distillation_indices
from keras.utils import plot_model

class Models(object):

    """
    Class builds U-Net models, including:
        Regular UNet (Ronneberger et al.)
        Thin U-Net (few feature channels per layer, either regular or separable conv)
        Distillation U-Net (model adapted for knowledge distillation)
    """

    def __init__(self, model_cfg = None):

        self.model_cfg = model_cfg

        self.input = Input(self.model_cfg['input_dims'] + (1,))
        self.input_dropout = Dropout(self.model_cfg['dropout_input'])(self.input)

        self.teacher_dir = None


    def block(self, input, from_skip = None, up_or_down = False, num_filters = 32, separable = False, lowest = False, extra_convs = None, output_block = False):
        """

        Basic block function.
        Constructs a single down-sampling or up-sampling block
        (Maybe) Merge -> Conv -> Conv -> (Maybe) Conv  -> BatchNorm -> (Maybe) Dropout -> Down/Upsampling

        :param input: Input tensor
        :param from_skip: Tensor to concatenate from possible skip connection
        :param up_or_down: Bool: 1 if downsampling block, 0 if upsampling block
        :param num_filters: Number of filters in Conv layers
        :param separable: Bool: 1 if separable convolutional block, 0 if regular convolutional block
        :param lowest: Bool: 1 if this is the lowest (most downsampled) block in the network
        :param extra_convs: Int: Number of extra Conv layers in block (if None, function defautls to two Conv layers)
        :param output_block: Bool: 1 if this the output block (resulting in prediction)
        :return: Output Tensor

        """
        if not separable:
            conv_layer = Conv2D
        else:
            conv_layer = SeparableConv2D

        #Concatenate outputs from previous block and skip connection if this is an upsampling block
        if not up_or_down and from_skip is not None:
            merge = concatenate([from_skip, input], axis=3)
            conv1 = conv_layer(num_filters, self.model_cfg['filter_size'], padding="same",
                             activation="relu", kernel_regularizer = l2(self.model_cfg['l2_reg']))(merge)
        else:
            conv1 = conv_layer(num_filters, self.model_cfg['filter_size'], padding="same",
                             activation="relu", kernel_regularizer = l2(self.model_cfg['l2_reg']))(input)


        conv2 = conv_layer(num_filters, self.model_cfg['filter_size'], padding="same",
                             activation="relu", kernel_regularizer = l2(self.model_cfg['l2_reg']))(conv1)

        if extra_convs is not None:

            for count in range(extra_convs):
                conv2 = conv_layer(num_filters, self.model_cfg['filter_size'], padding="same",
                             activation="relu", kernel_regularizer = l2(self.model_cfg['l2_reg']))(conv2)

        norm = BatchNormalization()(conv2)

        if output_block:
            out = Conv2D(1, 1, padding="same", activation="sigmoid", name="out", kernel_regularizer=l2(self.model_cfg['l2_reg']))(norm)
            return out

        if lowest: # Add dropout if this is the most downsampled block of the network
            drop = Dropout(self.model_cfg['dropout_lowest'])(norm)
            out = UpSampling2D()(drop)
            return out

        if up_or_down:
            out = MaxPooling2D()(norm)
            return out, norm
        else:
            out = UpSampling2D()(norm)
            return out


    def get_original_uNet(self):

        """
        # Builds the original UNet (as proposed by Ronneberger et al.)
        :returns UNet model
        """

        x = self.input_dropout
        to_skip = [] # Maintain a list of layers for concatenation through skip connections

        for i in range(4):
            x, to_skip_layer = self.block(x, up_or_down=1, num_filters=64*(2**(i)))
            to_skip.append(to_skip_layer)
        to_skip.append(None)

        for i in reversed(range(4)):

            if i == 3: # Check if this is the most downsampled block
                lowest = True
            else:
                lowest = False

            x = self.block(x, from_skip= to_skip[i+1], num_filters=64 *(2**(i+1)), lowest = lowest)
            x = Conv2D(filters=64 *(2**(i)), kernel_size=2, padding="same", activation="relu",
                       kernel_regularizer=l2(self.model_cfg['l2_reg']))(x)

        x = self.block(x, from_skip= to_skip[0], num_filters=64, output_block=True)

        return Model(input = self.input, output = x)


    def get_thin_uNet(self, separable = None, num_blocks = None):

        """
        Builds a thin uNet with only model_cfg['num_filters_thin'] feature channels per layer. Up convolutions are removed
        Also allows adjustment for number of blocks in donwsampling and upsampling paths

        :param separable: Bool: False if regular conv Unet, True if purely separable convolution UNet
        :param num_blocks: Number of donwsampling blocks in network
        :return: Thin Unet model
        """

        if num_blocks is None:
            num_blocks = self.model_cfg['num_blocks']
        if separable is None:
            separable = self.model_cfg['separable']

        num_filters = self.model_cfg['num_filters_thin']
        x = self.input_dropout
        to_skip = []

        for i in range(num_blocks):
            x, to_skip_layer = self.block(x, up_or_down= 1, num_filters=num_filters, separable= separable)
            to_skip.append(to_skip_layer)
        to_skip.append(None)

        for i in reversed(range(num_blocks)):

            if i == num_blocks - 1: # Check if this is the most downsampled block
                lowest = True
            else:
                lowest = False

            x = self.block(x, from_skip= to_skip[i+1], num_filters=num_filters, lowest = lowest, separable= separable)

        x = self.block(x, from_skip= to_skip[0], num_filters=32, output_block=True, separable= separable)

        return Model(input = self.input, output = x)

    def get_distillation_uNet(self, num_blocks = None):

        """
        Builds a model to be trainined via knowledge distillation
        Expects a teacher network with the same number of downsampling blocks
        Contains function 'get_distillation_indices' which dictates which layers are supervised. By default final Conv
        layer in each block is disillation supervised

        :param num_blocks: Number of blocks in downsampling path
        :return: Student model, Teacher model

        """

        if num_blocks is None:
            num_blocks = self.model_cfg['num_blocks']

        num_filters = self.model_cfg['num_filters_thin']
        teacher_model_base = self.get_thin_uNet(num_blocks=num_blocks, separable=False)

        teacher_outputs = [teacher_model_base.layers[i].output for i in get_distillation_indices(teacher_model_base)]
        teacher_model = Model(inputs=teacher_model_base.input, outputs=[*teacher_outputs, teacher_model_base.output])

        x = self.input_dropout
        to_skip = []

        for i in range(num_blocks):
            if i == 0:
                x, to_skip_layer = self.block(x, up_or_down=1, num_filters=num_filters, separable=True)
            elif i - num_blocks + 1 == 0:
                x, to_skip_layer = self.block(x, up_or_down=1, num_filters=num_filters, extra_convs = 1, separable=True)
            else:
                x, to_skip_layer = self.block(x, up_or_down=1, num_filters=num_filters, separable=True)
            to_skip.append(to_skip_layer)

        to_skip.append(None)

        for i in reversed(range(num_blocks)):

            if i == num_blocks - 1:  # Check if this is the most downsampled block
                lowest = True
                x = self.block(x, from_skip=to_skip[i + 1], num_filters=num_filters, extra_convs=0, lowest=lowest, separable=True)
            elif i == num_blocks - 2:
                lowest = False
                x = self.block(x, from_skip=to_skip[i + 1], num_filters=num_filters, extra_convs=0, lowest=lowest, separable=True)
            else:
                x = self.block(x, from_skip=to_skip[i + 1], num_filters=num_filters, lowest=lowest, separable=True)

        x = self.block(x, from_skip=to_skip[0], num_filters=32, output_block=True, separable=True)

        student_model_base = Model(input = self.input, output = x)

        # Get the the student output layers
        student_outputs = [student_model_base.layers[i].output for i in get_distillation_indices(student_model_base)]
        student_outputs.append(student_model_base.output)

        # Create a student model to be optimized with distillation
        student_model = Model(inputs=student_model_base.input, outputs=student_outputs)

        return student_model, teacher_model