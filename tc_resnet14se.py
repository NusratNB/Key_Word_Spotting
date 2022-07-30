from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, ReLU, BatchNormalization, Add, AveragePooling1D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, Reshape, Dense, multiply, add, Permute, Conv2D, Concatenate
from tensorflow.keras import backend as K


def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, filters)

    se = GlobalAveragePooling1D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def get_residual_block_type_one_se(input_tensor, c, k, kernel_size, se=False):
    multiplied_size = int(c * k)
    x = Conv1D(multiplied_size, kernel_size, strides=1,
               use_bias=False, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(multiplied_size, kernel_size, strides=1,
               use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    if se:
        x = squeeze_excite_block(x)
    x = Add()([x, input_tensor])
    x = ReLU()(x)
    
    return x

def _tensor_shape(tensor):
    TF = False
    return getattr(tensor, '_shape_val') if TF else getattr(tensor, 'shape')

def get_residual_block_type_two_se(input_tensor, c, k, kernel_size, se=False):
    multiplied_size = int(c * k)
    x1 = Conv1D(multiplied_size, kernel_size, strides=2,
                use_bias=False, padding='same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv1D(multiplied_size, kernel_size, strides=1,
                use_bias=False, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    if se:
        x1 = squeeze_excite_block(x1)

    x2 = Conv1D(multiplied_size, 1, strides=2,
                use_bias=False, padding='same')(input_tensor)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    x = Add()([x1, x2])
    x = ReLU()(x)
    return x


def get_residual_block_type_one(input_tensor, c, k):
    multiplied_size = int(c * k)
    x = Conv1D(multiplied_size, 9, strides=1,
               use_bias=False, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(multiplied_size, 9, strides=1,
               use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    x = ReLU()(x)
    return x


def get_residual_block_type_two(input_tensor, c, k):
    multiplied_size = int(c * k)
    x1 = Conv1D(multiplied_size, 9, strides=2,
                use_bias=False, padding='same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv1D(multiplied_size, 9, strides=1,
                use_bias=False, padding='same')(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv1D(multiplied_size, 1, strides=2,
                use_bias=False, padding='same')(input_tensor)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    x = Add()([x1, x2])
    x = ReLU()(x)
    return x


DROPOUT_RATE = 0.5

def get_tc_recnet_14_se(input_shape, num_classes, k):
    input_layer = input_shape# Input(input_shape)
    x = Conv1D(int(16 * k), 3, strides=1, use_bias=False,
               padding='same')(input_layer)
    
    x = get_residual_block_type_two_se(x, 24, k, kernel_size=9, se=True)
    x = get_residual_block_type_one_se(x, 24, k, kernel_size=9, se=True)
    x = get_residual_block_type_two_se(x, 32, k, kernel_size=9, se=True)
    x = get_residual_block_type_one_se(x, 32, k, kernel_size=9, se=True)
    x = get_residual_block_type_two_se(x, 48, k, kernel_size=9, se=True)
    x = get_residual_block_type_one_se(x, 48, k, kernel_size=9, se=True)
    x = AveragePooling1D(3, 1)(x)
    x = Flatten()(x)
    x = Dropout(DROPOUT_RATE)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return output_layer


def get_tc_resnet_14(input_shape, num_classes, k):
    input_layer = Input(input_shape)
    x = Conv1D(int(16 * k), 3, strides=1, use_bias=False,
               padding='same')(input_layer)
    x = get_residual_block_type_two(x, 24, k)
    x = get_residual_block_type_one(x, 24, k)
    x = get_residual_block_type_two(x, 32, k)
    x = get_residual_block_type_one(x, 32, k)
    x = get_residual_block_type_two(x, 48, k)
    x = get_residual_block_type_one(x, 48, k)
    x = AveragePooling1D(3, 1)(x)
    x = Flatten()(x)
    x = Dropout(DROPOUT_RATE)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)
