import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Layer

# define merge layer
class Merge_Layer(Layer):
    def __init__(self, **kwargs):
        super(Merge_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.para1 = self.add_weight(shape=(input_shape[0][1], input_shape[0][2]),
                                     initializer='uniform', trainable=True,
                                     name='para1')
        self.para2 = self.add_weight(shape=(input_shape[1][1], input_shape[1][2]),
                                     initializer='uniform', trainable=True,
                                     name='para2')
        super(Merge_Layer, self).build(input_shape)

    def call(self, inputs):
        mat1 = inputs[0]
        mat2 = inputs[1]
        output = mat1 * self.para1 + mat2 * self.para2
        # output = mat1 * 0.1 + mat2 * 0.9
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def build_model(input_data, input_HA):
    _, k, t_input, num_links = input_data.shape
    _, _, t_pre = input_HA.shape
    x = keras.layers.BatchNormalization(input_shape =(k,t_input,num_links))(input_data)

    x = keras.layers.Conv2D(
                               filters = num_links,
                               kernel_size = 3,
                               strides = 1,
                               padding="SAME",
                               activation='relu')(x)

    x = keras.layers.AveragePooling2D(pool_size = (2,2),
                                    strides = 1,
                                    padding = "SAME",
                                    )(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
                           filters = num_links,
                           kernel_size = 3,
                           strides = 1,
                           padding="SAME",
                           activation='relu')(x)

    x = keras.layers.AveragePooling2D(pool_size = (2,2),
                                    strides = 1,
                                    padding = "SAME",
                                    )(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(num_links*2*t_pre, activation='relu', name='dense_1')(x)
    x = keras.layers.Dense(num_links*t_pre, activation='relu', name='dense_2')(x)

    output = keras.layers.Reshape((num_links,t_pre))(x)

    output_final = Merge_Layer()([output, input_HA])

    # construct model
    finish_model = keras.models.Model([input_data,input_HA], [output_final])

    finish_model.summary()
    return finish_model