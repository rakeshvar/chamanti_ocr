import os.path
import pickle

from tensorflow import keras
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Reshape, Dense, \
    Bidirectional, LSTM, SimpleRNN, GRU


class CTCLayer(Layer):
    """ You can directly add the loss to the model. But having this class makes the model summary look good. """
    def __init__(self, name, width_down):
        super().__init__(name=name)
        self.width_down = width_down

    def call(self, labels, excitations, image_width, labels_length):
        image_width //= self.width_down
        self.add_loss(keras.backend.ctc_batch_cost(labels, excitations, image_width, labels_length))
        return excitations


class ModelBuilder:
    def __init__(self, layer_args, xy_info, list_of_weights=None):
        # Information about the input image and the output labels
        batch_size = xy_info["batch_size"]
        slab_max_wd = xy_info["slab_max_wd"]
        slab_ht = xy_info["slab_ht"]
        labels_max_len = xy_info["labels_max_len"]
        alphabet_size = xy_info["alphabet_size"]

        # Inputs to the Neural Network
        image = Input(name="image", batch_shape=(batch_size, slab_max_wd, slab_ht, 1), dtype="float32")
        labeling = Input(name="labeling", batch_shape=(batch_size, labels_max_len,), dtype="int64")
        image_width = Input(name='image_width', batch_shape=(batch_size, 1,), dtype='int64')
        labeling_length = Input(name='labeling_length', batch_shape=(batch_size, 1,), dtype='int64')

        i, width_down, height_down, num_filters = 0, 1, 1, 1
        layers = []

        # Add conv pool layers
        for largs in layer_args:
            name = largs['name']
            if name.lower().startswith("conv"):
                num_filters = largs['filters']
                layer = Conv2D(**largs)
            elif name.lower().startswith("pool"):
                pool_size = largs["pool_size"]
                width_down *= pool_size[0]
                height_down *= pool_size[1]
                layer = MaxPooling2D(**largs)
            else:
                break
            layers.append(layer)
            i += 1

        # Flatten 3D maps to 2D
        new_shape = slab_max_wd // width_down, (slab_ht // height_down) * num_filters
        layers.append(Reshape(target_shape=new_shape, name="reshape"))

        # Add dense layers and lstm layers
        for largs in layer_args[i:]:
            name = largs['name']
            if name.lower().startswith('den'):
                layer = Dense(**largs)
            elif name.lower().startswith('lstm'):
                layer = Bidirectional(LSTM(**largs, return_sequences=True, stateful=True))
            elif name.lower().startswith('rnn'):
                layer = Bidirectional(SimpleRNN(**largs, return_sequences=True, stateful=True))
            elif name.lower().startswith('gru'):
                layer = Bidirectional(GRU(**largs, return_sequences=True, stateful=True))
            else:
                raise ValueError(f"Did not understand layer args: {largs}")
            layers.append(layer)

        # Add final probability softmax layer
        layers.append(Dense(alphabet_size + 1, activation="softmax", name="output"))

        # Build the network and add loss
        x = image
        for layer in layers:
            x = layer(x)
        x = CTCLayer("ctc_loss", width_down)(labeling, x, image_width, labeling_length)

        # Compile a model
        model_inputs = [image, labeling, image_width, labeling_length]
        model = keras.models.Model(inputs=model_inputs, outputs=x, name="crnn_ctc_model")
        if list_of_weights is not None:
            model.set_weights()

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        model.width_scaled_down_by = width_down

        # Store all in self
        for la in layer_args:
            la.pop("weights", None)
            if 'activation' in la and type(la['activation']) != str:
                la['activation'] = la['activation'].name

        self.model = model
        self.xy_info = xy_info
        self.layer_args = layer_args
        self.layers = layers

    def save_model_specs_weights(self, filename):
        if filename[-4:] != '.pkl':
            filename += '.pkl'
        filename = os.path.basename(filename)

        info = self.xy_info, self.layer_args, self.model.get_weights()
        with open(filename, "wb") as f:
            pickle.dump(info, f, 3)
        print("Saved ", filename)


def model_builder_from_chamanti(pkl_file_name):
    with open(pkl_file_name, "rb") as f:
        xy_info, layer_args, list_of_weights = pickle.load(f)
    return ModelBuilder(xy_info, layer_args, list_of_weights)
