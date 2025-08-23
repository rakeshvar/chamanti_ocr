import pickle
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Reshape, Dense, \
    Bidirectional, LSTM, SimpleRNN, GRU

from banti2chamanti import banti2chamanti
from utils import rnn, write_dict


class CRNNReshape(Layer):
    """
    Custom layer to reshape (B, h, w, C) to (B, w, C*h) for CRNN models.
     to convert CNN feature maps to RNN sequence input.
    """
    def __init__(self, **kwargs):
        super(CRNNReshape, self).__init__(**kwargs)

    def call(self, inputs):
        B, h, w, c = inputs.shape
        x = tf.transpose(inputs, perm=[0, 2, 1, 3])  # (B, w, h, C)
        return tf.reshape(x, [B, w, h*c])

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        return (batch, width, height * channels if height and channels else None)

    def get_config(self):
        config = super(CRNNReshape, self).get_config()
        return config

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
    def __init__(self, xy_info, layer_args, list_of_weights=None):
        # Information about the input image and the output labels
        batch_size = xy_info["batch_size"]
        slab_max_wd = xy_info["slab_max_wd"]
        slab_ht = xy_info["slab_ht"]
        labels_max_len = xy_info["labels_max_len"]
        alphabet_size = xy_info["alphabet_size"]

        # Inputs to the Neural Network
        image = Input(name="image", batch_shape=(batch_size, slab_ht, slab_max_wd, 1), dtype="float32")
        labeling = Input(name="labeling", batch_shape=(batch_size, labels_max_len,), dtype="int64")
        image_width = Input(name='image_width', batch_shape=(batch_size, 1,), dtype='int64')
        labeling_length = Input(name='labeling_length', batch_shape=(batch_size, 1,), dtype='int64')

        i, height_down, width_down, num_filters = 0, 1, 1, 1
        layers = []

        # Add conv pool layers
        for largs in layer_args:
            name = largs['name']
            if name.lower().startswith("conv"):
                num_filters = largs['filters']
                layer = Conv2D(**largs)
            elif name.lower().startswith("pool"):
                pool_size = largs["pool_size"]
                height_down *= pool_size[0]
                width_down *= pool_size[1]
                layer = MaxPooling2D(**largs)
            else:
                break
            layers.append(layer)
            i += 1

        # Flatten 3D maps to 2D
        # (B, H//h_down, W//w_down, C) -> (B, W//w_down, C*H//h_down)
        # new_shape = slab_max_wd // width_down, (slab_ht // height_down) * num_filters
        layers.append(CRNNReshape(name="reshape"))

        # Add dense layers
        while layer_args[i]['name'].lower().startswith('den'):
            layers.append(Dense(**layer_args[i]))
            i += 1

        # Add recurrent layers
        for largs in layer_args[i:]:
            name = largs['name']
            if name.lower().startswith('lstm'):
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
        out = image
        for layer_ in layers:
            out = layer_(out)
        out = CTCLayer("ctc_loss", width_down)(labeling, out, image_width, labeling_length)

        # Compile a model
        model_inputs = [image, labeling, image_width, labeling_length]
        model = keras.models.Model(inputs=model_inputs, outputs=out, name="crnn_ctc_model")
        if list_of_weights is not None:
            model.set_weights(list_of_weights)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        model.width_scaled_down_by = width_down

        # Simplify layer_args to be picklable
        for largs in layer_args:
            largs.pop("weights", None)
            if 'activation' in largs:
                if type(largs['activation']) != str:
                    largs['activation'] = largs['activation'].name
            write_dict(largs)

        # Store all in self
        self.model = model
        self.xy_info = xy_info
        self.layer_args = layer_args
        self.layers = layers

    def save_model_specs_weights(self, filename):
        if filename[-4:] != '.pkl':
            filename += '.pkl'

        info = self.xy_info, self.layer_args, self.model.get_weights()
        with open(filename, "wb") as f:
            pickle.dump(info, f, 3)
        print("Saved ", filename)

    @classmethod
    def from_chamanti(cls, xy_info, pkl_file_name):
        with open(pkl_file_name, "rb") as f:
            xy_info_read, layer_args, list_of_weights = pickle.load(f)
        for k in xy_info_read:
            assert xy_info[k] == xy_info_read[k], f"Could not match Key '{k}' : {xy_info[k]} != {xy_info_read[k]}"
        print("Initializng from Chamanti model at ", pkl_file_name)
        for i, (l, w) in enumerate(zip(layer_args, list_of_weights)):
            print("Layer ", i, " ", l, w.shape )
        return cls(xy_info, layer_args, list_of_weights)

    @classmethod
    def from_banti(cls, xy_info, pkl_file_name, rnn_args=None, *args, **kwargs):
        layer_args = banti2chamanti(pkl_file_name, *args, **kwargs)
        if type(rnn_args) is str:
            name = rnn_args.rstrip('0123456789')
            assert name in ('lstm', 'gru', 'rnn')
            try:
                nunits = int(rnn_args[len(name):])
            except ValueError:
                nunits = layer_args[-1]['nunits']//2
            layer_args.append(rnn(name, nunits))
        elif type(rnn_args) is list:
            layer_args += rnn_args
        return cls(xy_info, layer_args)
