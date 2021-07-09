from tensorflow import keras
from tensorflow.keras import layers

from model_specs import actv
from utils import couple


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, image_widths, labeling_lengths):
        loss = self.loss_fn(y_true, y_pred, image_widths, labeling_lengths)
        self.add_loss(loss)
        return y_pred


def conv_pool(layer_specs, inputs, weights=None):
    if weights is None:
        weights = [None for _ in layer_specs]

    x = inputs
    width_down, height_down, filters = 1, 1, 1
    for spec, lwts in zip(layer_specs, weights):
        name = spec['name']
        if name.lower().startswith("conv"):
            filters = spec.pop('filters')
            kernel_size = spec.pop('kernel_size')
            x = layers.Conv2D(filters, kernel_size, **spec)(x)
        elif name.lower().startswith("pool"):
            pool_size = couple(spec.pop("pool_size"))
            x = layers.MaxPooling2D(pool_size, **spec)(x)
            width_down *= pool_size[0]
            height_down *= pool_size[1]

    return x, width_down, height_down, filters


def build_model(num_chars,
                img_height, img_width_max, batch_size,
                max_labelings_len,
                num_dense,
                num_lstm_out,
                conv_pool_specs):
    image = layers.Input(name="image", batch_shape=(batch_size, img_width_max, img_height, 1), dtype="float64")
    labeling = layers.Input(name="labeling", batch_shape=(batch_size, max_labelings_len,), dtype="int64")
    image_width = layers.Input(name='image_width', batch_shape=(batch_size, 1,), dtype='int64')
    labeling_length = layers.Input(name='labeling_length', batch_shape=(batch_size, 1,), dtype='int64')

    x, width_down, height_down, num_kernels = conv_pool(conv_pool_specs, image)
    img_width_max = img_width_max // width_down
    img_height = img_height // height_down
    new_shape = img_width_max, img_height * num_kernels
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    if num_dense > 0:
        x = layers.Dense(num_dense, activation=actv("relu05"), name="dense1")(x)
    x = layers.Bidirectional(layers.LSTM(num_lstm_out, return_sequences=True, stateful=True))(x)
    x = layers.Dense(num_chars + 1, activation="softmax", name="softmax")(x)
    output = CTCLayer(name="ctc_loss")(labeling, x, image_width//width_down, labeling_length)

    model_inputs = [image, labeling, image_width, labeling_length]
    model = keras.models.Model(inputs=model_inputs, outputs=output, name="ocr_model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    model.width_scaled_down_by = width_down
    return model