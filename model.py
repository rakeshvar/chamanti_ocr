from tensorflow import keras
from tensorflow.keras import layers


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def build_model(num_chars,
                img_height, img_width,
                labelslen,
                num_dense_features,
                num_out,
                layerspecs):
    input_img = layers.Input(name="image", shape=(img_width, img_height, 1), dtype="float64")
    labels    = layers.Input(name="label", shape=(labelslen,), dtype="int64")
    input_length = layers.Input(name='input_length', shape=(1,), dtype='int64')
    label_length = layers.Input(name='label_length', shape=(1,), dtype='int64')

    x = input_img
    width_down, height_down, num_kernels = 1, 1, 1
    for lyr in layerspecs:
        if lyr[0].startswith('conv'):
            x = layers.Conv2D(lyr[1], lyr[2], activation="tanh", kernel_initializer="he_normal", padding="same", name=lyr[0])(x)
            num_kernels = lyr[1]
        if lyr[0].startswith('pool'):
            x = layers.MaxPooling2D(lyr[1], name=lyr[0])(x)
            width_down *= lyr[1][0]
            height_down *= lyr[1][1]

    img_width = img_width // width_down
    img_height = img_height // height_down
    new_shape = img_width, img_height * num_kernels
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    if num_dense_features > 0:
        x = layers.Dense(num_dense_features, activation="relu", name="dense1")(x)
    x = layers.Bidirectional(layers.LSTM(num_out, return_sequences=True))(x)
    x = layers.Dense(num_chars + 1, activation="softmax", name="softmax")(x)
    output = CTCLayer(name="ctc_loss")(labels, x, input_length//width_down, label_length)

    model_inputs = [input_img, labels, input_length, label_length]
    model = keras.models.Model(inputs=model_inputs, outputs=output, name="ocr_model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    return model