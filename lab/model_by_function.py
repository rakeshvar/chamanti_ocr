from itertools import repeat

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM

from ctc_layer import CTCLayer
from utils import couple


def conv_pool(x, specs, weights):
    print("Adding conv-ppol layers.")
    i, width_down, height_down, num_filters = 0, 1, 1, 1
    slab_max_wd, slab_ht = x.shape[1:3]

    for spec, wts in zip(specs, weights):
        name = spec['name']
        print(name)
        if name.lower().startswith("conv"):
            num_filters = spec.pop('filters')
            trainable = wts is None
            if not trainable:
                wts = [w.T for w in wts]                        # TODO: Move it out of here
            x = Conv2D(num_filters, spec.pop('kernel_size'),
                       trainable=trainable,
                       weights=wts,
                       **spec)(x)
        elif name.lower().startswith("pool"):
            pool_size = couple(spec.pop("pool_size"))
            x = MaxPooling2D(pool_size, **spec)(x)
            width_down *= pool_size[0]
            height_down *= pool_size[1]
        else:
            break
        i += 1

    new_shape = slab_max_wd//width_down, (slab_ht//height_down) * num_filters
    x = Reshape(target_shape=new_shape, name="reshape")(x)

    if i < len(specs):
        spec = specs[i]
        try:
            wts = weights[i]
        except TypeError:
            wts = next(weights)
        if spec['name'].lower().startswith('dense'):
            x = Dense(spec.pop('nunits'), weights=wts, trainable=wts is None, **spec)(x)
        else:
            raise ValueError(f"Did not understand layer {spec}")

    return x, width_down


def build_model(num_chars,
                slab_ht, slab_max_wd, batch_size,
                max_labelings_len,
                num_lstm_out,
                conv_pool_specs,
                conv_pool_wts=repeat(None)):
    image = Input(name="image", batch_shape=(batch_size, slab_max_wd, slab_ht, 1), dtype="float32")
    labeling = Input(name="labeling", batch_shape=(batch_size, max_labelings_len,), dtype="int64")
    image_width = Input(name='image_width', batch_shape=(batch_size, 1,), dtype='int64')
    labeling_length = Input(name='labeling_length', batch_shape=(batch_size, 1,), dtype='int64')

    x, width_down = conv_pool(image, conv_pool_specs, conv_pool_wts)
    x = Bidirectional(LSTM(num_lstm_out, return_sequences=True, stateful=True))(x)
    x = Dense(num_chars + 1, activation="softmax", name="softmax")(x)
    output = CTCLayer(name="ctc_loss")(labeling, x, image_width//width_down, labeling_length)

    model_inputs = [image, labeling, image_width, labeling_length]
    model = keras.models.Model(inputs=model_inputs, outputs=output, name="crnn_ctc_model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    model.width_scaled_down_by = width_down
    return model
