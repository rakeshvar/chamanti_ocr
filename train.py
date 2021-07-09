from tensorflow import keras
import utils

import sys
from model import build_model

sys.path.append("../Lekhaka")
import telugu as lang
from Lekhaka import Scribe, Deformer, Noiser
from Lekhaka import ParallelDataGenerator as Gen

# Initialize
from default_args import scribe_args, elastic_args, noise_args
alphabet_size = len(lang.symbols)
batch_size = 32

scriber = Scribe(lang, **scribe_args)
printer = utils.Printer(lang.symbols)
deformer = Deformer(**elastic_args)
noiser = Noiser(**noise_args)
gen = Gen(scriber, deformer, noiser, batch_size)
print(scriber)


def data_generator():
    while True:
        yield gen.get(), None


# CRNN Params
num_dense_features = 0
num_lstm_out = 66
from model_specs import specs_2 as layer_specs

# Model
model = build_model(alphabet_size, scribe_args["height"], scriber.width, batch_size,
                    gen.labelswidth, num_dense_features, num_lstm_out, layer_specs)
prediction_model = keras.models.Model(model.get_layer(name="image").input,
                                      model.get_layer(name="softmax").output)
model.summary()
# prediction_model.summary()


class MyCallBack(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        num_samples = 5
        image, labels, image_lengths, label_lengths = gen.get()
        probabilities = prediction_model.predict(image)
        for i in range(num_samples):
            printer.show_all(labels[i][:label_lengths[i]],
                             image[i, :image_lengths[i]:2, ::2, 0].T,
                             probabilities[i, :image_lengths[i]//model.width_scaled_down_by].T,
                             (i == num_samples-1))


history = model.fit(data_generator(), steps_per_epoch=10, epochs=10, callbacks=[MyCallBack()])
