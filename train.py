from tensorflow import keras
import utils

import sys
sys.path.append("..")
from IndicScribe.scribe import Scribe
from model import build_model
import IndicScribe.telugu as lang

"""Initialize """
scribe_args = {
    'height':	81,
    'hbuffer':	5,
    'vbuffer':	0,
    'maxangle':	0,
    'nchars_per_sample': 10,
}
lang.select_labeler('cv')
alphabet_size = len(lang.symbols)

"""CRNN Params"""
num_dense_features = 0
num_lstm_out = 66
from model_specs import specs_2 as layer_specs

"""Initialize Scriber"""
scriber = Scribe(lang, **scribe_args)
printer = utils.Printer(lang.symbols)
print(scriber)
def data_generator():
    while True:
        yield scriber.get_batch(), None

"""Model"""
model = build_model(alphabet_size, scribe_args["height"], scriber.width, scriber.labelswidth, num_dense_features, num_lstm_out, layer_specs)
prediction_model = keras.models.Model(model.get_layer(name="image").input,
                                      model.get_layer(name="softmax").output)
# model.summary()
# prediction_model.summary()


class MyCallBack(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        num_samples = 5
        for (i, batch) in zip(range(num_samples), data_generator()):
            image, labels, image_lengths, label_lengths = batch[0]
            probabilities = prediction_model.predict(image)
            printer.show_all(labels[0][:label_lengths[0]], image[0, :image_lengths[0], :, 0].T,
                             probabilities[0].T, (i == num_samples-1))

history = model.fit(data_generator(), steps_per_epoch=100, epochs=100, callbacks=[MyCallBack()])
