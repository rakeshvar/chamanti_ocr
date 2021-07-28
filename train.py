
import sys

from tensorflow import keras

from banti2chamanti import banti2chamanti
from model_builder import ModelBuilder
from model_specs import specs
from post_process import PostProcessor

sys.path.append("../Lekhaka")
import telugu as lang
from Lekhaka import Scribe, Deformer, Noiser
from Lekhaka import DataGenerator as Gen

from default_args import scribe_args, elastic_args, noise_args

########################################################################### Initialize
printer = PostProcessor(lang.symbols)

alphabet_size = len(lang.symbols)
batch_size = 32
scriber = Scribe(lang, **scribe_args)
deformer = Deformer(**elastic_args)
noiser = Noiser(**noise_args)
datagen = Gen(scriber, deformer, noiser, batch_size)

print(scriber)

############################################################################# CRNN Params
if len(sys.argv) == 1:
    sys.argv += ['spec', '1']

command = sys.argv[1]
param = sys.argv[2]
if command == 'spec':
    layers = specs[int(param)]
elif command == 'banti':
    layers = banti2chamanti(param)
    param = param[:-4]
elif command == 'chamanti':
    layers = ...

pkl_fname = f"{command[:2]}-{param}-{{:02d}}-{{}}".format
print("Saving to files like: ", pkl_fname(0, 99))

############################################################################## Model
print("\n\nBuilding Model")
xy_info = {
    "batch_size": batch_size,
    "slab_max_wd": scriber.width,
    "slab_ht": scribe_args["height"],
    "labels_max_len": datagen.labelswidth,
    "alphabet_size": alphabet_size
}
mb = ModelBuilder(layers, xy_info)
model = mb.model
model.summary()
prediction_model = keras.models.Model(model.get_layer(name="image").input,
                                      model.get_layer(name="output").output)


class MyCallBack(keras.callbacks.Callback):
    @staticmethod
    def on_epoch_begin(epoch, logs=None):
        image, labels, image_lengths, label_lengths = datagen.get()
        probabilities = prediction_model.predict(image)
        probs_lengths = image_lengths // model.width_scaled_down_by
        ederr = printer.show_batch(image, image_lengths, labels, label_lengths, probabilities, probs_lengths)
        mb.save_model_specs_weights(pkl_fname(epoch, ederr))


history = model.fit(datagen.keras_data_generator(), steps_per_epoch=100, epochs=100, callbacks=[MyCallBack()])
print(history)
