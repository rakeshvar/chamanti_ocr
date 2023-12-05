import sys

from tensorflow import keras

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

xy_info = {
    "batch_size": batch_size,
    "slab_max_wd": scriber.width,
    "slab_ht": scribe_args["height"],
    "labels_max_len": datagen.labelswidth,
    "alphabet_size": alphabet_size
}
############################################################################# CRNN Params
if len(sys.argv) == 1:
    print("""Usage:
    python {0} command argument(s)
    command - spec / banti / chamanti
    {0} spec 0
    {0} banti deepcnn.pkl lstm66
    {0} chamanti cnn-rnn.pkl 
    """.format(sys.argv[0]))
    sys.exit(-1)

command = sys.argv[1]
argument = 0 if command == 'spec' and len(sys.argv) < 3 else sys.argv[2]
if command == 'spec':
    layers = specs[int(argument)]
    mb = ModelBuilder(xy_info, layers)
    pkl_namer = f"sp-{argument}-{{:02d}}-{{}}".format
elif command == 'banti':
    rnnarg = 'rnn66' if len(sys.argv) < 4 else sys.argv[3]
    mb = ModelBuilder.from_banti(xy_info, argument, rnnarg)
    pkl_namer = f"bn-{argument[:-4]}-{{:02d}}-{{}}".format
elif command == 'chamanti':
    mb = ModelBuilder.from_chamanti(xy_info, argument)
    pkl_namer = f"ch-{argument[:-4]}-{{:02d}}-{{}}".format
else:
    raise ValueError("Did not understand args: ", sys.argv[1:])

print("Saving to files like: ", pkl_namer(0, 99))

############################################################################## Model
print("\n\nBuilding Model")
model = mb.model
model.summary()
prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="output").output)


class MyCallBack(keras.callbacks.Callback):
    @staticmethod
    def on_epoch_begin(epoch, logs=None):
        image, labels, image_lengths, label_lengths = datagen.get()
        probabilities = prediction_model.predict(image)
        probs_lengths = image_lengths // model.width_scaled_down_by
        ederr = printer.show_batch(image, image_lengths, labels, label_lengths, probabilities, probs_lengths)
        mb.save_model_specs_weights(pkl_namer(epoch, ederr))


history = model.fit(datagen.keras_data_generator(), steps_per_epoch=100, epochs=100, callbacks=[MyCallBack()])
print(history)
