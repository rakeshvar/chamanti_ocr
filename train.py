import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from model_builder import ModelBuilder
from model_specs import specs
from post_process import PostProcessor
from argparser import args

# Need to make sure Lekhaka and telugu are in path
try:
 import telugu as lang
 from Lekhaka import Scribe, Deformer, Noiser
 from Lekhaka import DataGenerator
except ModuleNotFoundError:
 import Lekhaka.telugu as lang
 from Lekhaka.Lekhaka import Scribe, Deformer, Noiser
 from Lekhaka.Lekhaka import DataGenerator

from default_args import scribe_args, elastic_args, noise_args

############################################################################## Parse Args & Initialize
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

batch_size = args.batch_size
printer = PostProcessor(lang.symbols)
alphabet_size = len(lang.symbols)
scriber = Scribe(lang, **scribe_args)
deformer = Deformer(**elastic_args)
noiser = Noiser(**noise_args)
datagen = DataGenerator(scriber, deformer, noiser, batch_size)

print(scriber)

xy_info = {
    "batch_size": batch_size,
    "slab_max_wd": scriber.width,
    "slab_ht": scribe_args["height"],
    "labels_max_len": datagen.labelswidth,
    "alphabet_size": alphabet_size
}

############################################################################# Set-Up Dataset

dataset = tf.data.Dataset.from_generator(
    datagen.generator,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, scriber.width, scriber.height, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, datagen.labelswidth), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int32))
).prefetch(buffer_size=tf.data.AUTOTUNE)

def proper_x_y(images, labels, image_lengths, label_lengths):
    return {
        'image': images,
        'labeling': labels,
        'image_width': image_lengths,
        'labeling_length': label_lengths
    }, labels  # Dummy target for model.fit

dataset = dataset.map(proper_x_y)

########################################################################### Command
if args.command == "spec":
    print(f"Running spec with num={args.num}")
    layers = specs[int(args.num)]
    mb = ModelBuilder(xy_info, layers)
    pkl_namer = f"{out_dir}/sp-{args.num}-{{:02d}}-{{}}".format

elif args.command == "banti":
    print(f"Running banti with {args.pkl_file}, {args.string_arg}")
    in_pkl_stem = Path(args.pkl_file).stem
    mb = ModelBuilder.from_banti(xy_info, args.pkl_file, args.rnnarg)
    pkl_namer = f"{out_dir}/bn-{in_pkl_stem}-{{:02d}}-{{}}".format

elif args.command == "chamanti":
    print(f"Running chamanti with {args.pkl_file}")
    in_pkl_stem = Path(args.pkl_file).stem
    mb = ModelBuilder.from_chamanti(xy_info, args.pkl_file)
    pkl_namer = f"{out_dir}/ch-{in_pkl_stem}-{{:02d}}-{{}}".format

else:
    raise ValueError("Unknown command received: ", args.command)

print("Saving to files like: ", pkl_namer(0, 99), ".pkl")

############################################################################## Model
print("\n\nBuilding Model")
model = mb.model
model.summary()
prediction_model = keras.models.Model(
    model.get_layer(name="image").output,
    model.get_layer(name="output").output)

class MyCallBack(keras.callbacks.Callback):
    @staticmethod
    def on_epoch_begin(epoch, logs=None):
        image, labels, image_lengths, label_lengths = datagen.get()
        probabilities = prediction_model.predict(image)
        probs_lengths = image_lengths // model.width_scaled_down_by
        ederr = printer.show_batch(image, image_lengths, labels, label_lengths, probabilities, probs_lengths)
        mb.save_model_specs_weights(pkl_namer(epoch, ederr))

history = model.fit(dataset, steps_per_epoch=args.steps_per_epoch, epochs=args.num_epochs, callbacks=[MyCallBack()])
print(history)
