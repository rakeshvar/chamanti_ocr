from itertools import repeat

import numpy as np
from tensorflow import keras

import sys
sys.path.append("../../Lekhaka")
import telugu as lang
from Lekhaka import Scribe, Deformer, Noiser
from Lekhaka import DataGenerator as Gen

from lab.model_by_function import build_model
from banti2chamanti import banti2chamanti
import utils

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

# CRNN Params
convpool_specs, convpool_wts = banti2chamanti(sys.argv[1])
num_lstm_out = 66

# Model
model = build_model(alphabet_size, scribe_args["height"], scriber.width, batch_size,
                    gen.labelswidth, num_lstm_out, convpool_specs, convpool_wts)
feature_model = keras.models.Model(model.get_layer(name="image").input,
                                   [model.get_layer(name="dense1").input,
                                   model.get_layer(name="dense1").output])
model.summary()
feature_model.summary()

from PIL import Image


def as255(v):
    return (255*(v-v.min())/(v.max()-v.min())).T.astype('uint8')


def repea8(a, times):
    h, w = a.shape
    b = np.vstack(repeat(a, times))
    return b.T.reshape((w*times, h)).T


image, labels, image_lengths, label_lengths = gen.get()
cpfs, features = feature_model.predict(image)
_, fw, fh = features.shape
print(image.shape, cpfs.shape, features.shape)

for i in range(batch_size):
    img = as255(1-image[i, :, :, 0])
    cpf = repea8(as255(cpfs[i]), model.width_scaled_down_by)
    feat = repea8(as255(features[i]), model.width_scaled_down_by)
    img = np.vstack((img, cpf, feat))
    Image.fromarray(img).save(f'features/{i}_aimg.png')
