from tensorflow import keras

from utils import couple


def actv(name):
    if name.startswith('relu') and len(name) > 4:
        return lambda x: keras.activations.relu(x, alpha=int(name[4:])/100.)
    return name


def cp(name, num_filters, kernel_size, activation):
    return {'name': name, 'filters': num_filters, 'kernel_size': couple(kernel_size), 
            'activation': actv(activation), 'padding': 'same'}


def pp(name, pool_sz):
    return {'name':name, 'pool_size': couple(pool_sz)}


specs_1 = (
    cp('conv1a', 7, 3, 'relu50'),
    cp('conv1b', 7, 3, 'relu50'),    pp('pool1', 2),
    cp('conv2a', 21, 3, 'relu20'),
    cp('conv2b', 21, 3, 'relu20'),    pp('pool2', 2),
    cp('conv3a', 63, 3, 'relu10'),
    cp('conv3b', 63, 3, 'relu10'),    pp('pool3', 2),
)

specs_2 = (
    cp('conv1a', 7, 3, 'relu50'),
    cp('conv1b', 7, 3, 'relu50'),    pp('pool1', (2, 3)), 
    cp('conv2a', 21, 3, 'relu50'),
    cp('conv2b', 21, 3, 'relu50'),    pp('pool2', (2, 3)),
    cp('conv3a', 63, 3, 'relu10'),
    cp('conv3b', 63, 3, 'relu10'),    pp('pool3', (2, 3)),
)

# Pooling is specified (along width, along height) of the original slab
