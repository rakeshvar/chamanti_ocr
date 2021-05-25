specs_1 = (
    ['conv1a', 7, (3, 3)],
    ['conv1b', 7, (3, 3)],
    ['pool1', (2, 2)],
    ['conv2a', 21, (3, 3)],
    ['conv2b', 21, (3, 3)],
    ['pool2', (2, 2)],
    ['conv3a', 63, (3, 3)],
    ['conv3b', 63, (3, 3)],
    ['pool3', (2, 2)],
)

specs_2 = (
    ['conv1a', 7, (3, 3)],
    ['conv1b', 7, (3, 3)],
    ['pool1', (2, 3)],             # Along width, along height of original slab
    ['conv2a', 21, (3, 3)],
    ['conv2b', 21, (3, 3)],
    ['pool2', (2, 3)],
    ['conv3a', 63, (3, 3)],
    ['conv3b', 63, (3, 3)],
    ['pool3', (2, 3)],
)
