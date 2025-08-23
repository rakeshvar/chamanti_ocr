from utils import cp, pp, rnn

specs_0 = (
    cp('conv1a', 7, 3, 'relu50'),
    cp('conv1b', 7, 3, 'relu50'),    pp('pool1', 2),
    cp('conv2a', 21, 3, 'relu20'),
    cp('conv2b', 21, 3, 'relu20'),    pp('pool2', 2),
    cp('conv3a', 63, 3, 'relu10'),
    cp('conv3b', 63, 3, 'relu10'),    pp('pool3', 2),
    rnn('lstm', 66),
)

specs_1 = (
    cp('conv1a', 7, 3, 'relu50'),
    cp('conv1b', 7, 3, 'relu50'),    pp('pool1', (3, 2)),
    cp('conv2a', 21, 3, 'relu50'),
    cp('conv2b', 21, 3, 'relu50'),    pp('pool2', (3, 2)),
    cp('conv3a', 63, 3, 'relu10'),
    cp('conv3b', 63, 3, 'relu10'),    pp('pool3', (3, 2)),
    rnn('lstm', 66),
)

specs = [specs_0, specs_1]
