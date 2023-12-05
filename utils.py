import ast
import sys

from tensorflow import keras


def insert_blanks(y, blank, num_blanks_at_start=1):
    # Insert blanks at alternate locations in the labelling (blank is blank)
    y1 = [blank] * num_blanks_at_start
    for char in y:
        y1 += [char, blank]
    return y1


def read_args(files, default='configs/default.ast'):
    with open(default, 'r') as dfp:
        args = ast.literal_eval(dfp.read())

    for config_file in files:
        with open(config_file, 'r') as cfp:
            override_args = ast.literal_eval(cfp.read())

        for key in args:
            if key in override_args:
                try:
                    args[key].update(override_args[key])
                except AttributeError:
                    args[key] = override_args[key]

    return args


def pprint_probs(probs):
    for row in (10 * probs).astype(int):
        for val in row:
            print(f'{val:+04d}', end='')
        print()


def write_dict(d, f=sys.stdout, level=0):
    tabs = '\t' * level
    print(file=f)
    for k in sorted(d.keys()):
        v = d[k]
        print('{}{}: '.format(tabs, k), file=f, end='')
        if type(v) is dict:
            write_dict(v, f, level+1)
        else:
            print('{}'.format(v), file=f)


def couple(a):
    return a if (type(a) is tuple) else (a, a)


def actv(name):
    if name.startswith('relu') and len(name) > 4:
        def _f(x):
            return keras.activations.relu(x, alpha=int(name[4:]) / 100.)
        _f.name = name
        return _f
    else:
        return name


def cp(name, num_filters, kernel_size, activation, **kwargs):
    return {'name': name,
            'filters': num_filters, 'kernel_size': couple(kernel_size),
            'activation': actv(activation), 'padding': 'same', **kwargs}


def pp(name, pool_sz):
    return {'name': name, 'pool_size': couple(pool_sz)}


def den(name, nunits, activation, **kwargs):
    return {'name': name, 'units': nunits, 'activation': actv(activation), **kwargs}


def rnn(name, nunits, **kwargs):
    return {'name': name, 'units': nunits, **kwargs}
