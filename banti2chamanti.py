import pickle

from utils import cp, pp, den


def banti2chamanti(banti_pkl_file_name,
                   dont_pool_along_width=True,
                   conv_trainable=False,
                   dense_trainable=True):
    """
    Input to Conv layer is (bz, SW, SH, m)
    Kernals are k, (ksw, ksh)
    Output shape is (bz, SW, SH, k)

    Kernal weights will be (ksw, ksh, m, k)

    Banti kernal weights will be (k, m, ksh, ksw) 
        So just apply transpose before loading to chamanti
    """
    print("Loading weights from ", banti_pkl_file_name)
    with open(banti_pkl_file_name, 'rb') as f:
        d = pickle.load(f)

    banti_specs = d["layers"]
    banti_wts = d["allwts"]
    num_banti_layers = len(banti_specs)

    layer_args = []
    last_pool = None
    n, nconv, npool, nmaps = 0, 0, 0, 1

    while n < num_banti_layers:
        name, spec = banti_specs[n]
        if name == 'ElasticLayer':
            pass
        elif name == 'ConvLayer':
            if 'mode' in spec:
                assert spec['mode'] == 'same'
            nconv += 1
            nmaps = spec['num_maps']
            w, b = banti_wts[n]
            layer_args.append(cp(f'conv{nconv}', nmaps, spec['filter_sz'], spec['actvn'],
                                 weights=(w.T, b), trainable=conv_trainable))
        elif name == 'PoolLayer':
            npool += 1
            last_pool = pp(f'pool{npool}', spec['pool_sz'])
            layer_args.append(last_pool)
        else:
            break
        n += 1

    if dont_pool_along_width:
        last_pool['pool_size'] = (1, last_pool['pool_size'][1])

    if n < num_banti_layers:
        name, spec = banti_specs[n]
        if name in ('SoftmaxLayer', 'HiddenLayer'):
            w, b = banti_wts[n]
            nin, nout = w.shape
            d = int((nin//nmaps)**.5)
            assert nin == nmaps*d*d                         # eg: 5342 = 162*6*6
            w = w.reshape((nmaps, d, d, nout))
            w = w[:, :, d//2-1, :].reshape((nmaps*d, nout))  # Extract the middle column
            actvn = spec.get("actvn", "linear")
            layer_args.append(den('dense1', nout, actvn, weights=(w, b), trainable=dense_trainable))

    return layer_args
