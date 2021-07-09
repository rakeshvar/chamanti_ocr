import pickle

from model_specs import cp, pp


def banti2chamanti(banti_pkl_file_name):
    with open(banti_pkl_file_name, 'rb') as f:
        d = pickle.load(f)
        banti_specs = d["layers"]
        banti_wts = d["allwts"]

    cpspecs = []
    wts = []
    nconv, npool = 0, 0
    for (name, params), wb in zip(banti_specs, banti_wts):
        if name == 'ConvLayer':
            nconv += 1
            cpspecs.append(cp(f'conv{nconv}', params['num_maps'], params['filter_sz'], params['actvn']))
        elif name == 'PoolLayer':
            npool += 1
            cpspecs.append(pp(f'pool{npool}', params['pool_sz']))
        else:
            print(f"Omittng Layer {name} {params}")
            continue
        wts.append(wb)

    return cpspecs, wts
