slab_ht = 96

elastic_args0 = {
    'translation': 0,
    'zoom': .0,
    'elastic_magnitude': 0,
    'sigma': 1,
    'angle': 0,
    'nearest': True}

elastic_args = {
    'translation': 5,
    'zoom': .15,
    'elastic_magnitude': 0,
    'sigma': 30,
    'angle': 3,
    'nearest': True}

noise_args = {
    'num_blots': slab_ht // 3,
    'erase_fraction': .9,
    'minsize': 4,
    'maxsize': 9}

scribe_args = {
    'height': slab_ht,
    'hbuffer': 5,
    'vbuffer': 0,
    'nchars_per_sample': 10,
}
