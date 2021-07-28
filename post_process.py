from itertools import count

import editdistance
import numpy as np
from colored import fg, attr
reset = attr('reset')
cols = np.array([f'{fg(231)}█'] + [f'{fg(250 -i)}█' for i in range(19)])


def slab_log(slab, names=None):
    cslab = cols[np.clip(np.floor(-2*np.log2(slab)).astype(int), 0, 19)]
    if names is None:
        names = ' ' * slab.shape[0]
    for ir, r in enumerate(cslab):
        print(f'{ir:2d}¦' + ''.join(r) + reset + f'¦ {names[ir]}')


bins1 = [0, .1, .25, .4, .6, .75, .9, 1.0001], '- .*o0@#+'
bins2 = [0, .15, .35, .65, .85, 1.0001],  '- ░▒▓█+'


def get_printer(thresholds, chars):
    def _printer(slab, name=None, col_names=None):
        name and print(f"Name: ", name)
        for irow, row in enumerate(slab):
            print(f'{irow:2d}¦', end='')
            for val in row:
                for t, c in zip(thresholds, chars):
                    if val < t:
                        print(c, end='')
                        break
                else:
                    print(chars[-1], end='')
            print('¦ ', col_names[irow] if col_names else '')
    return _printer


slab_print = get_printer(*bins2)


class PostProcessor:
    def __init__(self, symbols):
        self.n_classes = len(symbols)
        self.symbols = symbols + ['_']

    def labels_to_chars(self, labels):
        return [self.symbols[label] for label in labels]

    def remove_blanks_repeats(self, labels):
        labels_out = []
        for i, label in enumerate(labels):
            if (label != self.n_classes) and (i == 0 or label != labels[i-1]):
                labels_out.append(label)
        return labels_out

    def decode(self, softmax_firings):
        max_labels = np.argmax(softmax_firings, 0)
        return self.remove_blanks_repeats(max_labels)

    def show_all(self, shown_labels, shown_img, softmax_firings, show_imgs):
        shown_chars = self.labels_to_chars(shown_labels)
        print(f'Shown : {shown_labels} {shown_chars}')

        if softmax_firings is not None:
            seen_labels = self.decode(softmax_firings)
            seen_chars = self.labels_to_chars(seen_labels)
            print(f'Seen  : {seen_labels} {seen_chars}')

        if show_imgs:
            print('Image Shown:')
            slab_print(shown_img)

        if show_imgs and softmax_firings is not None:
            seen_labels = list(set(seen_labels) - set(shown_labels))
            seen_chars = self.labels_to_chars(seen_labels)
            l = list(shown_labels) + [0, self.n_classes] + seen_labels
            c = shown_chars + ['space', 'blank'] + seen_chars
            print('SoftMax Firings:')
            slab_log(softmax_firings[l], c)

    def editdistances(self, truths, lengths, probabilities, problengths):
        guesses = [self.decode(prob[:il].T) for prob, il in zip(probabilities, problengths)]
        tot_len = sum(max(l1, l2) for l1, l2 in zip(lengths, problengths))
        tot_dist = sum(editdistance.eval(t, g) for t, l, g in zip(truths, lengths, guesses))
        err = tot_dist / tot_len
        print(f"Edit distance: {tot_dist}/{tot_len} = {err:.2f}")
        return int(100*err)

    def show_batch(self,
                   images, image_lengths,
                   labels, label_lengths,
                   probabilities, probs_lengths,
                   num_samples=5):
        for i in range(num_samples):
            self.show_all(labels[i, :label_lengths[i]],
                          images[i, :image_lengths[i]:3, ::2, 0].T,
                          probabilities[i, :probs_lengths[i]].T,
                          i == num_samples-1)
        return self.editdistances(labels, label_lengths, probabilities, probs_lengths)
