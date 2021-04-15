from __future__ import absolute_import, print_function

from six.moves import range


def gen_split_overlap(seq, size, overlap):
    if size < 1 or overlap < 0:
        raise ValueError("size must be >= 1 and overlap >= 0")

    if overlap % 2 != 0:
        raise ValueError("overlap need to be even")

    if overlap > size:
        raise ValueError("overlap need to be less than size")

    step_size = size - overlap

    if len(seq) <= size:
        yield seq, True, True
    else:
        for i in range(0, len(seq) - overlap, step_size):
            yield seq[i : i + size], i == 0, i + size >= len(seq)
