from collections import namedtuple

LC = namedtuple('LC', ('x', 'y', 'yerr'))# time, flux/mag, flux/mag err

Aff = namedtuple('Aff', ('tx', 'ty', 'dx', 'dy'))# translation, dilation in x, y
