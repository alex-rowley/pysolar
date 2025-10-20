from . import \
    constants, \
    solartime as stime, \
    radiation, \
    util, \
    solar, \
    numeric, \
    vectorised

def use_numpy():
    numeric.use_numpy()

def use_math():
    numeric.use_math()