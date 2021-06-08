from .iarray import Array
from .constructors import from_buffer, open
from enum import Enum

# Compression codecs
class Codecs(Enum):
    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    ZLIB = 4
    ZSTD = 5
    LIZARD = 6


# Filters
class Filters(Enum):
    NOFILTER = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    DELTA = 4
    TRUNC_PREC = 8

from .config_params import (
    Config,
    Store,
    set_config,
    get_config,
    config,
    reset_config_defaults,
)
