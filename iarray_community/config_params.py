###########################################################################################
# Copyright INAOS GmbH, Thalwil, 2018.
# Copyright Francesc Alted, 2018.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of INAOS GmbH
# and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
# Information and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

import iarray_community as ia

from dataclasses import dataclass, field, fields, replace, asdict
from typing import List, Sequence, Any, Union
from contextlib import contextmanager
import numpy as np
import copy
from enum import Enum


# Compression codecs
class Codec(Enum):
    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    ZLIB = 4
    ZSTD = 5
    LIZARD = 6


# Filters
class Filter(Enum):
    NOFILTER = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    DELTA = 4
    TRUNC_PREC = 8


@dataclass
class DefaultConfig:
    codec: Any
    clevel: Any
    use_dict: Any
    filters: Any
    nthreads: Any
    fp_mantissa_bits: Any
    store: Any
    dtype: Any


@dataclass
class DefaultStore:
    chunks: Any
    blocks: Any
    urlpath: Any
    mode: Any
    contiguous: Any


def default_filters():
    return [Filter.SHUFFLE]


@dataclass
class Defaults(object):
    # Config params
    # Keep in sync the defaults below with Config.__doc__ docstring.
    _config = None
    codec: Codec = Codec.LZ4
    clevel: int = 9
    use_dict: bool = False
    filters: List[Filter] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    dtype: (np.float32, np.float64) = np.float64

    # Store
    _store = None
    chunks: Sequence = None
    blocks: Sequence = None
    urlpath: str = None
    mode: str = "r"
    contiguous: bool = None

    # Keep track of the special params set with default values for consistency checks with btune
    compat_params: set = field(default_factory=set)
    check_compat: bool = True

    def __post_init__(self):
        # Initialize config and store with its getters and setters
        self.config = self.config

    # Accessors only meant to serve as default_factory
    def _codec(self):
        self.compat_params.add("codec")
        return self.codec

    def _clevel(self):
        self.compat_params.add("clevel")
        return self.clevel

    def _use_dict(self):
        return self.use_dict

    def _filters(self):
        self.compat_params.add("filters")
        return self.filters

    def _nthreads(self):
        return self.nthreads

    def _fp_mantissa_bits(self):
        return self.fp_mantissa_bits

    def _dtype(self):
        return self.dtype

    @property
    def config(self):
        if self._config is None:
            # Bootstrap the defaults
            return DefaultConfig(
                codec=self.codec,
                clevel=self.clevel,
                use_dict=self.use_dict,
                filters=self.filters,
                nthreads=self.nthreads,
                fp_mantissa_bits=self.fp_mantissa_bits,
                store=self.store,
                dtype=self.dtype,
            )
        return self._config

    @config.setter
    def config(self, value):
        if not hasattr(value, "codec"):
            raise ValueError(f"You need to use a `Config` instance")
        self.codec = value.codec
        self.clevel = value.clevel
        self.use_dict = value.use_dict
        self.filters = value.filters
        self.nthreads = value.nthreads
        self.fp_mantissa_bits = value.fp_mantissa_bits
        self.dtype = value.dtype
        self._store = value.store
        self._config = value
        if self._store is not None:
            self.set_store(self._store)

    def _chunks(self):
        return self.chunks

    def _blocks(self):
        return self.blocks

    def _urlpath(self):
        return self.urlpath

    def _mode(self):
        return self.mode

    def _contiguous(self):
        return self.contiguous

    @property
    def store(self):
        if self._store is None:
            # Bootstrap the defaults
            return DefaultStore(
                chunks=self.chunks,
                blocks=self.blocks,
                urlpath=self.urlpath,
                mode=self.mode,
                contiguous=self.contiguous,
            )
        return self._store

    def set_store(self, value):
        if not hasattr(value, "chunks"):
            raise ValueError(f"You need to use a `Store` instance")
        self.chunks = value.chunks
        self.blocks = value.blocks
        self.urlpath = value.urlpath
        self.mode = value.mode
        self.contiguous = value.contiguous
        self._store = value


# Global variable where the defaults for config params are stored
defaults = Defaults()


@dataclass
class Store:
    """Dataclass for hosting different store properties.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    chunks : list, tuple
        The chunk shape for the output array.
    blocks : list, tuple
        The block shape for the output array.
    urlpath : str
        The name of the file for persistently storing the output array.  If None (the default),
        the output array will be stored in-memory.
    mode : str
        Persistence mode: 'r' means read only (must exist); 'r+' means read/write (must exist);
        'a' means read/write (create if doesn’t exist); 'w' means create (overwrite if exists);
        'w-' means create (fail if exists).  Default is 'r'.
    contiguous : bool
        If True, the output array will be stored contiguously, even when in-memory.  If False,
        the store will be sparse. The default value is False for in-memory and True for persistent
        storage.
    """

    global defaults
    chunks: Union[Sequence, None] = field(default_factory=defaults._chunks)
    blocks: Union[Sequence, None] = field(default_factory=defaults._blocks)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    mode: bytes or str = field(default_factory=defaults._mode)
    contiguous: bool = field(default_factory=defaults._contiguous)

    def __post_init__(self):
        self.urlpath = (
            self.urlpath.encode("utf-8") if isinstance(self.urlpath, str) else self.urlpath
        )
        if self.contiguous is None and self.urlpath is not None:
            self.contiguous = True
        else:
            self.contiguous = self.contiguous
        self.mode = self.mode.encode("utf-8") if isinstance(self.mode, str) else self.mode


@dataclass
class Config():
    """Dataclass for hosting the different ironArray parameters.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    codec : :class:`Codec`
        The codec to be used inside Blosc.  Default is :py:obj:`Codec.ZSTD <Codec>`.
    clevel : int
        The compression level.  It can have values between 0 (no compression) and
        9 (max compression).  Default is 1.
    filters : :class:`Filter` list
        The list of filters for Blosc.  Default is [:py:obj:`Filter.BITSHUFFLE <Filter>`].
    fp_mantissa_bits : int
        The number of bits to be kept in the mantissa in output arrays.  If 0 (the default),
        no precision is capped.  FYI, double precision have 52 bit in mantissa, whereas
        single precision has 23 bit.  For example, if you set this to 23 for doubles,
        you will be using a compressed store very close as if you were using singles.
        This automatically activates the ia.Filter.TRUNC_PREC at the front of the filter list.
    use_dict : bool
        Whether Blosc should use a dictionary for enhanced compression (currently only
        supported by :py:obj:`Codec.ZSTD <Codec>`).  Default is False.
    nthreads : int
        The number of threads for internal ironArray operations.  This number can be
        silently capped to be the number of *logical* cores in the system.  If 0
        (the default), the number of logical cores in the system is used.
    dtype: (np.float32, np.float64)
        The data type to use. The default is np.float64.
    store : :class:`Store`
        Store instance where you can specify different properties of the output
        store.  See :py:obj:`Store` docs for details.  For convenience, you can also
        pass all the Store parameters directly in this constructor too.

    See Also
    --------
    set_config_defaults
    config
    """

    codec: Codec = field(default_factory=defaults._codec)
    clevel: int = field(default_factory=defaults._clevel)
    filters: List[Filter] = field(default_factory=defaults._filters)
    fp_mantissa_bits: int = field(default_factory=defaults._fp_mantissa_bits)
    use_dict: bool = field(default_factory=defaults._use_dict)
    nthreads: int = field(default_factory=defaults._nthreads)
    dtype: (np.float32, np.float64) = field(default_factory=defaults._dtype)
    store: Store = None  # delayed initialization

    # These belong to Store, but we accept them in top level too
    chunks: Union[Sequence, None] = field(default_factory=defaults._chunks)
    blocks: Union[Sequence, None] = field(default_factory=defaults._blocks)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    mode: bytes or str = field(default_factory=defaults._mode)
    contiguous: bool = field(default_factory=defaults._contiguous)

    def __post_init__(self):
        if defaults.check_compat:
            self.check_config_params()
        # Restore variable for next time
        defaults.compat_params = set()
        defaults.check_compat = True

        if self.urlpath is not None and self.contiguous is None:
            self.contiguous = True

        if self.store is None:
            self.store = Store(
                chunks=self.chunks,
                blocks=self.blocks,
                urlpath=self.urlpath,
                mode=self.mode,
                contiguous=self.contiguous,
            )

        if self.nthreads <= 0:
            ncores = 1
            self.nthreads = ncores

        # Activate TRUNC_PREC filter only if mantissa_bits > 0
        if self.fp_mantissa_bits != 0 and Filter.TRUNC_PREC not in self.filters:
            self.filters.insert(0, Filter.TRUNC_PREC)
        # De-activate TRUNC_PREC filter if mantissa_bits == 0
        if self.fp_mantissa_bits == 0 and Filter.TRUNC_PREC in self.filters:
            self.filters.pop(0)


    def _replace(self, **kwargs):
        # When a replace is done a new object from the class is created with all its params passed as kwargs
        defaults.check_compat = False
        cfg_ = replace(self, **kwargs)
        if "store" in kwargs:
            store = kwargs["store"]
            if store is not None:
                for field in fields(Store):
                    setattr(cfg_, field.name, getattr(store, field.name))
        else:  # avoid overwriting the store
            store_args = dict(
                (field.name, kwargs[field.name]) for field in fields(Store) if field.name in kwargs
            )
            cfg_.store = replace(cfg_.store, **store_args)
        return cfg_

    def __deepcopy__(self, memodict={}):
        kwargs = asdict(self)
        # asdict is recursive, but we need the store kwarg as a Store object
        kwargs["store"] = Store(**kwargs["store"])
        defaults.check_compat = False
        cfg = Config(**kwargs)
        return cfg

    def check_config_params(self, **kwargs):
        pass

    @property
    def kwargs(self):
        return asdict(self)

    @property
    def cat_kwargs(self):
        kwargs = {
            'codec': self.codec,
            'clevel': self.clevel,
            'usedict': self.use_dict,
            'nthreads': self.nthreads,
            'filters': self.filters,
            'filtersmeta': [0] * len(self.filters),  # no actual meta info for SHUFFLE, but anyway...
            'chunks': self.chunks,
            'blocks': self.blocks,
            'urlpath': self.urlpath,
            'sequencial': self.contiguous
        }
        return kwargs

# Global config
global_config = Config()


def get_config_defaults():
    """Get the global defaults for iarray operations.

    Returns
    -------
    :class:`Config`
        The existing global configuration.

    See Also
    --------
    set_config_defaults
    """
    return global_config


def set_config_defaults(cfg: Config = None, **kwargs):
    """Set the global defaults for iarray operations.

    Parameters
    ----------
    cfg : :class:`Config`
        The configuration that will become the default for iarray operations.
        If None, the defaults are not changed.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :class:`Config`
        The new global configuration.

    See Also
    --------
    Config
    get_config_defaults
    """
    global global_config
    global defaults

    cfg_old = get_config_defaults()

    if cfg is None:
        cfg = copy.deepcopy(cfg_old)
    else:
        cfg = copy.deepcopy(cfg)

    if kwargs != {}:
        cfg.check_config_params(**kwargs)
        # The default when creating frames on-disk is to use contiguous storage (mainly because of performance  reasons)
        if (kwargs.get("contiguous", None) is None
            and cfg.contiguous is None
            and kwargs.get("urlpath", None) is not None):
            cfg = cfg._replace(**dict(kwargs, contiguous=True))
        else:
            cfg = cfg._replace(**kwargs)

    global_config = cfg
    defaults.config = cfg

    return get_config_defaults()


# Initialize the configuration


@contextmanager
def config(cfg: Config = None, shape=None, **kwargs):
    """Create a context with some specific configuration parameters.

    All parameters are the same than in :class:`Config()`.
    The only difference is that this does not set global defaults.

    See Also
    --------
    set_config_defaults
    Config
    """
    global global_config
    global defaults

    cfg_aux = get_config_defaults()
    cfg = set_config_defaults(cfg, **kwargs)

    try:
        yield cfg
    finally:
        defaults.config = cfg_aux
        global_config = cfg_aux


def reset_config_defaults():
    """Reset the defaults of the configuration parameters."""
    global global_config
    global defaults

    defaults.config = Defaults()
    global_config = Config()
    return global_config
