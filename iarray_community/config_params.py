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


# Global variable for random seed
RANDOM_SEED = 0


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
    enforce_frame: Any
    plainbuffer: Any


def default_filters():
    return [Filters.BITSHUFFLE]


@dataclass
class Defaults(object):
    # Config params
    # Keep in sync the defaults below with Config.__doc__ docstring.
    _config = None
    codec: Codecs = Codecs.ZSTD
    clevel: int = 1
    use_dict: bool = False
    filters: List[Filters] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    dtype: (np.float32, np.float64) = np.float64

    # Store
    _store = None
    chunks: Sequence = None
    blocks: Sequence = None
    urlpath: str = None
    enforce_frame: bool = False
    plainbuffer: bool = True

    def __post_init__(self):
        # Initialize config and store with its getters and setters
        self.config = self.config

    # Accessors only meant to serve as default_factory
    def _codec(self):
        return self.codec

    def _clevel(self):
        return self.clevel

    def _use_dict(self):
        return self.use_dict

    def _filters(self):
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

    def _enforce_frame(self):
        return self.enforce_frame

    def _plainbuffer(self):
        return self.plainbuffer

    @property
    def store(self):
        if self._store is None:
            # Bootstrap the defaults
            return DefaultStore(
                chunks=self.chunks,
                blocks=self.blocks,
                urlpath=self.urlpath,
                enforce_frame=self.enforce_frame,
                plainbuffer=self.plainbuffer,
            )
        return self._store

    def set_store(self, value):
        if not hasattr(value, "chunks"):
            raise ValueError(f"You need to use a `Store` instance")
        self.chunks = value.chunks
        self.blocks = value.blocks
        self.urlpath = value.urlpath
        self.enforce_frame = value.enforce_frame
        self.plainbuffer = value.plainbuffer
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
        The chunks for the output array.  If None (the default), a sensible default
        will be used based on the shape of the array and the size of caches in the current
        processor.
    blocks : list, tuple
        The blocks for the output array.  If None (the default), a sensible default
        will be used based on the shape of the array and the size of caches in the current
        processor.
    urlpath : str
        The name of the file for persistently storing the output array.  If None (the default),
        the output array will be stored in-memory.
    enforce_frame : bool
        If True, the output array will be stored as a frame, even when in-memory.  If False
        (the default), the store will be sparse.  Currently, persistent store only supports
        the frame format. When in-memory, the array can be in sparse (the default)
        or contiguous form (frame), depending on this flag.
    plainbuffer : bool
        When True, the output array will be stored on a plain, contiguous buffer, without
        any compression.  This can help faster data sharing among other data containers
        (e.g. NumPy).  When False (the default), the output array will be stored in a Blosc
        container, which can be compressed (the default).
    """

    global defaults
    chunks: Union[Sequence, None] = field(default_factory=defaults._chunks)
    blocks: Union[Sequence, None] = field(default_factory=defaults._blocks)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    enforce_frame: bool = field(default_factory=defaults._enforce_frame)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

    def __post_init__(self):
        self.urlpath = (
            self.urlpath.encode("utf-8") if isinstance(self.urlpath, str) else self.urlpath
        )
        self.enforce_frame = True if self.urlpath else self.enforce_frame

        if self.chunks and self.blocks:
            self.plainbuffer = False
        elif not self.chunks and not self.blocks:
            self.plainbuffer = True
        else:
            if self.plainbuffer:
                raise ValueError("plainbuffer array does not support neither a chunks nor blocks")
            else:
                raise ValueError("blosc array needs chunks and blocks")

@dataclass
class Config():
    """Dataclass for hosting the different ironArray parameters.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    codec : Codecs
        The codec to be used inside Blosc.  Default is :py:obj:`Codecs.ZSTD <Codecs>`.
    clevel : int
        The compression level.  It can have values between 0 (no compression) and
        9 (max compression).  Default is 1.
    favor : Favors
        What favor when compressing. Possible values are :py:obj:`Favors.SPEED <Favors>`
        for better speed, :py:obj:`Favors.CRATIO <Favors>` for bettwer compresion ratios
        and :py:obj:`Favors.BALANCE <Favors>`.  Default is :py:obj:`Favors.BALANCE <Favors>`.
    filters : list
        The list of filters for Blosc.  Default is [:py:obj:`Filters.BITSHUFFLE <Filters>`].
    fp_mantissa_bits : int
        The number of bits to be kept in the mantissa in output arrays.  If 0 (the default),
        no precision is capped.  FYI, double precision have 52 bit in mantissa, whereas
        single precision has 23 bit.  For example, if you set this to 23 for doubles,
        you will be using a compressed store very close as if you were using singles.
    use_dict : bool
        Whether Blosc should use a dictionary for enhanced compression (currently only
        supported by :py:obj:`Codecs.ZSTD <Codecs>`).  Default is False.
    nthreads : int
        The number of threads for internal ironArray operations.  This number can be
        silently capped to be the number of *logical* cores in the system.  If 0
        (the default), the number of logical cores in the system is used.
    eval_method : Eval
        Method to evaluate expressions.  The default is :py:obj:`Eval.AUTO <Eval>`, where the
        expression is analyzed and the more convenient method is used.
    seed : int
        The default seed for internal random generators.  If None (the default), a
        seed will automatically be generated internally for you.
    random_gen : RandomGen
        The random generator to be used.  The default is
        :py:obj:`RandomGen.MERSENNE_TWISTER <RandomGen>`.
    btune: bool
        Enable btune machinery. The default is True.
    dtype: (np.float32, np.float64)
        The data type to use. The default is np.float64.
    store : Store
        Store instance where you can specify different properties of the output
        store.  See :py:obj:`Store` docs for details.  For convenience, you can also
        pass all the Store parameters directly in this constructor too.

    See Also
    --------
    set_config
    config
    """

    codec: Codecs = field(default_factory=defaults._codec)
    clevel: int = field(default_factory=defaults._clevel)
    filters: List[Filters] = field(default_factory=defaults._filters)
    fp_mantissa_bits: int = field(default_factory=defaults._fp_mantissa_bits)
    use_dict: bool = field(default_factory=defaults._use_dict)
    nthreads: int = field(default_factory=defaults._nthreads)
    dtype: (np.float32, np.float64) = field(default_factory=defaults._dtype)
    store: Store = None  # delayed initialization

    # These belong to Store, but we accept them in top level too
    chunks: Union[Sequence, None] = field(default_factory=defaults._chunks)
    blocks: Union[Sequence, None] = field(default_factory=defaults._blocks)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    enforce_frame: bool = field(default_factory=defaults._enforce_frame)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

    def __post_init__(self):
        if self.store is None:
            self.store = Store(
                chunks=self.chunks,
                blocks=self.blocks,
                urlpath=self.urlpath,
                enforce_frame=self.enforce_frame,
                plainbuffer=self.plainbuffer,
            )
        if self.nthreads == 0:
            self.nthreads = 1

    def _replace(self, **kwargs):
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
        cfg = Config(**kwargs)
        return cfg

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
            'sequencial': self.enforce_frame
        }
        return kwargs

# Global config
global_config = Config()
global_diff = []


def get_config(cfg=None):
    """Get the global defaults for iarray operations.

    Parameters
    ----------
    cfg
        The base configuration to which the changes will apply.

    Returns
    -------
    ia.Config
        The existing global configuration.

    See Also
    --------
    ia.set_config
    """
    global global_config
    global global_diff

    if not cfg:
        cfg = global_config
    else:
        cfg = copy.deepcopy(cfg)

    for diff in global_diff:
        cfg = cfg._replace(**diff)

    return cfg


def set_config(cfg: Config = None, shape=None, **kwargs):
    """Set the global defaults for iarray operations.

    Parameters
    ----------
    cfg : ia.Config
        The configuration that will become the default for iarray operations.
        If None, the defaults are not changed.
    shape : Sequence
        This is not part of the global configuration as such, but if passed,
        it will be used so as to compute sensible defaults for store properties
        like chunks and blocks.  This is mainly meant for internal use.
    kwargs : dict
        A dictionary for setting some or all of the fields in the ia.Config
        dataclass that should override the current configuration.

    Returns
    -------
    ia.Config
        The new global configuration.

    See Also
    --------
    ia.Config
    ia.get_config
    """
    global global_config
    global global_diff
    global defaults

    cfg_old = get_config()
    d_old = asdict(cfg_old)

    if cfg is None:
        cfg = copy.deepcopy(cfg_old)
    else:
        cfg = copy.deepcopy(cfg)

    if kwargs != {}:
        cfg = cfg._replace(**kwargs)
    if shape is not None:
        cfg._replace(**{"store": cfg.store})

    d = asdict(cfg)

    diff = {k: d[k] for k in d.keys() if d_old[k] != d[k]}
    if "store" in diff:
        diff["store"] = Store(**diff["store"])

    global_diff.append(diff)
    defaults.config = cfg

    return get_config()


# Initialize the configuration


@contextmanager
def config(cfg: Config = None, shape=None, **kwargs):
    """Create a context with some specific configuration parameters.

    All parameters are the same than in `ia.set_config()`.
    The only difference is that this does not set global defaults.

    See Also
    --------
    ia.set_config
    ia.Config
    """
    global global_config
    global global_diff
    global defaults

    cfg = set_config(cfg, shape, **kwargs)

    try:
        yield cfg
    finally:
        global_diff.pop()

        cfg_old = copy.deepcopy(global_config)
        for diff in global_diff:
            cfg_old = cfg_old._replace(**diff)
        defaults.config = cfg_old


def reset_config_defaults():
    """Reset the defaults of the configuration parameters."""
    global global_config
    global global_diff
    global defaults

    defaults.config = Defaults()
    global_config = Config()
    global_diff = []
    return global_config


if __name__ == "__main__":
    cfg_ = get_config()
    print("Defaults:", cfg_)
    assert cfg_.store.enforce_frame is False

    set_config(store=Store(enforce_frame=True, chunks=(100, 100), blocks=(10, 10)))
    cfg = get_config()
    print("1st form:", cfg)
    assert cfg.store.enforce_frame is True

    set_config(enforce_frame=False)
    cfg = get_config()
    print("2nd form:", cfg)
    assert cfg.store.enforce_frame is False

    set_config(Config(clevel=5))
    cfg = get_config()
    print("3rd form:", cfg)
    assert cfg.clevel == 5

    with config(clevel=0, enforce_frame=True) as cfg_new:
        print("Context form:", cfg_new)
        assert cfg_new.store.enforce_frame is True
        assert get_config().clevel == 0

    cfg = ia.Config(codec=ia.Codecs.BLOSCLZ)
    cfg2 = ia.set_config(cfg=cfg, codec=ia.Codecs.LIZARD)
    print("Standalone config:", cfg)
    print("Global config", cfg2)

    cfg = ia.set_config(cfg_)
    print("Defaults config:", cfg)
