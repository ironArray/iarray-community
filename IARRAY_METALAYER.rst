ironArray metalayer
+++++++++++++++++++

ironArray containers are created by specifying a Blosc2 metalayer on top of a Caterva container for storing
dtype information.  Specifically, the ironArray metalayer is named 'iarray' and it follows this format::

    |----0-----|---0x01---|---0x02---|---0x03---|
    |---0x93---|-version--|--dtype---|--flags---|
    |----------|----------|----------|----------|
         ^          ^          ^          ^
         |          |          |          |
         |          |          |          +-- [msgpack] positive fixnum for flags (up to 7 flags, currently unused)
         |          |          +-- [msgpack] positive fixnum for the data type (up to 127)
         |          +-- [msgpack] positive fixnum for the metalayer format version (up to 127)
         +-- [msgpack] fixarray with 3 elements


- The supported data types are:
    - float64: 0
    - float32: 1
    - int64: 10
    - int32: 11
    - int16: 12
    - int8: 13
    - uint64: 16
    - uint32: 17
    - uint16: 18
    - uint8: 19
    - bool: 24
