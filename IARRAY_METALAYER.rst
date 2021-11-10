ironArray metalayer
+++++++++++++++++++

ironArray containers are created by specifying a Blosc2 metalayer on top of a Caterva container for storing
dtype information.  Specifically, the ironArray metalayer is named 'iarray' and it follows this format::

    |-0-|-1-|-2-|
    | v | dt| fl|
    |---|---|---|
      ^   ^   ^
      |   |   |
      |   |   +--[msgpack] positive fixnum for flags (up to 7 flags, currently unused)
      |   +--[msgpack] positive fixnum for the data type (up to 127, 0 = float64, 1 = float32)
      +--[msgpack] positive fixnum for the metalayer format version (up to 127)
