ironArray metalayer
+++++++++++++++++++

ironArray containers are created by specifying a metalayer on top of a Caterva container for storing
dtype information.  Specifically, ironArray metalayer is named 'iarray' and it follows this format::

    |-   1   -|-  2  -|-  3  -|
    | version | dtype | flags |
    |---------|-------|-------|
         ^        ^       ^
         |        |       |
         |        |       +--[msgpack] flags (currently unused)
         |        +--[msgpack] positive fixnum for the data type (up to 127, 0 = float64, 1 = float32)
         +--[msgpack] positive fixnum for the metalayer format version (up to 127)
