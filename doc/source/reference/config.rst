-------------
Configuration
-------------

If you are working with known datasets with stable characteristics, setting the same compression and storage properties every time you create an ironArray array object can become tedious and repetitive. However, you can set default properties in the global configuration to avoid this annoying situation.

.. currentmodule:: iarray_community


Storage properties
==================

.. autosummary::
   :toctree: autofiles/config/
   :nosignatures:

   Store

Configuration object
====================

.. autosummary::
   :toctree: autofiles/config-class/
   :nosignatures:

   Config

Set / Get configuration params
------------------------------

.. autosummary::
   :toctree: autofiles/config/
   :nosignatures:

   set_config
   get_config
   config
   reset_config_defaults


Enumerated classes
==================

.. autosummary::
   :toctree: autofiles/config/
   :nosignatures:

   Codecs
   Filters


Global variables
================

.. py:attribute:: iarray.__version__
