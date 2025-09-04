"""
Protobuf message definitions for neurOS.

This package contains the ``brain_event.proto`` file defining the
``BrainEvent`` message used for transporting events through Kafka,
ZeroMQ and Redis.  To generate Python bindings, run:

.. code-block:: bash

    protoc --python_out=. brain_event.proto

Alternatively, use ``grpcio-tools`` or your build system of choice.
The Python modules generated should be imported by your ingestion
pipeline to serialise and deserialise events efficiently.
"""

__all__ = []