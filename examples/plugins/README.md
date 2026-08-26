# External plugin examples

This directory contains **standalone example distributions**, not neurOS workspace packages.

The distinction is deliberate. These examples prove that Python entry-point discovery works across the same packaging boundary an external lab or company would use.

- [`neuros-example-plugin`](neuros-example-plugin/README.md): reference `neuros.sources` + `neuros.transforms` package with a deterministic source, provenance-preserving transform, tests, version bounds, and config-first runtime example.

Do not add an external-plugin example to the root workspace merely to make CI easier. A reference plugin should be built and installed as its own wheel in a clean environment.
