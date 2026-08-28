"""Cross-plane scientific evidence orchestration for neurOS.

This namespace lives in the user-facing ``neuros`` distribution so studies may
compose optional ORION scientific authority with optional ecosystem adapters
without reversing dependency direction in lower-level packages.

Study modules must keep heavyweight scientific imports local to execution paths
so the base neurOS runtime remains dependency-light.
"""

__all__: list[str] = []
