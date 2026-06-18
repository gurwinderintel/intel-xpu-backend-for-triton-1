from conversion import float_conversion
from core_ops import dot_scaled
from core_ops import canonicalize_pointers

if __name__ == '__main__':
    for mod in (float_conversion, dot_scaled, canonicalize_pointers):
        mod.run_benchmarks()
