"""Dictionary-only diagnostic: all uploaded matching logic retained.
Cache pure string normalization for speed; do not change algorithmic semantics.
"""
from pathlib import Path
from functools import lru_cache
import audited_counts as a
m = a.legacy
m.NEGATIVE_DIRECTIONS = list(a.NEGATIVE)
m.POSITIVE_DIRECTIONS = list(a.POSITIVE)
m.DIRECTION_BASES = m.vocabulary_bases(m.NEGATIVE_DIRECTIONS + m.POSITIVE_DIRECTIONS)
m.normalize_pattern = lru_cache(maxsize=1024)(m.normalize_pattern)
m.run(a.ROOT/'data/minutes', a.ROOT/'results/dictionary_only_correction')
