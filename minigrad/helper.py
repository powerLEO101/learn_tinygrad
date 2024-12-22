import os, functools
from typing import Iterator
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
RETAIN_GRAD = getenv("RETAIN_GRAD")
