"""Explicit opt-in native Gear; no auto compilation or network downloads."""
import ctypes
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=4)
def library(path):
    p=Path(path)
    if not p.is_absolute(): raise ValueError('GLYPH_CDC_NATIVE must be an absolute trusted library path')
    lib=ctypes.CDLL(str(p.resolve(strict=True)))
    fn=lib.glyph_gear_cuts
    fn.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint64),
                 ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,
                 ctypes.POINTER(ctypes.c_size_t),ctypes.c_size_t]
    fn.restype=ctypes.c_size_t
    return lib

def split(data,gear,minimum,target,maximum,path):
    if not isinstance(data,bytes): raise TypeError('native split requires immutable bytes')
    if minimum<=0 or maximum<minimum or target<=0 or target & (target-1):
        raise ValueError('invalid chunk parameters')
    table=(ctypes.c_uint64*256)(*gear)
    capacity=len(data)//minimum+1
    ends=(ctypes.c_size_t*capacity)()
    count=library(path).glyph_gear_cuts(data,len(data),table,minimum,target,maximum,ends,capacity)
    if count>capacity: raise ValueError('native boundary buffer failure')
    start=0
    for i in range(count):
        end=ends[i]
        if not start<end<=len(data): raise ValueError('invalid native boundary')
        yield start,data[start:end]
        start=end
    if start!=len(data): raise ValueError('native boundaries incomplete')
