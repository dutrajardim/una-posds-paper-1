import unicodedata
import functools as ft
from collections.abc import Callable

def snake_case (txt: str) -> str:
    composition = compose(undescore_string, strip_accents)
    return composition(txt)

def strip_accents(txt: str) -> str:
    return unicodedata.normalize('NFD', txt) \
        .encode('ascii', 'ignore') \
        .decode('utf-8')

def undescore_string(txt: str) -> str:
    return txt.lower() \
        .replace(' ', '_') \
        .replace('.', '_')

def compose(*fns: Callable) -> Callable:
    return ft.reduce(lambda f, g: lambda x: f(g(x)), fns, lambda x: x)