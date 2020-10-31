from .base import BaseTokenizer
from .nltk import NLTKTokenizer
from .whitespace import WhitespaceTokenizer
from .regex import RegexTokenizer
from typing import Union


tokenizer_map = {
    NLTKTokenizer.NAME: NLTKTokenizer,
    WhitespaceTokenizer.NAME: WhitespaceTokenizer,
    RegexTokenizer.NAME: RegexTokenizer,
    None: WhitespaceTokenizer,
}


def get_tokenizer(value: Union[str, 'BaseTokenizer']):
    if value is None or isinstance(value, str):
        return tokenizer_map[value]()
    elif isinstance(value, BaseTokenizer):
        return value
    raise ValueError(f'invalid tokenizer, {value}')
