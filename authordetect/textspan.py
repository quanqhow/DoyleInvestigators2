#! /usr/bin/python3

import re
from .textutils import iter2str
from .typedlist import tlist
from typing import Tuple, Union, Iterable


__all__ = ['TextSpan']


class TextSpan(tlist):
    """Composable container representing a text span.

    Args:
        span ((int,int)): Index bounds from original string.
            End bound is non-inclusive to satisfy: text == orig[slice(*span)]

    Kwargs:
        Arguments passed directly to tlist().
    """
    def __init__(
        self,
        data: Union[str, Iterable['TextSpan']] = None,
        span: Iterable[int] = (0, 0),
        *,
        delim: str = ' ',
        **kwargs,
    ):
        # NOTE: Allow modifying these attributes to improve API use cases.
        self.__dict__['span'] = span
        self.__dict__['delim'] = delim
        kwargs['dtype'] = (str, bytes, type(self))
        super().__init__(
            [data] if isinstance(data, kwargs['dtype']) else data,
            **kwargs,
        )

    def __contains__(self, item):
        return bool(list(self.search(item)))

    def __add__(self, other) -> str:
        return str(self) + str(other)

    def __iadd__(self, other):
        raise ArithmeticError(f'invalid arithmetic assignment operation on {type(self)}')

    def __str__(self):
        # Generated string follows order of tokens, not their spans.
        # NOTE: sort() will do an in-place sort based on spans.
        if len(self.delim) == 1:
            return iter2str(self, self.depth * self.delim)
        return iter2str(self, self.delim)

    @property
    def str(self):
        return str(self)

    @property
    def size(self):
        return sum(1 for _ in self.iter_tokens())

    @property
    def tokens(self):
        return list(str(t) for t in self.iter_tokens())

    @property
    def vocabulary(self):
        return set(str(t) for t in self.iter_tokens())

    @property
    def depth(self):
        def depth(data):
            return (
                0
                if isinstance(data, (str, bytes)) or len(data) == 0
                else 1 + max(map(depth, data))
            )

        return depth(self)

    def sort(self, **kwargs):
        """Recursive in-place sort. Defaults to sorting by spans."""
        kwargs['key'] = kwargs.get('key', lambda ts: ts.span)
        for item in self:
            if isinstance(item, type(self)):
                super().sort(**kwargs)
                item.sort(**kwargs)

    def count(self, value: str, *, exact_match: bool = False) -> int:
        """Search for value in text spans and return count."""
        return len(list(self.search(value, exact_match=exact_match)))

    def search(
        self,
        value: str,
        *,
        exact_match: bool = False,
    ) -> Iterable[Iterable[int]]:
        """Search for value in text spans and return spans."""
        # Use regex to find tokens in strings or multi-word spans
        if not exact_match:
            value = value.lower()
            pattern = re.compile(fr'\b{value}\b')

        def search(obj, span=(0, 0)):
            for item in obj:
                if isinstance(item, type(self)):
                    yield from search(item, item.span)
                elif exact_match:
                    if item == value:
                        yield span
                else:
                    for _ in pattern.finditer(item.lower()):
                        yield span

        yield from search(self)

    # def describe_span(self, span: Tuple[int, int]) -> Iterable[int]:
    #     """Find the hierarchy of structure indexes where a span is located."""
    #     def is_span_bounded_by(s1, s2) -> bool:
    #         """Checks if span1 is bounded by span2."""
    #         return s1[0] >= s2[0] and s1[1] <= s2[1]
    #
    #     idxs = []
    #     # Early exit if span is empty
    #     if span[0] == span[1]:
    #         return idxs
    #     # for item in obj:
    #     #     if isinstance(item, type(self)):
    #     #         if is_span_bounded_by
    #     return idxs

    def get_tokens(self, min_depth=2):
        depth = self.depth
        if depth >= min_depth:
            tokens = list(self.iter_tokens(depth - (min_depth - 1)))
            return type(self)(tokens, (tokens[0].span[0], tokens[-1].span[1]))
        return type(self)()

    def iter_tokens(self, depth: int = -1):
        def iter_tokens(obj, curr_depth):
            for item in obj:
                if curr_depth < 0:
                    if isinstance(item, type(self)):
                        yield from iter_tokens(item, curr_depth)
                    else:
                        yield item
                elif curr_depth == depth:
                    yield item
                elif curr_depth < depth:
                    if isinstance(item, type(self)):
                        yield from iter_tokens(item, curr_depth + 1)
                    else:
                        yield item

        if depth == 0:
            yield self
        elif depth < 0:
            yield from iter_tokens(self, depth)
        else:
            yield from iter_tokens(self, 1)

    def substitute(self, text: str, *, force_capitalization: bool = True):
        """Substitutes the given tokens in a text."""
        new_text = ''
        for i, tspan in enumerate(self):
            # Get text before first new token, if token is "at the beginning"
            # if i == 0 and tspan.span[0] < 3:
                # new_text = text[:tspan.span[0]]

            # Check capitalization for new token.
            # First letter capitalization is applied if any of the first two
            # characters of original token are uppercase to allow a symbol
            # (e.g., quotation mark) and letter combination.
            # NOTE: This is tricky in the sense that if a substituted token is
            # not wanted capitalized, this will force it. Also, this is
            # tokenizer dependent.
            new_token = str(tspan)
            old_token = text[tspan.span[0]:tspan.span[1]]
            if force_capitalization and any(map(str.isupper, old_token[:2])):
                if old_token.isupper():
                    # Caplitalize entire token
                    new_token = new_token.upper()
                elif len(new_token) > 0:
                    # Capitalize first character
                    new_token = new_token[0].upper() + new_token[1:]

            # Get text from end of new token to begin of next token or end
            # of text
            post_text = ''
            if i < len(self) - 1:
                post_text = text[slice(tspan.span[1], self[i+1].span[0])]

            # Substitute new token
            new_text += new_token + post_text
        return new_text
