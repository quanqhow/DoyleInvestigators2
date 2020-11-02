#! /usr/bin/python3

import json


with open('translations.json') as fd:
    tables = json.load(fd)


def translate(text: str, to_country: str = 'uk', tag: bool = False):
    to_country = to_country.lower()
    if to_country not in ['us', 'uk']:
        raise ValueError("`country` must be one of ['us', 'uk']")

    conversion = 'us_to_uk' if to_country == 'uk' else 'uk_to_us'
    for source, target in tables[conversion].items():
        target = f'<{source}|{target}>' if tag else target
        text = text.replace(source, target)

    return text
