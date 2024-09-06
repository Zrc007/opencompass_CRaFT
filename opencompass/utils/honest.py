REFUSE_KEYWORDS = {
    'en': [
        "I don't know",
    ],
    'zh': [
        '不知道',
    ],
    'zht': [
        '不知道',
    ],
}
# all to lower-case
REFUSE_KEYWORDS = {
    k: [_.lower() for _ in v]
    for k, v in REFUSE_KEYWORDS.items()
}
# extend en to all other langs
for lang in REFUSE_KEYWORDS:
    if lang == 'en':
        continue
    REFUSE_KEYWORDS[lang].extend(REFUSE_KEYWORDS['en'])
print(f'REFUSE_KEYWORDS: {REFUSE_KEYWORDS}')


def check_is_refuse(text: str, lang: str) -> bool:
    return any([kw in text.lower() for kw in REFUSE_KEYWORDS[lang]])
