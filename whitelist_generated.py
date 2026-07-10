# Автоматически сгенерировано probe_corpus_resulttype_matrix.py
# Тестовая лемма: 'мать'
# ВНИМАНИЕ: 'EMPTY' может означать как 'фича не поддерживается корпусом',
# так и 'этой леммы просто нет в данном корпусе'. Перед тем как класть
# EMPTY-корпуса в чёрный список, имеет смысл перепроверить на 1-2 других леммах.

RESULTTYPE_CORPUS_WHITELIST = {
    'PORTRAIT_WORD_INFO': ['BLOGS', 'CLASSICS', 'GICR', 'KIDS', 'MAIN', 'MID_RUS', 'OLD_RUS', 'POETIC', 'SPOKEN'],
    'PORTRAIT_CONCORDANCE': ['BLOGS', 'CLASSICS', 'GICR', 'KIDS', 'MAIN', 'MID_RUS', 'OLD_RUS', 'POETIC', 'SPOKEN'],
    'PORTRAIT_STATS': ['BLOGS', 'CLASSICS', 'GICR', 'KIDS', 'MAIN', 'MID_RUS', 'POETIC', 'SPOKEN'],
    'PORTRAIT_SKETCH': ['CLASSICS', 'GICR', 'KIDS', 'MAIN', 'MID_RUS', 'OLD_RUS', 'SPOKEN'],
    'PORTRAIT_FREQUENCY': ['CLASSICS', 'GICR', 'KIDS', 'MAIN', 'MID_RUS', 'OLD_RUS', 'SPOKEN'],
    'PORTRAIT_SIMILAR': ['CLASSICS', 'KIDS', 'MAIN', 'MID_RUS'],
    'PORTRAIT_MORPHEME': ['MAIN'],
    'PORTRAIT_WORDFORMS': ['MAIN', 'OLD_RUS'],
    'PORTRAIT_COGNATES': [],
    'PORTRAIT_FIRST_MENTION': ['MAIN', 'MID_RUS', 'OLD_RUS'],
    'PORTRAIT_MEANING': [],
}

# Корпуса, где запрос прошёл (200 OK), но данные пусты - требуют перепроверки:
RESULTTYPE_CORPUS_EMPTY = {
    'PORTRAIT_WORD_INFO': ['BIRCHBARK'],
    'PORTRAIT_CONCORDANCE': ['BIRCHBARK'],
    'PORTRAIT_STATS': [],
    'PORTRAIT_SKETCH': ['BIRCHBARK'],
    'PORTRAIT_FREQUENCY': ['BIRCHBARK'],
    'PORTRAIT_SIMILAR': ['BIRCHBARK', 'GICR', 'OLD_RUS', 'SPOKEN'],
    'PORTRAIT_MORPHEME': ['BIRCHBARK'],
    'PORTRAIT_WORDFORMS': ['BIRCHBARK', 'CLASSICS', 'GICR', 'KIDS', 'MID_RUS', 'SPOKEN'],
    'PORTRAIT_COGNATES': ['BIRCHBARK', 'CLASSICS', 'GICR', 'KIDS', 'MAIN', 'MID_RUS', 'OLD_RUS', 'SPOKEN'],
    'PORTRAIT_FIRST_MENTION': ['BIRCHBARK', 'BLOGS', 'CLASSICS', 'GICR', 'KIDS', 'POETIC', 'SPOKEN'],
    'PORTRAIT_MEANING': ['BIRCHBARK', 'CLASSICS', 'GICR', 'KIDS', 'MAIN', 'MID_RUS', 'OLD_RUS', 'SPOKEN'],
}

# Корпуса, где запрос упал с ошибкой (HTTP 4xx/5xx или сеть):
RESULTTYPE_CORPUS_ERRORS = {
    'PORTRAIT_WORD_INFO': [],
    'PORTRAIT_CONCORDANCE': [],
    'PORTRAIT_STATS': ['BIRCHBARK', 'OLD_RUS'],
    'PORTRAIT_SKETCH': ['BLOGS', 'POETIC'],
    'PORTRAIT_FREQUENCY': ['BLOGS', 'POETIC'],
    'PORTRAIT_SIMILAR': ['BLOGS', 'POETIC'],
    'PORTRAIT_MORPHEME': ['BLOGS', 'CLASSICS', 'GICR', 'KIDS', 'MID_RUS', 'OLD_RUS', 'POETIC', 'SPOKEN'],
    'PORTRAIT_WORDFORMS': ['BLOGS', 'POETIC'],
    'PORTRAIT_COGNATES': ['BLOGS', 'POETIC'],
    'PORTRAIT_FIRST_MENTION': [],
    'PORTRAIT_MEANING': ['BLOGS', 'POETIC'],
}