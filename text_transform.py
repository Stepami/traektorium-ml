import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

def clean_word(word):
    if re.match(r'[a-zа-я]+', word) and re.search(r'[а-я]+', word):
        return re.sub(r'[a-z]+', '', word)
    return word

def stem_word(ru_morph, en_stemmer, word):
    if re.match(r'[a-z]+', word):
        return en_stemmer.lemmatize(word)
    return ru_morph.parse(word)[0].normal_form

def process_text(lines):
    russian_stopwords = stopwords.words("russian")
    english_stopwords = stopwords.words("english")

    ru_morph = pymorphy2.MorphAnalyzer()
    en_stemmer = WordNetLemmatizer()

    for i, line in tqdm(enumerate(lines)):
        lines[i] = re.sub(r'[^a-zа-я\s]', '', line.lower())
        words = [
            stem_word(ru_morph, en_stemmer, clean_word(word)) \
                for word in lines[i].split() \
                    if word not in russian_stopwords \
                        and word not in english_stopwords
        ]
        lines[i] = ' '.join(words)

    return lines