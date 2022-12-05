import re
import nltk
import spacy
import unidecode
from tqdm import tqdm

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
porter = PorterStemmer()
tokenizer = ToktokTokenizer()
stopword_list = stopwords.words('english') #  Stopwords must be downloaded using " nltk.download('stopwords')" once in jupyternotebook, the comment or delete the line
nlp = spacy.load('en_core_web_sm') # 'en_core_web_sm' must be downloaded using "python -m spacy download en_core_web_sm" once in jupyternotebook, the comment or delete the line

def remove_html_tags(text):
    text_without_tags = re.sub('<[^<]+?>', '', text)
    return text_without_tags 


def remove_extra_new_lines(text):
    text_removed_extra_lines = ' '.join(text.splitlines())
    return text_removed_extra_lines


def remove_accented_chars(text):
    text_no_accented = unidecode.unidecode(text)
    return text_no_accented


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    for contraction in CONTRACTION_MAP.keys():
        if contraction in text:
            text = text.replace(contraction, CONTRACTION_MAP[contraction])
    return text


def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_words = []
    for token in doc:
        lemmatized_words.append(token.lemma_)
    lemmatized_text = ' '.join(map(str,lemmatized_words))
    return lemmatized_text


def stem_text(text):
    stemmed_text= []
    tokenized_text = tokenizer.tokenize(text)
    for word in tokenized_text:
        stemmed_word= porter.stem(word)
        stemmed_text.append(stemmed_word)
    stemmed_text = ' '.join(map(str,stemmed_text))   
    return stemmed_text


def remove_special_chars(text, remove_digits=False):
    text_no_special_chars = re.sub(r"[^a-zA-Z ]", "", text)
    return text_no_special_chars


def remove_extra_whitespace(text):
    text_no_whitespace = " ".join(text.split())
    return text_no_whitespace


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokenized_text = tokenizer.tokenize(text.lower())
    clean_words = []
    for word in tokenized_text:
        if word not in stopword_list:
            clean_words.append(word)
    text_no_stop_words = ' '.join(map(str,clean_words))
    return text_no_stop_words


def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=True, # Starting with true
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in tqdm(corpus):
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
