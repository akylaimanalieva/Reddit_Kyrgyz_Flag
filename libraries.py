# libraries.py
import pandas as pd
import praw
from bertopic import BERTopic
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from flair.models import TextClassifier
from flair.data import Sentence
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from wordcloud import WordCloud
import networkx as nx
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer