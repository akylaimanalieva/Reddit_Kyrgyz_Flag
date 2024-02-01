# Importing libraries
from libraries import (
    pd, praw, BERTopic, TextBlob, stopwords, word_tokenize, plt,
    WordCloud, Counter, TextClassifier, Sentence, LdaModel, Dictionary,
    WordCloud, nx, nltk, WordNetLemmatizer
)

# Reddit API credentials
reddit_client_id = 'client_id'
reddit_client_secret = 'client_secret'
reddit_user_agent = 'user_agent'

# Function to get Reddit comments from a given URL
def get_reddit_comments(url):
    reddit = praw.Reddit(client_id=reddit_client_id,
                         client_secret=reddit_client_secret,
                         user_agent=reddit_user_agent)

    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=None)  # Replace all MoreComments
    comments = []

    for comment in submission.comments.list():
        comments.append(comment.body)

    return comments

# Advanced text preprocessing
def preprocess_text_advanced(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmatized_tokens)

# Function to perform topic modeling using Gensim's LDA
def perform_topic_modeling_lda(comments):
    preprocessed_comments = [preprocess_text_advanced(comment) for comment in comments]
    tokenized_comments = [word_tokenize(comment) for comment in preprocessed_comments]

    dictionary = Dictionary(tokenized_comments)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_comments]

    lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

    topics = lda_model.print_topics(num_words=5)

    return lda_model, topics

# Function to analyze sentiment using Flair
def analyze_sentiment_flair(comments):
    classifier = TextClassifier.load('sentiment')
    sentiment_scores = []

    for comment in comments:
        sentence = Sentence(comment)
        classifier.predict(sentence)
        sentiment = sentence.labels[0].value
        sentiment_scores.append(1 if sentiment == 'POSITIVE' else -1 if sentiment == 'NEGATIVE' else 0)

    return sentiment_scores

# Function to visualize emotions using matplotlib
def visualize_emotions(sentiment_scores):
    positive_count = sum(1 for score in sentiment_scores if score > 0)
    negative_count = sum(1 for score in sentiment_scores if score < 0)

    labels = ['Positive', 'Negative']
    values = [positive_count, negative_count]

    plt.bar(labels, values, color=['green', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Emotions from Reddit Comments')
    plt.show()

# Function to get most frequently used words and their counts
def get_most_frequent_words(comments, top_n=40):
    words = [word for comment in comments for word in word_tokenize(comment)]
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

# Function to visualize topics using a bar chart
def visualize_topics(lda_model, topics, num_words=5):
    topics_data = []
    for topic_id, topic in topics:
        words = [word.split("*")[1].strip('"') for word in topic.split(" + ")]
        topics_data.append((topic_id, words))

    plt.figure(figsize=(12, 8))
    topics_data = sorted(topics_data, key=lambda x: x[0])  # Sort topics by topic_id

    word_labels = set(word for _, words in topics_data for word in words)

    for topic_id, words in topics_data:
        values = [float(word.split("*")[0]) if word.split("*")[0].replace(".", "").isdigit() else 0.0 for word in words]
        plt.barh(f'Topic {topic_id}', values, label=f'Topic {topic_id}', alpha=0.7)

    plt.xlabel('Probability')
    plt.title('Top Words in Each Topic')
    plt.yticks(range(len(topics_data)), [f'Topic {topic_id}' for topic_id, _ in topics_data])
    plt.legend(loc='best', bbox_to_anchor=(1, 1))

    # Adding word labels to the right of the bars
    for i, (_, words) in enumerate(topics_data):
        max_value = max([float(word.split("*")[0]) if word.split("*")[0].replace(".", "").isdigit() else 0.0 for word in words])
        for j, word in enumerate(words):
            plt.text(max_value + 0.01, i + j / len(words), f'{word}', va='center', ha='left', fontsize=8)

    plt.tight_layout()
    plt.show()  

# Function to visualize word cloud
def visualize_wordcloud(word_counts):
    wordcloud = WordCloud(width=1000, height=500, max_words=300, background_color='white').generate_from_frequencies(word_counts)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Most Frequent Words')
    plt.show()

# URLs of Reddit threads
urls = [
    'https://www.reddit.com/r/vexillology/comments/185sk2m/kyrgyzstan_passed_legislation_to_change_flag_from/',
    'https://www.reddit.com/r/vexillology/comments/18mresb/kyrgyzstan_redesigned_its_flag/',
    'https://www.reddit.com/r/vexillology/comments/1923xbd/new_national_flag_of_kyrgyzstan_kyrgyz_republic/'
]

# Comments from Reddit threads
all_comments = []
for url in urls:
    all_comments.extend(get_reddit_comments(url))

# Most frequently used words and their counts
most_frequent_words = get_most_frequent_words(all_comments)
print("Most frequently used words and their counts:")
for word, count in most_frequent_words:
    print(f"{word}: {count}")

# Topic modeling using Gensim's LDA
topic_model_lda, topics_lda = perform_topic_modeling_lda(all_comments)

# Visualizing topics using Gensim's LDA
visualize_topics(topic_model_lda, topics_lda, num_words=5)

# Visualizing word cloud for most frequent words
word_counts = dict(most_frequent_words)
visualize_wordcloud(word_counts)

# Analyzing sentiment using Flair
sentiment_scores_flair = analyze_sentiment_flair(all_comments)

# Visualizing emotions
visualize_emotions(sentiment_scores_flair)