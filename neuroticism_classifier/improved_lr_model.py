#======================================================================================================================
# IMPORTS
#======================================================================================================================

import csv, re
import codecs
import string
from datetime import datetime
from math import floor, ceil
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from spacy.lang.en import English
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from textblob import Word

#======================================================================================================================
# CONSTANTS
#======================================================================================================================

FEATURES_FILE = './wcpr_mypersonality.csv'
STOPWORD_FILE = './mallet_en_stoplist.txt'
NGRAMS = [1,2,3] # Takes up to 4

#======================================================================================================================
# FUNCTIONS
#======================================================================================================================

# Returns the hour of day from a datetime string
def get_hour(datetime_str):
    timestamp = datetime.strptime(datetime_str, '%m/%d/%y %I:%M %p')
    time_of_day_in_minutes = (timestamp.hour * 60) + timestamp.minute
    return time_of_day_in_minutes / (24 * 60)

# Extracts status information from csv and returns them in a list such that the first 80% of the list are instances
# of 80% of each users total statuses, and the last 20% of the list are instances of 20% of each users total statuses
def divide_by_user():
    with open(FEATURES_FILE, 'r') as f:
        reader = csv.reader(f)
        users = []

        # Ignore header
        next(reader, None)
        for row in reader:
            text = re.sub(r'[\s]+', ' ', row[1])    # Removing whitespace
            text = re.sub(r'["-=+]+', ' ', text)    # Removing symbols
            time_range = get_hour(row[12])

            if (wrapped_i := [i for i, user in enumerate(users) if user['auth_id'] == row[0]]):
                i = wrapped_i[0]
                users[i]['entries'].append(text)
                users[i]['time'].append(time_range)
            else:
                users.append({
                    'auth_id': row[0],
                    'entries': [text],
                    'is_ext': 0 if row[7] == 'n' else 1,
                    'is_agr': 0 if row[9] == 'n' else 1,
                    'is_con': 0 if row[10] == 'n' else 1,
                    'is_opn': 0 if row[11] == 'n' else 1,
                    'is_neu': row[8],
                    'time': [time_range]
                })

        # Dividing by user
        posts_train = []
        posts_test = []
        label_train = []
        label_test = []
        for user in users:
            is_ext = user['is_ext']
            is_neu = user['is_neu']
            is_agr = user['is_agr']
            is_con = user['is_con']
            is_opn = user['is_opn']

            entries_and_day_phases = list(zip(user['entries'], user['time']))            
            if (post_count := len(user['entries'])) > 4:
                train_limit = ceil(post_count * 0.8)
                posts_train += [ [entry, is_ext, is_agr, is_con, is_opn, time] for entry, time in entries_and_day_phases[:train_limit] ]
                posts_test += [ [entry, is_ext, is_agr, is_con, is_opn, time] for entry, time in entries_and_day_phases[train_limit:] ]
                label_train += ([is_neu] * train_limit)
                label_test += ([is_neu] * (post_count - train_limit))
            else:
                posts_test += [ [entry, is_ext, is_agr, is_con, is_opn, time] for entry, time in entries_and_day_phases ]
                label_test += ([is_neu] * post_count)

        return (posts_train + posts_test), (label_train + label_test)

# Load stop words list
def load_stopwords(filename):
    stopwords = []
    with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
        stopwords = fp.read().split('\n')
    return set(stopwords)

# Converts lines of text to ngram feature strings
# Always creates unigrams and bigrams, but can optionally create 3-grams and 4-grams
def convert_lines_to_feature_strings(lines, stopwords, remove_elongated, remove_stopword_ngrams=True, applied_ngrams=[2, 3, 4]):
    print(" Converting from raw text to ngram features")
    if remove_stopword_ngrams:
        print(" Includes filtering stopword ngrams")
        
    print(" Initializing")
    nlp          = English(parser=False)
    all_features = []
    print(" Iterating through documents extracting ngram features")

    stemmer = PorterStemmer()

    for line in tqdm(lines):
        
        # Get spacy tokenization and normalize the tokens
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)
        if remove_elongated:
            normalized_tokens = [clean_elongated_word(token) if token not in stopwords else token for token in normalized_tokens]
            normalized_tokens = [stemmer.stem(token) if token not in stopwords else token for token in normalized_tokens]

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams          = [token   for token in normalized_tokens if token not in stopwords and token not in string.punctuation]

        # Collect string bigram tokens as features
        bigrams = []
        bigram_tokens     = ["_".join(bigram) for bigram in bigrams]
        bigrams           = ngrams(normalized_tokens, 2) 
        bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_ngrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        # Collect string trigram tokens as features
        trigrams = []
        trigram_tokens    = ["_".join(trigram) for trigram in trigrams]
        trigrams          = ngrams(normalized_tokens, 3)
        trigrams          = filter_punctuation_trigrams(trigrams)
        if remove_stopword_ngrams:
            trigrams = filter_stopword_trigrams(trigrams, stopwords)
        trigram_tokens = ["_".join(trigram) for trigram in trigrams]

        # Collect string tetragram tokens as features
        tetragrams = []
        tetragram_tokens  = ["_".join(tetragram) for tetragram in tetragrams]
        tetragrams        = ngrams(normalized_tokens, 4)
        tetragrams        = filter_punctuation_tetragrams(tetragrams)
        if remove_stopword_ngrams:
            tetragrams = filter_stopword_tetragrams(tetragrams, stopwords)
        tetragram_tokens = ["_".join(tetragram) for tetragram in tetragrams]

        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'
        feature_string = " ".join(unigrams) + " " + " ".join(bigram_tokens)
        if 3 in applied_ngrams:
            feature_string += " " + " ".join(trigram_tokens)
        if 4 in applied_ngrams:
            feature_string += " " + " ".join(tetragram_tokens)

        # Add this feature string to the output
        all_features.append(feature_string)
        
    return all_features

# Converts feature strings to a bag of words representation
def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,1)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                            analyzer=analyzefn,
                                            lowercase=True,
                                            ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

# Normalizes the tokens
def normalize_tokens(tokenlist):
    normalized_tokens = [token.lower().replace('_','+') for token in tokenlist   # lowercase, _ => +
                             if re.search('[^\s]', token) is not None            # ignore whitespace tokens
                             and not token.startswith("@")                       # ignore  handles
                        ]
    return normalized_tokens        

def whitespace_tokenizer(line):
    return line.split()

# Returns all ngrams of size n in sentence, where an ngram is itself a list of tokens
def ngrams(tokens, n):
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]

# Filters values with punctionation tokens for bigrams and returns the items that were not removed
def filter_punctuation_bigrams(ngrams):
    punct = string.punctuation
    return [ngram   for ngram in ngrams   if ngram[0] not in punct and ngram[1] not in punct]

# Filters values with punctionation tokens for 3-grams and returns the items that were not removed
def filter_punctuation_trigrams(ngrams):
    punct = string.punctuation
    return [ngram for ngram in ngrams if ngram[0] not in punct and ngram[1] not in punct and ngram[2] not in punct]

# Filters values with punctionation tokens for 4-grams and returns the items that were not removed
def filter_punctuation_tetragrams(ngrams):
    punct = string.punctuation
    return [ngram for ngram in ngrams if ngram[0] not in punct and ngram[1] not in punct and ngram[2] not in punct and ngram[3] not in punct]

# Filters values with stop words for bigrams and returns the items that were not removed
def filter_stopword_bigrams(ngrams, stopwords):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams, stopwords is a set of words like 'the'
    # Removes ngrams like ['in','the'] and ['senator','from'] where either word is a stopword
    # Returns list with the items that were not removed
    result = [ngram   for ngram in ngrams   if ngram[0] not in stopwords and ngram[1] not in stopwords]
    return result

# Filters values with stop words for 3-grams and returns the items that were not removed
def filter_stopword_trigrams(ngrams, stopwords):
    result = [ngram for ngram in ngrams if ngram[0] not in stopwords and ngram[1] not in stopwords and ngram[2] not in stopwords]
    return result

# Filters values with stop words for 4-grams and returns the items that were not removed
def filter_stopword_tetragrams(ngrams, stopwords):
    result = [ngram for ngram in ngrams if ngram[0] not in stopwords and ngram[1] not in stopwords and ngram[2] not in stopwords and ngram[3] not in stopwords]
    return result

# Recieves a string that is returned if identified as a word and if not, it is shortened until it is identified as one
# If no word can be found, the original string is returned
def clean_elongated_word(word):
    original_word = word
    tb_word = Word(word)
    match_found = bool([spellcheck for spellcheck in tb_word.spellcheck() if spellcheck[0] == word and spellcheck[1]])

    if match_found:
        return original_word

    letter_pos = len(word)
    while not match_found:
        # Break if letter_pos - 1 = 0
        if not (letter_pos - 1):
            break

        curr_letter = word[letter_pos - 1]
        prev_letter = word[letter_pos - 2]

        if curr_letter == prev_letter:
            word = word[:letter_pos - 2] + word[letter_pos - 1:]
            tb_word = Word(word)
            match_found = bool([spellcheck for spellcheck in tb_word.spellcheck() if spellcheck[0] == word and spellcheck[1]])
            if match_found:
                break
        
        letter_pos -= 1
    
    return tb_word if match_found else original_word

#======================================================================================================================
# MAIN
#======================================================================================================================

def main(remove_elongated):
    # Retrieving dataset with the 80-20 split per user statuses already factored in
    print('Getting user data...')
    raw_X, raw_y = divide_by_user()
    df = pd.DataFrame(raw_X, columns=['STATUS', 'isEXT', 'isAGR', 'isCON', 'isOPN', 'time'])

    # Retrieving stop words list
    stop_words = load_stopwords(STOPWORD_FILE)

    # Converting status texts to bag of words representation
    STATUS = convert_lines_to_feature_strings(df.STATUS, stop_words, remove_elongated, applied_ngrams=NGRAMS)
    STATUS, vectorizer = convert_text_into_features(STATUS, stop_words, whitespace_tokenizer)

    # Creating a new data frame comprised of the bag of words and other values for the X
    print('Organizing dataset...')
    STATUS = pd.DataFrame(STATUS.todense(), columns=vectorizer.get_feature_names())
    isEXT = df.isEXT
    isAGR = df.isAGR
    isCON = df.isCON
    isOPN = df.isOPN
    time = df.time
    X = pd.concat([STATUS, isEXT, isAGR, isCON, isOPN, time], axis='columns')

    # Gettting the labels
    y = pd.DataFrame(raw_y, columns=['isNEU'])

    print('Splitting train and test data...')
    
    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Creating model
    print('Creating Logistic Regression Model...')
    lr_classifier = LogisticRegression(solver='liblinear')
    lr_classifier.fit(X_train, y_train.values.ravel())

    # Testing model
    print('Testing model...')
    predicted_labels = lr_classifier.predict(X_test)

    # Calculating and showing accuracy values
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test)))
    for label in ['n', 'y']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test, pos_label=label)))

    # Generating plots
    print("Generating plots...")
    metrics.plot_confusion_matrix(lr_classifier, X_test, y_test, normalize='true')
    plt.show()

#======================================================================================================================
# END
#======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options for running this script')
    parser.add_argument('--remove_elongated', default=False, action='store_true', help="Use elongated words from documents.")
    args = parser.parse_args()
    main(args.remove_elongated)