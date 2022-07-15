#======================================================================================================================
# IMPORTS
#======================================================================================================================

import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import csv, re
import string
from tqdm import tqdm
import codecs
from collections import Counter
from spacy.lang.en import English

#======================================================================================================================
# CONSTANTS
#======================================================================================================================

# Hard-wired variables
input_speechfile   = "./wcpr_mypersonality.csv"
stopwords_file     = "./mallet_en_stoplist.txt"

#======================================================================================================================
# FUNCTIONS
#======================================================================================================================

# This is the similar to read_and_clean_lines in the previous assignment, but
# rather than just returning a list of cleaned lines of text, we should return
# returns two lists (of the same length): the cleaned lines and the party of the person who was speaking
#
# Make sure to replace line-internal whitespace (newlines, tabs, etc.) in text with a space.
#
# For information on how to read from a gzipped file, rather than uncompressing and reading, see
# https://stackoverflow.com/questions/10566558/python-read-lines-from-compressed-text-files#30868178
#
# For info on parsing jsonlines, see https://www.geeksforgeeks.org/json-loads-in-python/.
# (There are other ways of doing it, of course.)
#
# Specify in the function call the chamber that is being analyzed, if none are specified the function
#  will utilize all the documents fom the House and Senate
def read_and_clean_lines(infile):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    neurotic_flags = []

    with open(infile, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            lines.append(row[1])
            neurotic_flags.append(row[8])

    print("Read {} status posts.".format(len(lines)))
    print("Read {} labels".format(len(neurotic_flags)))
    return lines, neurotic_flags

# Read a set of stoplist words from filename, assuming it contains one word per line
# Return a python Set data structure (https://www.w3schools.com/python/python_sets.asp)
def load_stopwords(filename):
    stopwords = []
    with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
        stopwords = fp.read().split('\n')
    return set(stopwords)


# Call sklearn's train_test_split function to split the dataset into training items/labels
# and test items/labels.  See https://realpython.com/train-test-split-python-data/
# (or Google train_test_split) for how to make this call.
#
# Note that the train_test_split function returns four sequences: X_train, X_test, y_train, y_test
# X_train and y_train  are the training items and labels, respectively
# X_test  and y_test   are the test items and labels, respectively
#
# This function should return those four values
def split_training_set(lines, labels, test_size=0.2, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=test_size, shuffle=False)
    print("Training set label counts: {}".format(Counter(y_train)))
    print("Test set     label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test

# Converting text into features.
# Inputs:
#    X - a sequence of raw text strings to be processed
#    analyzefn - either built-in (see CountVectorizer documentation), or a function we provide from strings to feature-lists
#
#    Arguments used by the words analyzer
#      stopwords - set of stopwords (used by "word" analyzer")
#      lowercase - true if normalizing by lowercasing
#      ngram_range - (N,M) for using ngrams of sizes N up to M as features, e.g. (1,2) for unigrams and bigrams
#
#  Outputs:
#     X_features - corresponding feature vector for each raw text item in X
#     training_vectorizer - vectorizer object that can now be applied to some new X', e.g. containing test texts
#    
# You can find documentation at https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# and there's a nice, readable discussion at https://medium.com/swlh/understanding-count-vectorizer-5dd71530c1b
#
def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

# Input:
#    lines     - a raw text corpus, where each element in the list is a string
#    stopwords - a set of strings that are stopwords
#    remove_stopword_bigrams = True or False
#
# Output:  a corresponding list converting the raw strings to space-separated features
#
# The features extracted should include non-stopword, non-punctuation unigrams,
# plus the bigram features that were counted in collect_bigram_counts from the previous assignment
# represented as underscore_separated tokens.
# Example:
#   Input:  ["This is Remy's dinner.",
#            "Remy will eat it."]
#   Output: ["remy 's dinner remy_'s 's_dinner",
#            "remy eat"]
def convert_lines_to_feature_strings(lines, stopwords, remove_stopword_bigrams=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
        
    print(" Initializing")
    nlp          = English(parser=False)
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):
        
        # Get spacy tokenization and normalize the tokens
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams          = [token   for token in normalized_tokens
                                 if token not in stopwords and token not in string.punctuation]

        # Collect string bigram tokens as features
        bigrams = []
        bigram_tokens     = ["_".join(bigram) for bigram in bigrams]
        bigrams           = ngrams(normalized_tokens, 2) 
        bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'

        # feature_string = []
        feature_string = " ".join(unigrams) + " " + " ".join(bigram_tokens)

        # Add this feature string to the output
        all_features.append(feature_string)

        
    return all_features

# Split on whitespace, e.g. "a    b_c  d" returns tokens ['a','b_c','d']
def whitespace_tokenizer(line):
    return line.split()

def normalize_tokens(tokenlist):
    # Input: list of tokens as strings,  e.g. ['I', ' ', 'saw', ' ', '@psresnik', ' ', 'on', ' ','Twitter']
    # Output: list of tokens where
    normalized_tokens = [token.lower().replace('_','+') for token in tokenlist   # lowercase, _ => +
                             if re.search('[^\s]', token) is not None            # ignore whitespace tokens
                             and not token.startswith("@")                       # ignore  handles
                        ]
    return normalized_tokens        

# Take a list of string tokens and return all ngrams of length n,
# representing each ngram as a list of  tokens.
# E.g. ngrams(['the','quick','brown','fox'], 2)
# returns [['the','quick'], ['quick','brown'], ['brown','fox']]
# Note that this should work for any n, not just unigrams and bigrams
def ngrams(tokens, n):
    # Returns all ngrams of size n in sentence, where an ngram is itself a list of tokens
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]

def filter_punctuation_bigrams(ngrams):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams
    # Removes ngrams like ['today','.'] where either token is a punctuation character
    # Returns list with the items that were not removed
    punct = string.punctuation
    return [ngram   for ngram in ngrams   if ngram[0] not in punct and ngram[1] not in punct]

def filter_stopword_bigrams(ngrams, stopwords):
    result = [ngram   for ngram in ngrams   if ngram[0] not in stopwords and ngram[1] not in stopwords]
    return result
        
#======================================================================================================================
# MAIN
#======================================================================================================================
    
def main():
    # Get stop words in a list
    stop_words = load_stopwords(stopwords_file)

    # Read the dataset in and split it into training documents/labels (X) and test documents/labels (y)
    X_train, X_test, y_train, y_test = split_training_set(*read_and_clean_lines(input_speechfile))
    
    # Roll your own feature extraction.
    # Call convert_lines_to_feature_strings() to get your features
    # as a whitespace-separated string that will now represent the document.
    print("Creating feature strings for training data")
    X_train_feature_strings = convert_lines_to_feature_strings(X_train, stop_words)
    print("Creating feature strings for test data")
    X_test_documents        = convert_lines_to_feature_strings(X_test,  stop_words)
    
    # Call CountVectorizer with whitespace-based tokenization as the analyzer, so that it uses exactly your features,
    # but without doing any of its own analysis/feature-extraction.
    X_features_train, training_vectorizer = convert_text_into_features(X_train_feature_strings, stop_words, whitespace_tokenizer)
        
    # Create a logistic regression classifier trained on the featurized training data
    lr_classifier = LogisticRegression(solver='liblinear')
    lr_classifier.fit(X_features_train, y_train)

    # Apply the "vectorizer" created using the training data to the test documents, to create testset feature vectors
    X_test_features =  training_vectorizer.transform(X_test_documents)

    # Classify the test data and see how well you perform
    # For various evaluation scores see https://scikit-learn.org/stable/modules/model_evaluation.html
    print("Classifying test data...")
    predicted_labels = lr_classifier.predict(X_test_features)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test)))
    for label in ['n', 'y']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test, pos_label=label)))
    
    # Graph the confusion matrix to show accuracies
    print("Generating plots...")
    metrics.plot_confusion_matrix(lr_classifier, X_test_features, y_test, normalize='true')
    plt.show()

#======================================================================================================================
# END
#======================================================================================================================

if __name__ == "__main__":
    main()

