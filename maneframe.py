import numpy as np
import os 
#import enchant
from nltk.corpus import words
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from datetime import datetime

# HELPPER FUNCTIONS

# Given a directory path, get all of the file names and join the path to create an absolute path
def GetListOfFileNames(dir_path):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


# Calculate a the projection of vector a onto b, where both a and b are scipy sprase matrix
def VectorProjection(a, b):
    numerator = np.dot(a, b)
    #print(numerator.shape)
    denominator = np.dot(b.T, b)
    #print(denominator.shape)
    frac = numerator/denominator
    return np.multiply(b, frac)


def Distance(a, b):
    #print(f"a: {a.shape}")
    #print(f"b: {b.shape}")
    diff = a - b
    sum_square = np.dot(diff, diff.T)
    #print(f"sum_square: {sum_square.shape}")
    return np.sqrt(sum_square)


# Global constant variables
CURRENT_DIRECTORY = os.getcwd()
DIR_OF_TEXT = os.path.join(CURRENT_DIRECTORY, "en_data/text")
FILE_PATH_TEXTS = GetListOfFileNames(DIR_OF_TEXT)
STOP_WORDS = stopwords.words('english')
#EN_DICT = enchant.Dict("en_US")

POLAR_WORD_SETS =[
    ["negative", "positive"],
    ["displeased", "pleased"],
    ["disapproving", "approving"],
    ["disliking", "liking"],
    ["fear", "hope"],
    ["distress", "joy"],
    ["shame", "pride"],
    ["reproach", "admiration"],
    ["hate", "love"],
    ["disgust", "interest"],
    ["fears-confirmed", "satisfaction"],
    ["disappointment", "relief"],
    ["resentment", "happy-for"],
    ["pity", "gloating"]
]

polar_word_index =[
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1]
]


# Init document-token matrix
embedding_sets = {
    "count_vectorizer": text.CountVectorizer,
    "tfidf_vectorizer": text.TfidfVectorizer,
    "hashing_vectorizer": text.HashingVectorizer
}


CURRENT_SET = "count_vectorizer"


vectorizer_spgc = embedding_sets[CURRENT_SET](  input="filename", 
                                    encoding='utf-8',
                                    decode_error='ignore',
                                    strip_accents='unicode',
                                    lowercase=True,
                                    stop_words=STOP_WORDS,
                                    ngram_range = (1, 1),
                                    analyzer='word',
                                    dtype=np.float64)

# Construct document-token matrix, then transpose for token-document matrix
count_embeding = vectorizer_spgc.fit_transform(FILE_PATH_TEXTS)
count_embeding = count_embeding.transpose()

# At this point, do some clean up so we can make some room to calculate tocken positions
FILE_PATH_TEXTS.clear()


print(f"term_freq_document shape: {count_embeding.shape}")

# From the vocabulary created by CountVectorizer, find the token index. If not found, keep -1
feature_list = vectorizer_spgc.get_feature_names_out()
for word_set_idx in range(len(POLAR_WORD_SETS)):
    for word_idx in range(len(POLAR_WORD_SETS[word_set_idx])):
        for (index, item) in enumerate(feature_list):
            if item == POLAR_WORD_SETS[word_set_idx][word_idx]:
                print(f"{item} at index {index}")
                polar_word_index[word_set_idx][word_idx] = index
                break
print(polar_word_index)


# TODO: Need to run this through all polar tokens
start=datetime.now()
for word_set_idx in range(len(POLAR_WORD_SETS)):
    negative_idx = polar_word_index[word_set_idx][0] 
    positive_idx = polar_word_index[word_set_idx][1]
    negative_str = POLAR_WORD_SETS[word_set_idx][0]
    positive_str = POLAR_WORD_SETS[word_set_idx][1]
    
    if negative_idx == -1 or positive_idx == -1:
        continue 

    print(f"Calculating SemAxis: {negative_str}_{positive_str}")
    
    negative_point = count_embeding[negative_idx].toarray()
    positive_point = count_embeding[positive_idx].toarray()
    #print(negative_point.shape)
    #print(positive_point.shape)

    pos_neg_vector = positive_point - negative_point
    semaxis_norm = pos_neg_vector / np.sqrt(np.sum(pos_neg_vector**2))
    midpoint = pos_neg_vector/2.0

    max_dist = 0.0
    token_distances = dict()
    num_words = 0
    for idex in range(count_embeding.shape[0]):

        if idex % 100000 == 0:
            print(f"Processed: {idex} tokens, found {num_words} words after {datetime.now()-start}")

        token_string = feature_list[idex]

        if any(char.isdigit() for char in token_string):
            continue

        #if not EN_DICT.check(token_string):
        #    continue
        if not token_string in words.words():
            continue

        projected_point = VectorProjection(count_embeding[idex].toarray(), semaxis_norm.T)
        #print(f"midpoint: {midpoint.shape}")
        #print(f"projected_point: {projected_point.shape}")
        dot_product = np.dot(projected_point.T, midpoint.T)
        #print(f"dot_product: {dot_product.shape}")
        #print(f"projected_point.T: {projected_point.T.shape}")
        #print(f"midpoint: {midpoint.shape}")
        dist = Distance(projected_point.T, midpoint)

        if max_dist < dist:
            max_dist = dist

        if dot_product < 0.0:
            dist = -dist

        token_distances[token_string] = dist
        num_words += 1
        #print(f"{token_string},{token_distances[token_string]}")
        


    for key in token_distances:
        token_distances[key] = token_distances[key]/max_dist


    dir_for_set = os.path.join(CURRENT_DIRECTORY, f"en_data/{CURRENT_SET}/")

    if not os.path.isdir(dir_for_set):
        os.mkdir(dir_for_set) 

    dir_for_polar = os.path.join(dir_for_set, f"{negative_str}_{positive_str}.csv")

    with open(dir_for_polar, 'w') as f:
        f.write(f"token, distance\n")
        for key in token_distances.keys():
            f.write(f"{key},{token_distances[key]}\n")