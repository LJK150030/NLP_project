import scipy.sparse
import numpy as np
import os 
import enchant
from sklearn.feature_extraction import text
from sklearn.preprocessing import normalize
from sklearn import metrics
from nltk.corpus import stopwords
from scipy.spatial import distance
from scipy.special import softmax
from scipy import sparse
from sklearn.preprocessing import normalize
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
    
def GeneralNorm(x, min_in, max_in, min_out, max_out):
    denominator = max_in - min_in
    if denominator == 0.0:
        return 0.0
    
    numerator = (x - min_in) * (max_out - min_out)
    frac = numerator/denominator
    result = frac + min_out
    #print(f"{result} = {numerator}/{denominator} + {min_out}")
    return result
    

# Global constant variables
CURRENT_DIRECTORY = os.getcwd()
STOP_WORDS = stopwords.words('english')
EN_DICT = enchant.Dict("en_US")

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
    ['remorse', 'gratification'],
    ['anger', 'gratitude'],
    ["fears-confirmed", "satisfaction"],
    ["disappointment", "relief"],
    ["resentment", "happy-for"],
    ["pity", "gloating"]
]

polar_word_index = np.full((16, 2), -1, dtype=np.int_)

# Init document-token matrix
embedding_sets = {
    "count_vectorizer": text.CountVectorizer,
    "tfidf_vectorizer": text.TfidfVectorizer
    #"hashing_vectorizer": text.HashingVectorizer
}

CALCULATION = [
    "projection",
    "dot_product"
]

for embed in embedding_sets:
    for calc in CALCULATION:
        # ---------------------------------------------------------------------------------------
        # Change these variables for the specific procedures, and where to find the documents
        CURRENT_SET = embed
        CURRENT_CALC = calc
        DIR_OF_TEXT = os.path.join(CURRENT_DIRECTORY, "en_data/spgc")
        FILE_PATH_TEXTS = GetListOfFileNames(DIR_OF_TEXT)
        # ---------------------------------------------------------------------------------------


        # Hour long computation, only do if necessary...

        print(f"Finding all valid tokens in Documents...")
        vectorizer_spgc = embedding_sets[CURRENT_SET](  input="filename", 
                                            encoding='utf-8',
                                            decode_error='ignore',
                                            strip_accents='unicode',
                                            lowercase=True,
                                            stop_words=STOP_WORDS,
                                            ngram_range = (1, 1),
                                            analyzer='word',
                                            dtype=np.float32)

        # Construct document-token matrix, then transpose for token-document matrix
        count_embeding = vectorizer_spgc.fit_transform(FILE_PATH_TEXTS)

        print(f"term_freq_document shape: {count_embeding.transpose().shape}")
        feature_list = vectorizer_spgc.get_feature_names_out()

        print(f"Finding valid words from tokenizer...")
        sub_word_list = list()
        for token in feature_list:
            if any(char.isdigit() for char in token):
                continue
            if EN_DICT.check(token):
                sub_word_list.append(token)

        fraction_of_words = len(sub_word_list)/len(feature_list)

        print(f"From the {len(feature_list)} tokens, only {len(sub_word_list)} are words. A {(1.0 - fraction_of_words) * 100.0}% reduction")

        vocab_dict = {word.lower(): idx for idx, word in enumerate(sub_word_list)}

        print(f"Using words as vocabulary for embedding step...")
        vectorizer_spgc = embedding_sets[CURRENT_SET](  input="filename", 
                                                        encoding='utf-8',
                                                        decode_error='ignore',
                                                        strip_accents='unicode',
                                                        lowercase=True,
                                                        stop_words=STOP_WORDS,
                                                        ngram_range = (1, 1),
                                                        analyzer='word',
                                                        vocabulary= vocab_dict,
                                                        dtype=np.float32)

        # Construct document-token matrix, then transpose for token-document matrix
        count_embeding = vectorizer_spgc.fit_transform(FILE_PATH_TEXTS)
        count_embeding = count_embeding.transpose()

        print(f"term_freq_document shape: {count_embeding.shape}")
        feature_list = vectorizer_spgc.get_feature_names_out()

        FILE_PATH_TEXTS.clear()


        # From the vocabulary created by CountVectorizer, find the token index. If not found, keep -1
        for word_set_idx in range(len(POLAR_WORD_SETS)):
            for word_idx in range(len(POLAR_WORD_SETS[word_set_idx])):
                for (index, item) in enumerate(feature_list):
                    if item == POLAR_WORD_SETS[word_set_idx][word_idx]:
                        print(f"{item} at index {index}")
                        polar_word_index[word_set_idx][word_idx] = index
                        break
        print(polar_word_index)


        polar_set_midpoints = np.zeros(shape=(len(POLAR_WORD_SETS), count_embeding.shape[1]), dtype=np.float32)
        semaxis_norms = np.zeros(shape=(len(POLAR_WORD_SETS), count_embeding.shape[1]), dtype=np.float32)
        polar_set_strings = list()


        for word_set_idx in range(len(POLAR_WORD_SETS)):
            negative_idx = polar_word_index[word_set_idx][0] 
            positive_idx = polar_word_index[word_set_idx][1]
            negative_str = POLAR_WORD_SETS[word_set_idx][0]
            positive_str = POLAR_WORD_SETS[word_set_idx][1]
            
            if negative_idx == -1 or positive_idx == -1:
                continue 
            
            negative_point = count_embeding[negative_idx].toarray()
            positive_point = count_embeding[positive_idx].toarray()
            pos_neg_vector = positive_point - negative_point
            semaxis_norm = pos_neg_vector / np.sqrt(np.sum(pos_neg_vector**2))
            midpoint = pos_neg_vector/2.0

            semaxis_norms[word_set_idx] = semaxis_norm
            polar_set_midpoints[word_set_idx] = midpoint[0]
            polar_set_strings.append(f"{negative_str}_{positive_str}")

        polar_set_midpoints = sparse.csr_matrix(polar_set_midpoints, dtype=np.float32)
        semaxis_norms = sparse.csr_matrix(semaxis_norms, dtype=np.float32)

        print(polar_set_midpoints.shape)
        print(semaxis_norms.shape)


        print(f"Normalizing count embeding...")
        count_embeding_normalized = normalize(count_embeding, norm='l2', axis=1)

        ## Calculate projected word position
        print(f"Calculating projected word...")
        print(f"...numerator calculation...")
        numerator = metrics.pairwise.cosine_similarity( X = count_embeding_normalized,
                                                        Y = semaxis_norms)
        numerator = sparse.csr_matrix(numerator)
        scalar = sparse.linalg.norm(x=count_embeding.transpose(), axis=0)
        scalar = np.resize(a=scalar, new_shape=(count_embeding.shape[0], len(POLAR_WORD_SETS)))
        print(f"numerator: {numerator.shape}")
        print(f"scalar: {scalar.shape}")

        print(f"...denominator calculation...")
        denominator = sparse.linalg.norm(   x=semaxis_norms,
                                            axis=1)

        #denominator = np.linalg.norm(x=semaxis_norms, axis=1)
        #denominator = np.resize(a=denominator, new_shape=(semaxis_norms.shape[1], len(POLAR_WORD_SETS)))

        numerator = numerator.multiply(scalar)
        print(f"denominator: {denominator.shape}")

        print(f"...projection scalar calculation...")
        #word_frac = np.divide(numerator, denominator)
        word_frac = numerator/denominator
        word_frac = sparse.csr_matrix(word_frac)
        print(f"word_frac: {word_frac.shape}")

        # This block is questionable, sometimes it asks for too much memory, and othertimes it works fine.

        print(f"...final word projection calculation...")
        dir_for_set = os.path.join(CURRENT_DIRECTORY, f"en_data/{CURRENT_SET}/")
        if not os.path.isdir(dir_for_set):
            os.mkdir(dir_for_set)
        dir_for_set = os.path.join(dir_for_set, f"{CURRENT_CALC}")
        if not os.path.isdir(dir_for_set):
            os.mkdir(dir_for_set)

        for polar_set_idx in range(len(POLAR_WORD_SETS)):
            negative_idx = polar_word_index[polar_set_idx][0] 
            positive_idx = polar_word_index[polar_set_idx][1]
            negative_str = POLAR_WORD_SETS[polar_set_idx][0]
            positive_str = POLAR_WORD_SETS[polar_set_idx][1]


            if negative_idx == -1 or positive_idx == -1:
                continue

            print(f"...Calculating polar set {polar_set_idx}_{negative_str}_{positive_str}...")
            
            #print(f"semaxis_norms: {semaxis_norms[polar_set_idx].shape}")
            #print(f"word_frac: {word_frac.T[polar_set_idx].shape}")
            word_projected_point = semaxis_norms[polar_set_idx].T @ word_frac.T[polar_set_idx]
            word_projected_point = sparse.csr_matrix(word_projected_point.T)
            #print(f"word_projected_point: {word_projected_point.shape}")

            # Distance of the word from the midpoint
            word_distance = metrics.pairwise.pairwise_distances(X=word_projected_point,
                                                                Y=midpoint,
                                                                metric = "euclidean",
                                                                n_jobs=-1,
                                                                force_all_finite = True)

            #print(f"count_embeding_normalized: {count_embeding_normalized[1]}")
            #print(f"word_projected_point: {word_projected_point[0]}")
            #print(f"word_distance: {word_distance}")
            #max = np.amax(word_distance) 
            #word_distance = word_distance/max
            word_distance = sparse.csr_matrix(word_distance)
            min_dist = sparse.csr_matrix.min(word_distance)
            max_dist = sparse.csr_matrix.max(word_distance)
            # Used to determine polarity
            change_basis_word_pos = count_embeding - midpoint
            word_dot_product = metrics.pairwise.linear_kernel(  X = change_basis_word_pos,
                                                                Y = semaxis_norms)
            word_dot_product = sparse.csr_matrix(word_dot_product)
            min_dot = np.min(word_dot_product.T[polar_set_idx])
            #print(f"min_dot: {min_dot}")
            max_dot = np.max(word_dot_product.T[polar_set_idx])
            #word_dp_softmax = softmax(word_dot_product.todense())
            #print(f"max_dot: {max_dot}")
            
            dir_for_polar = os.path.join(dir_for_set, f"{polar_set_idx}_{negative_str}_{positive_str}.csv")

            #print(f"word_distance:{word_distance.shape}")
            #print(f"word_dot_product:{word_dot_product.T[polar_set_idx].shape}")

            with open(dir_for_polar, 'w') as textfile:
                textfile.write("Word, Value\n")
                for idx in range(word_distance.shape[0]):
                    if CURRENT_CALC == CALCULATION[0]:
                        sign = 1.0
                        if word_dot_product.T[polar_set_idx].T[idx] < 0.0:
                            sign = -1.0
                        textfile.write(f"{feature_list[idx]},{(GeneralNorm(word_distance[idx].data[0], min_dist, max_dist, 0.0, 1.0) * sign)}\n")
                    elif CURRENT_CALC == CALCULATION[1]:
                        value = word_dot_product.T[polar_set_idx].T[idx]
                        if len(value.data) == 0:
                            value = 0.0
                        else:
                            value = value.data[0]
                        norm_value = GeneralNorm(value, min_dot, max_dot, -1.0, 1.0)
                        #print(f"value: {value}, min_dot: {min_dot}, max_dot:{max_dot}, normalize: {norm_value}")
                        textfile.write(f"{feature_list[idx]},{norm_value}\n")
                

            del word_projected_point
            del word_distance
            del word_dot_product