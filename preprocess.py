import json

import seaborn as sns

import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize

num_classes = 5
embedding_dim = 300
epochs = 50
batch_size = 128

sns.set(style="whitegrid")

stopwords = set(stopwords.words('english'))
# Detokenizer combines tokenized elements
detokenizer = TreebankWordDetokenizer()


def clean_description(desc):
    '''
    function to "clean" the review", get rid of extra words, articles an other words with minimal info
    '''
    desc = word_tokenize(desc.lower())
    desc = [token for token in desc if token not in stopwords and token.isalpha()]
    return detokenizer.detokenize(desc)



data_products = []
#word_occurrence = 0

class_counter = [0, 0, 0, 0, 0]

y = np.zeros([50000, num_classes])
pos = 0

cleanedReviews = []
reviews = []


#read json file line by line
with open('/home/jmateo/PycharmProjects/accenture/Beauty_5_50000.json') as json_file:
    for line in json_file:
        data_line = json.loads(line)

        #clean text with above function

        #data_line["cleaned_description"] = clean_description(data_line["reviewText"])

        reviews.append(data_line["reviewText"])
        cleanedReviews.append(clean_description(data_line["reviewText"]))

        y[pos, data_line["overall"]-1] = 1


        #add entire data line to data list
        #data_products.append(data_line)
        pos += 1

        #activate to count class distribution
        #class_counter[data_line["overall"]-1] += 1

#print("Total of rated products:", str(len(data_products)))

zippedList = list(zip(reviews, cleanedReviews))

#reviews_frame = pd.DataFrame(zippedList, columns=[reviews, cleanedReviews])
reviews_frame = pd.DataFrame(zippedList, columns=['reviews', 'cleanedReviews'])
print(reviews_frame)


reviews_frame.to_pickle("pandas_reviews.pkl")  # where to save it, usually as a .pkl
np.savetxt('targets.txt', y, delimiter=',')













