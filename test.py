from keras.models import load_model
#from main import clean_description
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords

max_len = 150
stopwords = set(stopwords.words('english'))

detokenizer = TreebankWordDetokenizer()

def clean_description(desc):
    '''
    function to "clean" the review", get rid of extra words, articles an other words with minimal info
    '''
    desc = word_tokenize(desc.lower())
    desc = [token for token in desc if token not in stopwords and token.isalpha()]
    return detokenizer.detokenize(desc)


model = load_model('model_batch128_weights_smaller.h5')

# 5,5,3,1,  2,4,4,5
review = [clean_description("Great product I would buy it again for sure.")]
review.append(clean_description("Another great product.  Just love this Biotin B-complex thickening conditioner.  It really helps the hair to look fuller and to grow."))
review.append(clean_description("I am half way through the bottle and still have problems with my scalp. I will continue to use (instead of almost everyday) to see it it makes a difference. The price isn't too bad though. so I might buy another one."))
review.append(clean_description("Please avoid this product. The conditional is so thick that it does not flow out when you want to use it. As for quality, it is horrible. It makes your hair sticky as if you haven't rinsed your hair for days. I don't see any good using it. I finally use Dove that I bought at Costco. It is cheaper and make your hair soft and weightless as well"))
review.append(clean_description("This perfume smells really synthetic and cheap...more like a dodgy air freshener or body spray than a classy perfume. I didn't like the smell at all. It's not about being a snob...I've tried perfumes by Lidl and Primark and had no complaints. This one is grossly overpriced in my opinion and I wouldn't want to give or receive it as a gift."))
review.append(clean_description("I decided recently to find a new scent as the one I'd been using had become ridiculously expensive, so I was glad to try this out. One of the things I like about it as that the scent lingers for several days on the wrists. It's not too floral, slightly spicy, but overall very subtle. I would buy it, 4 stars to it!"))
review.append(clean_description("The only drawback is that it's a common fragrance type which means you wouldn't feel too 'premium' with this."))

review.append(clean_description("The shampoo is really nice on my hair, it doest not dry my skin out, which is wonderful"))

#cleanedReview = clean_description(review)

#print(review)

#in_str = [cleanedReview]
test_frame = pd.DataFrame(review, columns=['reviews'])

# reviews_frame = pd.read_pickle("pandas_reviews.pkl")
# y = np.loadtxt("targets.txt", delimiter=',')
# X_train, X_val, _, _ = train_test_split(reviews_frame["cleanedReviews"], y, test_size=0.2)
#
# tokenizer = Tokenizer(num_words=None)
# # fit tokenizer
# tokenizer.fit_on_texts(X_train)
#tokenizer.fit_on_texts(test_frame["reviews"])

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(test_frame["reviews"])

X_test = pad_sequences(sequences_test, maxlen=max_len)



pred_test = model.predict(X_test)
print(pred_test)
pred_test = [np.argmax(x)+1 for x in pred_test]
# plus 1 has to be done because classes go from 0 to 4 instead of 1 to 5
print(pred_test)


