import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

reviews_frame = pd.read_pickle("pandas_reviews.pkl")
y = np.loadtxt("targets.txt", delimiter=',')

#flatten all strings into single list to count word ocurrence and output dict with this info

reviews = reviews_frame["cleanedReviews"].str.split()
flattened = [val for sublist in reviews for val in sublist]
word_ocurrence = dict(Counter(flattened))

total_words = sum(word_ocurrence.values())
print("Total number of words:", str(total_words))

# extract top values to be shown on diagram

top_keys = []
top_values = []

for key, value in sorted(word_ocurrence.items(), reverse=True, key=lambda item: item[1])[0:30]:
    top_keys.append(key)
    top_values.append(value/total_words)

    print("%s: %s" % (key, value))

# plot
ax = sns.barplot(x=top_values, y=top_keys)
# Setting title
ax.set_title("% Occurrence of Most Frequent Words")
plt.show()
