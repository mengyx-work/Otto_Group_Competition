import numpy as np
import pandas as pd
import lda
import lda.datasets
import numpy as np
import matplotlib.pyplot as plt

#X = lda.datasets.load_reuters()
train = pd.read_csv('train.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')
#sample = pd.read_csv('sampleSubmission.csv')
targets = train['target']
train = train.drop('target', axis=1)

print train.shape, test.shape

#LDA_Matrix = np.array(train)
LDA_Matrix = np.concatenate((train, test), axis=0)

model = lda.LDA(n_topics=9, n_iter=500, random_state=1)
model.fit(LDA_Matrix)
#model.fit(X)

topic_word = model.topic_word_
topic_word.shape
