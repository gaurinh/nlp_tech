
# coding: utf-8

# In[3]:

#example 1
from gensim.corpora import Dictionary
from gensim.models import ldamodel
import numpy


# In[4]:

texts = [['jam','friut','pulp'],
        ['press','strawberry','squeeze'],
        ['traffic','signal'],
        ['jam','jam','honey','kissan'],
        ['jam','sweet','tiptree','bread','jelly','spoon','knife','radio'],
        ['gap','door','jam'],
        ['jam','car','traffic'],
        ['sweet','preserve']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# In[8]:

model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=4)
print(model.print_topics(num_topics=4, num_words=2))

# In[9]:

model.show_topics()



# In[10]:

#sweet word belongs to a which topic
print("word Sweet belongs to which topic:")
print(model.get_term_topics('sweet'))


# In[14]:
print("word spoon belongs to which topic:")
print(model.get_term_topics('spoon'))





