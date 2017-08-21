import matplotlib.pyplot as plt
import pandas as pd
from nltk import FreqDist
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re, string
import nltk
from nltk.collocations import *
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.cluster import KMeans

tweet_df = pd.read_csv("/home/nlp/aegis/tweets.csv", encoding = "ISO-8859-1")

users = tweet_df['screenName'].tolist()
fd = FreqDist(users)
fd.plot(20)

tweet_content = tweet_df["text"].tolist()
stopwords = stopwords.words('english')
english_vocab = set(v.lower() for v in nltk.corpus.words.words())



def proc_content(con):
   if con.startswith('@null'):
       return "[Invalid Tweet]"
   con = re.sub(r'\$\w*','',con)
   con = re.sub(r'https?:\/\/.*\/\w*','',con)
   con = re.sub(r'['+string.punctuation+']+', ' ',con)
   twtok = TweetTokenizer(strip_handles=True, reduce_len=True)
   tokens = twtok.tokenize(con)
   tokens = [t.lower() for t in tokens if t not in stopwords and len(t) > 2 and t in english_vocab]
   return tokens
 
words = []
for tc in tweet_content:
    words += proc_content(tc)
   


 
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words, 10)
finder.apply_freq_filter(10)
print(finder.nbest(bigram_measures.likelihood_ratio, 10))



contents_clean= []
for cc in tweet_content:
    words = proc_content(cc)
    content_clean = " ".join(w for w in words if len(w) > 2 and w.isalpha())
    contents_clean.append(content_clean)
tweet_df['CleanedTweet'] = contents_clean





 
vec_tfidf = TfidfVectorizer(use_idf=True, ngram_range=(1,3))  
tfidfm = vec_tfidf.fit_transform(contents_clean)  
feature_names = vec_tfidf.get_feature_names() 
distance = 1 - cosine_similarity(tfidfm)  
print(distance) 


no_clusters = 3  
km = KMeans(n_clusters=no_clusters)  
km.fit(tfidfm)  
clusters = km.labels_.tolist()  
tweet_df['ClusterID'] = clusters
print(tweet_df['ClusterID'].value_counts())


prox_cen = km.cluster_centers_.argsort()[:, ::-2]
for i in range(no_clusters):
    print("Cluster {} : Words :".format(i))
    for cent in prox_cen[i, :10]: 
        print(' %s' % feature_names[cent])
    

from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(dc):
    stop_free = " ".join([i for i in dc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
texts = [text for text in contents_clean if len(text) > 2]
dc_clean = [clean(dc).split() for dc in texts]
dictionary = corpora.Dictionary(dc_clean)
dtm = [dictionary.doc2bow(dc) for dc in dc_clean]
ldamodel = models.ldamodel.LdaModel(dtm, num_topics=3, id2word = dictionary, passes=5)
for topic in ldamodel.show_topics(num_topics=6, formatted=False, num_words=6):
    print("Topic {}: Words: ".format(topic[0]))
    topicwords = [w for (w, val) in topic[1]]
    print(topicwords)



tagdoc = []
tagtweet = {}
for index,i in enumerate(contents_clean):
    if len(i) > 2: # Non empty tweets
         tag = u'SENT_{:d}'.format(index)
         sentence = TaggedDocument(words=gensim.utils.to_unicode(i).split(),tags=[tag])
         tagtweet[tag] = i
         tagdoc.append(sentence)
model = gensim.models.Doc2Vec(tagdoc, dm=0, alpha=0.05, size=20, 
min_alpha=0.025, min_count=0)
for epoch in range(60):
    if epoch % 20 == 0:
        print('Training epoch %s' % epoch)
    token_count = sum([len(i) for i in tagdoc])     
    model.train(tagdoc,total_examples = token_count,epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha





data = model.wv.syn0
cluster = KMeans(n_clusters=6)
centroid = cluster.fit_predict(data)
maptopic2word= {}
for i, val in enumerate(data):
    tag = model.docvecs.index_to_doctag(i)
    topic = centroid[i]
    if topic in maptopic2word.keys():
        for w in (tagtweet[tag].split()):
            maptopic2word[topic].append(w)
    else:
        maptopic2word[topic] = []
for tw in maptopic2word:
    words = maptopic2word[tw]
    print("Topic {} has words {}".format(tw, words[:5]))
     




















