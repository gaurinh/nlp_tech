#example 2
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
# Importing Gensim
import gensim
from gensim import corpora

#6 documents
doc1 = "Narendra Modi is indian politician and current prime minister of india."
doc2 = "Modi is a surname in India."
doc3 = "It is mostly associated with Baniyas,Grain merchants and Grocers."
doc4 = "Modi is a Hindu Nationalist. He is a member of the Rashtriya Swayamsevak Sangh (RSS)."
doc5 = "The surname is most commonly found amongst people from the Northern and Western states of Haryana, Madhya Pradesh,Bihar, Jharkhand, Rajasthan and Gujarat."
doc6 = "He is a member of the Bharatiya Janata Party (BJP)."

#combine data
doc_combo = [doc1, doc2, doc3, doc4, doc5,doc6]

#data preprocessing. remove stopwords and punctuation
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lem = WordNetLemmatizer()
def preproc(document):
    stoprem = " ".join([i for i in document.lower().split() if i not in stop])
    puncrem = ''.join(ch for ch in stoprem if ch not in exclude)
    lemword = " ".join(lem.lemmatize(word) for word in puncrem.split())
    return lemword

docpreproc = [clean(document).split() for document in doc_combo]  

# create dictionary where index is assigned to every unique term.
term_dictionary = corpora.Dictionary(docpreproc) 

# convert list to document-term-matrix
dtm = [term_dictionary.doc2bow(document) for document in docpreproc]

# Create the object for LDA model using gensim
Lda = gensim.models.ldamodel.LdaModel

# Run and train LDA model.
ldamodel = Lda(dtm, num_topics=5, id2word = term_dictionary, passes=50)

#show result
print(ldamodel.print_topics(num_topics=5, num_words=2))
