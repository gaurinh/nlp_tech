#uncomment this when required
#install.packages("twitteR")
#install.packages("tm")
#install.packages("wordcloud")
#install.packages("RColorBrewer")
#install.packages("ROAuth")
#install.packages("RCurl")
#install.packages("SnowballC")
#install.packages("topicmodels")
#install package sentiment140
#install.packages("devtools")
#install_github("sentiment140", "okugami79")
#install.packages("sentiment")
#install.packages("data.table")

# Load this list of libraries
library(data.table)
library(sentiment)
library(devtools)
library(topicmodels)
library(twitteR)
library(tm)
library(wordcloud)
library(RColorBrewer)
library(ROAuth)
library(RCurl)
library(tm)
library(SnowballC)
library(ggplot2)
library(quantmod)
library(TTR)

# get the credentials
reqURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"
consumerKey <- "xxxx"
consumerSecret <- "xxxx"
access_token <- "xxxx"
access_secret <- "xxxx"

setup_twitter_oauth(consumerKey,consumerSecret,access_token,access_secret)



# searching for tweets containing the keywords "infy"
gt = searchTwitter("infy infosys",n=1000)
gt

#length
n.gt = length(gt)
n.gt

#convert to dataframe
gt.df = twListToDF(gt)
gt.df
View(gt.df)
write.csv(gt.df, file="tweets.csv")

#fecth the tweets and write to tweets.csv.No need to fetch tweets again. uncomment line below and run the code.do not run lines above this.

#gt.df <- read.csv("tweets.csv")

# build a corpus, and specify the source to be character vectors
mCorpus <- Corpus(VectorSource(gt.df$text))
# remove URLs
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
mCorpus <- tm_map(mCorpus, content_transformer(removeURL))
# remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
mCorpus <- tm_map(mCorpus, content_transformer(removeNumPunct))
# remove stopwords
mStopwords <- stopwords('english')
mCorpus <- tm_map(mCorpus, removeWords, mStopwords)
# remove extra whitespace
myCorpus <- tm_map(myCorpus, stripWhitespace)
# keep a copy for stem completion later
mCorpusCopy <- mCorpus

#stemming
newCorpus <- tm_map(mCorpus, stemDocument) # stem words
writeLines(strwrap(mCorpus[[200]]$content, 60))
stemCompletion2 <- function(x, dictionary) {
  x <- unlist(strsplit(as.character(x), " "))
  x <- x[x != ""]
  x <- stemCompletion(x, dictionary=dictionary)
  x <- paste(x, sep="", collapse=" ")
  PlainTextDocument(stripWhitespace(x))
}
newCorpus <- lapply(mCorpus, stemCompletion2, dictionary=mCorpusCopy)
newCorpus <- Corpus(VectorSource(mCorpus))
writeLines(strwrap(myCorpus[[200]]$content, 60))


# count word frequency
wordFreq <- function(corpus, word) {
  results <- lapply(corpus,
                    function(x){grep(as.character(x), pattern=paste0("\\<",word)) }
  )
  sum(unlist(results))
}
n.infy <- wordFreq(mCorpus, "infy")
n.infosys <- wordFreq(mCorpus, "infosys")
cat(n.infy, n.infosys)

new.infy <- wordFreq(newCorpus, "infy")
new.infosys <- wordFreq(newCorpus, "infosys")
cat(n.infy, n.infosys)

tdm <- TermDocumentMatrix(newCorpus,
                          control = list(wordLengths = c(1, Inf)))
View(as.matrix(tdm))

# creating document term matrix  
tdm = TermDocumentMatrix(newCorpus,
                         control = list(removePunctuation = TRUE,
                                        removeNumbers = TRUE, tolower = TRUE))
tdm


idx <- which(dimnames(tdm)$Terms %in% c("infy","infosys","sikka","resign"))

idx
as.matrix(tdm[idx,100:200])


# find frequent words
(freq.terms <- findFreqTerms(tdm,lowfreq = 30))
freq.terms


term.freq <- rowSums(as.matrix(tdm))
term.freq <- subset(term.freq, term.freq >= 30)
df <- data.frame(term = names(term.freq), freq = term.freq)

df

#graph1
ggplot(df, aes(x=term, y=freq)) + geom_bar(stat="Identity") +
  xlab("Terms") + ylab("Count") + coord_flip() +
  theme(axis.text=element_text(size=7))

m <- as.matrix(tdm)
# calculate the frequency of words and sort it by frequency
word.freq <- sort(rowSums(m), decreasing = T)
# colors
pal <- brewer.pal(9, "BuGn")[-(1:4)]

# plot word cloud
#graph2
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 3,
          random.order = F, colors = pal)
findAssocs(tdm, "murthy", 0.2)
findAssocs(tdm, "infosys", 0.2)

#topic Modelling
dtm <- as.DocumentTermMatrix(tdm)
dtm
#ui = unique(dtm$i)
#dtm.new = dtm[ui,]

lda <- LDA(dtm, k = 8) # find 8 topics
lda
term <- terms(lda, 7) # first 7 terms of every topic
(term <- apply(term, MARGIN = 2, paste, collapse = ", "))



topics <- topics(lda) # 1st topic identified for every document (tweet)
topics <-as.data.frame(date=as.IDate(gt.df$created), topic=topics)
ggplot(data.frame(topics), aes(date)) + geom_density(position = "stack")



# sentiment analysis

sentiments <- sentiment(gt.df$text)
table(sentiments$polarity)

# sentiment plot
sentiments$score <- 0
sentiments$score[sentiments$polarity == "positive"] <- 1
sentiments$score[sentiments$polarity == "negative"] <- -1
sentiments$date <- as.IDate(gt.df$created)

result <- aggregate(score ~ date, data = sentiments, sum)
result
sentiments
#graph3
plot(result, type = "l")


#draw chart series and find whether result$score plot matched with stock price chartfor the same dates

getSymbols("INFY")
INFY['2017-08-17::2017-08-20']
#Graph4
chartSeries(INFY,theme='white',subset = '2017-08')
addROC(n=200)



###compare sentiment 140 result with this.

# getting positive and negative words txt file
#positive_words=scan("C:\\Users\\Public\\Documents\\positive-words.txt",what="character",comment.char=";")

#negative_words=scan("C:\\Users\\Public\\Documents\\negative-words.txt",what="character",comment.char=";")


#positive_words = c(positive_words, "new","nice" ,"good", "horizon")
#positive_words
#negative_words = c(negative_words, "wtf", "behind","resign","feels","ugly", "back","worse" ,"shitty", "bad", "no","freaking","sucks","horrible","miss")
# code for sentiment analysis
#score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
#{
#  library(plyr)
#  library(stringr)

# we got a vector of sentences. plyr will handle a list
# or a vector as an "l" for us
# we want a simple array ("a") of scores back, so we use 
# "l" + "a" + "ply" = "lapply":
#  scores = laply(sentences, function(sentence, pos.words, neg.words) {

# clean up sentences with R's regex-driven global substitute, gsub():
#    sentence = gsub('[[:punct:]]', '', sentence)
#    sentence = gsub('[[:cntrl:]]', '', sentence)
#    sentence = gsub('\\d+', '', sentence)
# and convert to lower case:
#    sentence = tolower(sentence)

# split into words. str_split is in the stringr package
#    word.list = str_split(sentence, '\\s+')
# sometimes a list() is one level of hierarchy too much
#    words = unlist(word.list)

# compare our words to the dictionaries of positive & negative terms
#    pos.matches = match(words, pos.words)
#    neg.matches = match(words, neg.words)

# match() returns the position of the matched term or NA
# we just want a TRUE/FALSE:
#    pos.matches = !is.na(pos.matches)
#    neg.matches = !is.na(neg.matches)

# and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
#    score = sum(pos.matches) - sum(neg.matches)

#    return(score)
#  }, pos.words, neg.words, .progress=.progress )

#  scores.df = data.frame(score=scores, text=sentences)
#  return(scores.df)
#}

#test <- ldply(gt,function(t) t$toDataFrame() )

# perform sentiment analysis for the dataset tweets and store it in Results
#Results=score.sentiment(test$text,positive_words,negative_words)
#View(Results)

# getting the summary of the Results
#summary(Results)

# plot histogram
#hist(Results$score)
#table(Results$score)

#library(ggplot2)
#qplot(Results$score,xlab = "Score of tweets")




