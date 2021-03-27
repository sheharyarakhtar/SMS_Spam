sms_raw <- read.csv('sms_spam.csv', stringsAsFactors = F)
str(sms_raw)
sms_raw$type <- as.factor(sms_raw$type)
library(tm)

sms_corpus <- Corpus(VectorSource(sms_raw$text))
inspect(sms_corpus)

corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

sms_dtm <- DocumentTermMatrix(corpus_clean)

sms_raw_train <- sms_raw[1:4169,]
sms_raw_test <- sms_raw[4170:5574,]

sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5574,]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5574]


install.packages('wordcloud')
library(wordcloud)

wordcloud(sms_corpus_train, min.freq = 40, random.order = F)
spam <- subset(sms_raw_train, type=='spam')
ham <- subset(sms_raw_train, type =='ham')

wordcloud(spam$text, max.words = 40, scale = c(3,0.5), random.order = F)
wordcloud(ham$text, max.words = 40, scale = c(3,0.5), random.order = F)


#remove terms with freq less than 5
sms_dict<-findFreqTerms(sms_dtm_train,5)
sms_train<-DocumentTermMatrix(sms_corpus_train,list(dictionary=sms_dict))
sms_test<-DocumentTermMatrix(sms_corpus_test,list(dictionary=sms_dict))

#naive bayes is trained on categorical data
#so the repition of words is not important, only their presense
#We make a function to convert the count to Y/N factor
convert_count <- function(x)
{
  x <- ifelse(x>0,1,0)
  x <- factor(x, level = c(0,1), label = c('No', 'Yes'))
  return(x)
}

sms_train <- apply(sms_train, MARGIN = 2, convert_count)
sms_test <- apply(sms_test, MARGIN = 2, convert_count)

#library for naive bayes
library(e1071)
sms_classifier <- naiveBayes(as.matrix(sms_train), sms_raw_train$type)
sms_test_pred <- predict(sms_classifier, as.matrix(sms_test))


CrossTable(x = sms_raw_test$type, y= sms_test_pred, prop.chisq = FALSE, prop.t = FALSE, dnn = c('actual', 'predicted'))
