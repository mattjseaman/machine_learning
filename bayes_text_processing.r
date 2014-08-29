library("tm")
library("e1071")

sms_raw = read.csv("sms_spam.csv", stringsAsFactors = F)

sms_raw$type = factor(sms_raw$type)

#using the tm (text mining) package, we create a corpus out of the text data in the dataframe
#the Corpus() function can store all sorts of text, including PDF and Word format
sms_corpus = Corpus(VectorSource(sms_raw$text))

#this block cleans up the text for further processing - this also takes the size of the corpus from 20 Mb to under 1 Mb!
#1. Convert all letters to lowercase (content_transformer neccessary here because 
#using "tolower" directly converts the text docs to character strings, which 
#cannot be processed by the coming DocumentTermMatrix function)
corpus_clean = tm_map(sms_corpus, content_transformer(tolower))
#2. Remove any numbers
corpus_clean = tm_map(corpus_clean, removeNumbers)
#3. Remove stopwords (like to, and, but, for...) - use the built in list of stopwords
corpus_clean = tm_map(corpus_clean, removeWords, stopwords())
#4. Remove punctuation
corpus_clean = tm_map(corpus_clean, removePunctuation)
#5. Remove whitespace (all the stuff we just removed is replaced by extra spaces)
corpus_clean = tm_map(corpus_clean, stripWhitespace)

#Create a Document-Term Matrix out of the new cleaned corpus.
#This creates a sparse matrix for the Bayesian algorithm to process.
sms_dtm = DocumentTermMatrix(corpus_clean)

#split the original data frame into test and training sets (can just shave off the end since the original list was in random order)
sms_raw_train = sms_raw[1:4169,]
sms_raw_test = sms_raw[4170:5559,]

#split the dtm into test and training sets (can just shave off the end since the original list was in random order)
sms_dtm_train = sms_dtm[1:4169,]
sms_dtm_test = sms_dtm[4170:5559]

#split the corpus into test and training sets (can just shave off the end since the original list was in random order)
sms_corpus_train = corpus_clean[1:4169]
sms_corpus_test = corpus_clean[4170:5559]

#we need to reduce the number of features in our dtm - filter only for those words that 
#appear at least 5 times in the dtm
sms_dict = findFreqTerms(sms_dtm_train, 5)
sms_train = DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test = DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

#Naive Bayes needs categorical features - the dtm currently contains counts.  Convert these counts to factors for the model.
convert_counts = function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}
sms_train = apply(sms_train, MARGIN = 2, convert_counts)
sms_test = apply(sms_test, MARGIN = 2, convert_counts)

#build our model on the sms_train matrix
sms_classifier = naiveBayes(sms_train, sms_raw_train$type)

#make our predictions based on the test data set
sms_test_pred = predict(sms_classifier, sms_test)

#try specifying laplace estimator to improve model performance
sms_classifier2 = naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 = predict(sms_classifier2, sms_test)

#############################################################

#Evaluate the model effectiveness
#library('gmodels')
CrossTable(sms_test_pred2, sms_raw_test$type, prop.chisq = F, prop.t = F, dnn = c('predicted', 'actual'))

#############################################################

#Reference for analyzing corpus with the wordcloud() function
#library("wordcloud")
#wordcloud(sms_corpus_train, min.freq = 40, random.order = F)
#spam = subset(sms_corpus_train, type == "spam")
#ham = subset(sms_corpus_train, type == "ham")
#wordcloud(spam$text, max.words = 40)
#wordcloud(ham$text, max.words = 40)

