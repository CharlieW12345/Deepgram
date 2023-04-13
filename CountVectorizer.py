from sklearn.feature_extraction.text import CountVectorizer
texts = ""
with open("Enhanced.txt",'r') as file:
    string = file.read()
    texts = [string]
    

def unigram(number):
    cv = CountVectorizer(stop_words="english",ngram_range=(1,1))   
    cv_fit = cv.fit_transform(texts)    
    word_list = cv.get_feature_names_out() 
    count_list = cv_fit.toarray().sum(axis=0)
    unigram = dict(zip(word_list,count_list))
    sorted_unigram = sorted(unigram.items(), key=lambda x:x[1], reverse=True)
    vector = sorted_unigram[0:number]
    keywordStr = ""
    for (i,j) in vector:
        keywordStr = keywordStr + "&keywords=" + i + ":6" #specify the intesifier for the keyword boosting here
    return keywordStr

def bigram(number):
    cv = CountVectorizer(stop_words="english",ngram_range=(2,2))   
    cv_fit = cv.fit_transform(texts)    
    word_list = cv.get_feature_names_out() 
    count_list = cv_fit.toarray().sum(axis=0)
    bigram = dict(zip(word_list,count_list))
    sorted_bigram = sorted(bigram.items(), key=lambda x:x[1], reverse=True)
    vector = sorted_bigram[0:number]
    keywordStr = ""
    for (i,j) in vector:
        keywordStr = keywordStr + "&keywords=" + i + ":6" 
    return keywordStr

def trigram(number):
    cv = CountVectorizer(stop_words="english",ngram_range=(3,3))   
    cv_fit = cv.fit_transform(texts)    
    word_list = cv.get_feature_names_out() 
    count_list = cv_fit.toarray().sum(axis=0)
    trigram = dict(zip(word_list,count_list))
    sorted_trigram = sorted(trigram.items(), key=lambda x:x[1], reverse=True)
    vector = sorted_trigram[0:number]
    keywordStr = ""
    for (i,j) in vector:
        keywordStr = keywordStr + "&keywords=" + i + ":6"
    return keywordStr



#print(trigram(20))


#function to remove ll and hv etc... don, can probably fill these words in if i really want a lot of work could be done into 
#perfecting the keywords that are chosen , althouigh if this never actually boosts the WER then there is no point