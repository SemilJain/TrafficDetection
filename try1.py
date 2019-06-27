import re
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import csv
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import warnings 
import pickle
#nltk.download('wordnet')
warnings.filterwarnings("ignore", category=FutureWarning)

train= pd.read_csv(r'trainingData.csv')
test=pd.read_csv(r'testingData.csv')
#combi = train.append(test)

combi = train.append(test, ignore_index=True)

#A) PreProcessing

def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(i, '', input_txt)
        
	return input_txt

combi['Tidy_Tweet'] = np.vectorize(remove_pattern)(combi['Tweet'], "@[\w]*") # remove twitter handles (@user)

combi['Tidy_Tweet'] = combi['Tidy_Tweet'].str.replace("[^a-zA-Z#]", " ") # remove special characters, numbers, punctuations

combi['Tidy_Tweet'] = combi['Tidy_Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #Removing Short Words

tokenized_tweet = combi['Tidy_Tweet'].apply(lambda x: x.split()) #Tokenization

#from nltk.stem.porter import *
#stemmer = PorterStemmer()

#tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

#print(tokenized_tweet.head())

wordnet_lemmatizer = WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [wordnet_lemmatizer.lemmatize(i) for i in x]) #Lemmatizing

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['Tidy_Tweet'] = tokenized_tweet

#all_words = ' '.join([text for text in train['Tidy_Tweet']])
from wordcloud import WordCloud
#wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

#plt.figure(figsize=(10, 7))
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis('off')
#plt.show()

# B) Words in Traffic Tweets

traffic_words = ' '.join([text for text in combi['Tidy_Tweet'][combi['Label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(traffic_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# C) Understanding the impact of Hashtags on tweets sentiment

# # function to collect hashtags
def hashtag_extract(x):
	hashtags = []
	# # Loop over the words in the tweet
	for i in x:
		ht = re.findall(r"#(\w+)", i)
		hashtags.append(ht)
	return hashtags

# # # extracting hashtags from non traffic tweets
HT_regular = hashtag_extract(combi['Tidy_Tweet'][combi['Label'] == 0])

# # # extracting hashtags from traffic tweets
HT_traffic = hashtag_extract(combi['Tidy_Tweet'][combi['Label'] == 1])

# # # unnesting list
HT_regular = sum(HT_regular,[])
HT_traffic = sum(HT_traffic,[])

b = nltk.FreqDist(HT_traffic)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
 # # selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

# D) Extracting Features

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=1, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['Tidy_Tweet'])
#print(bow)

# E) Training and Testing using Multinomial Naive Bayes

#writer=csv.writer(open('Combine.csv','w',encoding='utf-8'))
#headers=['Processed Tweet',]
#writer.writerow(headers)
#for x in combi:
#	for y in x:
#		writer.writerow(y)
print([test])
#print([combi])
#print(combi['Label'])

#with open(r'C:\Users\Bhavesh\Downloads\1_TrainingSet_Copy_1_2Class.csv', 'a') as f:
#		writer = csv.writer(f)
#		for y in test['Processed Tweet']:
#			writer.writerow([y])

combi = combi.rename(columns = {'fit': 'fit_feature'})
corpus = combi["Tidy_Tweet"]
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus).toarray()
print(X.shape)

#print(vectorizer.get_feature_names()[:500])

categories = combi["Label"].unique()
category_dict = {value:index for index, value in enumerate(categories)}
results = combi["Label"].map(category_dict)
print(category_dict)
print ("corpus size: %s" % len(vectorizer.get_feature_names()))
#X=np.random.shuffle(X)

x_train,x_test, y_train,y_test = train_test_split(X, results, test_size=0.2, random_state=1, )

#clf = MultinomialNB()

pkl_filename = 'multinomial.pkl'
multinomial_model_pkl = open(pkl_filename, 'rb')
clf = pickle.load(multinomial_model_pkl)

#clf.fit(x_train, y_train)

print(clf.score(x_test, y_test)) #printing f1 score of trained data
print(y_test)
print(clf.predict(x_test)) #predicting on train data
print(category_dict)

# Dump the trained decision tree classifier with Pickle

#pkl_filename = 'multinomial.pkl'
# Open the file to save as pkl file
#multinomial_model_pkl = open(pkl_filename, 'wb')
#pickle.dump(clf, multinomial_model_pkl)
# Close the pickle instances
#multinomial_model_pkl.close()

#checking on user defined tweets

#text = ["heavy traffic between powai and vikhroli","Hello what u doin today"]
#vec_text = vectorizer.transform(text).toarray()
#print((clf.predict(vec_text)[0]))
#print((clf.predict(vec_text)[1])) #if 0 tweet is non traffic or else it is traffic tweet

#print(test[1])
#vec_text = vectorizer.transform(test).toarray()
#print(list(category_dict.keys()[category_dict.values().index(clf.predict(vec_text)[0])]))
#print(list(category_dict.keys()[category_dict.values().index(clf.predict(vec_text)[0])]))

#pred = clf.predict(vec_text)[0]
#if(pred):
#	print("Tweet is:",text[0],"\nNo Traffic")
#else:
#	print("Tweet is:",text[0],"\nTraffic")

#pred = clf.predict(vec_text)[1]
#if(pred):
#	print("Tweet is:",text[1],"\nNo Traffic")
#else:
#	print("Tweet is:",text[1],"\nTraffic")
	
#print("Tweet is:",text[2],"\n",(clf.predict(vec_text)[2]))

#Checking on tweets collected fro twitter by us

text=pd.read_csv(r'thefile2.csv')
# print(text)

t_l = text['Processed Tweet'].values.tolist()
#print(t_l)
l= [x for x in t_l if str(x) != 'nan' ]
print(l)
vec_text = vectorizer.transform(l).toarray()
#print(vec_text)	
for i in range(len(vec_text)):
	
	pred = clf.predict(vec_text)[i]
	
	#print(pred)
	if(pred == 1):
		print("Tweet is:",l[i],"\nNo Traffic")
	else:
		print("Tweet is:",l[i],"\nTraffic")
