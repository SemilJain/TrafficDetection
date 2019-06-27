import csv
import os
import re
import nltk
import pandas as pd
#nltk.download('punkt')
coding:'utf-8'
#or nltk.download('all')

from nltk import word_tokenize
from nltk.stem import PorterStemmer
# import pattern.en
# from pattern.en import suggest
from autocorrect import spell

stopwords={'your', 'each', 'such', 're', 'both', 'couldn', 'doing', 'i', "don't", 'because', 'ours', 'how', 'd', 'we', 'those', 'doesn', "shouldn't", 'and', 'will', 'between', 'wouldn', 'shouldn', 'not', 'while', 'she', "won't", 'other', 'y', 'didn', 'me', 'into', 'was', 'he', 'most', 'under', 'aren', 'below', "hadn't", 'so', 'should', 'hasn', 'why', 'very', "you've", 'is', 'be', 'mightn', "aren't", 'during', 'my', 'as', 'been', 'll', 'were', 'hers', 'did', 'then', 'itself', "should've", 'now', 've', 'all', 'yours', 'over', "doesn't", 'her', 'own', 't', "wasn't", 'it', "weren't", 'where', 'am', 'm', 'by', 'wasn', 'against', 'out', 'weren', 'this', 'more', 'at', 'above', 'too', 'ain', 'for', "hasn't", "you're", 'here', 'can', 'with', 'when', 'a', 'after', 'before', 'down', "mightn't", 'ourselves', 'there', 'but', 'isn', "needn't", 'an', 'them', 'being', 'mustn', 'what', 'our', 'to', 'they', 'myself', "didn't", 'yourself', "you'll", 'haven', 'himself', 's', 'nor', 'off', "haven't", "wouldn't", 'the', 'only', 'herself', 'themselves', "she's", "it's", 'theirs', 'again', 'these', 'any', 'won', 'or', 'about', "isn't", 'yourselves', 'that', "you'd", 'once', 'whom', 'are', 'some', "shan't", 'few', 'their', 'having', "mustn't", 'through', 'in', 'shan', 'from', 'you', 'him', 'ma', 'o', 'does', 'just', 'hadn', 'who', 'if', 'until', 'no', "couldn't", 'which', 'on', 'of', 'up', 'same', 'do', 'further', 'needn', 'his', 'don', 'had', "that'll", 'than', 'have', 'has', 'its', 'A','a', '.', 'we', 'the', 'The', 'We', '``', 'review_body', 'review_date','#','RT','@'}

#os.chdir(r'Downloads')
#fields=['Tweet']
#reader=pd.read_csv('thefile.csv',skipinitialspace=True,usecols=fields)

#def remove_emoji(text):
    #return emoji_pattern.sub(r'', text).encode('utf8')
	
reader = csv.reader(open('thefile.csv'),quotechar='|')
#reader=remove_emoji(reader)
tweetData = list(reader)
print(tweetData)
#print(reader)
list=[]

for line in tweetData:
		innerlist=[]
		print(line)
		for field in line:
			tokens = word_tokenize(field)
			innerlist.append(tokens)
		list.append(innerlist)

# print(list)
# print(len(list))

flat = []
for i in list[1:]:
  for j in i:
    flat.append(j)
print("\n\nAfter Tokenization of CSV file:")
print (flat)
print(len(flat))

filtered_list = []
for w in flat:
	innerfiltered_list=[]
	for k in w:
		if k not in stopwords:
			innerfiltered_list.append(k)
	filtered_list.append(innerfiltered_list)

print("After removing StopWords:")
print(filtered_list)
print(len(filtered_list))

#Removing punctuation

#words = [word for word in filtered_list for w1 in word if w1.isalpha()]

words=[]
for w in filtered_list:
	innerwords=[]
	for w1 in w:
		if w1.isalpha():
			innerwords.append(w1)
	words.append(innerwords)
print("After Removing Punctuation:")
print(words)
print(len(words))

#Stemming

ps = PorterStemmer()
stem_list = []
for w in words:
	innerstem_list=[]
	for w1 in w:
		innerstem_list.append(ps.stem(w1))
	stem_list.append(innerstem_list)
	
print("After Stemming Process:")
print(stem_list)
print(len(stem_list))

#word correction

corrected_list=[]
for w in stem_list:
	innercorrected_list=[]
	for w1 in w:
		innercorrected_list.append(spell(w1))
	corrected_list.append(innercorrected_list)

print("After Correction:")
print(corrected_list)
print(len(corrected_list))

corrected_list = [x for x in corrected_list if x != []]
print(corrected_list)
print(len(corrected_list))

prefinal_list = []
new_list = []

for x in corrected_list:
	if x!=['sentiment']:
		for y in x:
			new_list.append(y)
	elif x==['sentiment']:
		prefinal_list.append(new_list)
		new_list = []

print(prefinal_list)

final_list = []

for l in prefinal_list:
	final_list.append(" ".join(l))
	
print(final_list)

writer = csv.writer(open("thefile2.csv", 'w', encoding = 'utf-8'))
headers = ['Processed Tweet',]
writer.writerow(headers)

for x in final_list:
	writer.writerow([x])

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(final_list)
print(bow)