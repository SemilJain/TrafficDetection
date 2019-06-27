
#for error with tweepy and textblob run following commands
#py -m pip install tweepy
#py -m pip install textblob

import tweepy
from textblob import TextBlob
import csv
import collections
import datetime
from dateutil import parser


consumer_key='lN3uAqt045EPQu1Jp20uOQ0kN'
consumer_secret='YsJmS5NBuvS3AVWRZAk8H3VPAMU2UBWZIlhc6x17Zi5b98yuxO'

access_token='1086596186226724864-Js1Q1YYAKDwT61vlWR8WrVbhpFqkoz'
access_token_secret='rtG5z1hFMVbanNozRNf9OeZo0jrhBvAi6MHFM8ifxUqfd'

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)

query=input("Enter query... \n")

public_tweet=api.search(q=query,count=3,tweet_mode='extended')
writer = csv.writer(open("thefile.csv", 'w'))
headers = ['Id','Tweet',
		'Sentiment',]
writer.writerow(headers)	
counter = collections.defaultdict(int)
count=0
Three_days_ago = datetime.datetime.utcnow()-datetime.timedelta(hours = 3)
for t in public_tweet:
	tweeted_datetime = parser.parse(str(t.created_at))
	if tweeted_datetime > Three_days_ago:
		print(t.full_text,'\nthe id of user tweet is',t.id) 
		analysis=TextBlob(t.full_text)
		print(analysis.sentiment)
	
#with open('thefile.csv', 'rb') as f:
#		data = list(csv.reader(f))
	count+=1
	data=t.full_text
	data1=analysis.sentiment
	items={count,data,data1}
	
	#writes horizontally the data a single character at a time. Perfectly working on 21-02-2019 at 10.42
	#writer.writerow([data])
	
	writer.writerow(items)
	
	#writes vertically the data a single character at a time.
	
	#for row in data:
	#	counter[row[0]] += 1
	
	#writes/over writes only last line		
	#writer = csv.writer(open("thefile.csv", 'w'))
	
	#here original stmt was if counter[row[0]]>=4
	
	#for row in data:
		#if counter[row[0]] >= 1:
			#writer.writerow(row)
	
	#this line can replace above all code of vertical writing.
	#writer.writerows(data)
