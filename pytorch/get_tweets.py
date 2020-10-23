import tweepy
import csv

consumer_key = '6vAmV50fns6vCcZPuwWcUDg2T'
consumer_secret = '2h7pq4x5UCcnWr2VQB5Uou4bjEFRlASjIu2Tc4D2KGF4l7PiJz'
BEAR = 'AAAAAAAAAAAAAAAAAAAAADsqJAEAAAAAURcyiP%2FsqLnUE6qeEoXxny6GdwA%3DmiiVSFPuqvAn91U2oM1yZTiINuGcibtzP81jzaDLADA0MzNK73'
access_key = '2767237849-Wlml8F4YceQpPLeoC9fB6l29wZxbnsPgGSPetc5'
access_secret = 'LW6UtNB1JUCnlTUQ6eG3NIu5vxDYp0hoxIHoqzIfostdc'

#OUT_FILE = "C:/Users/mjvie/projects/rnn/trump_tweets.csv"

def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        print(f"...{len(alltweets)} tweets downloaded so far")
    
    #transform the tweepy tweets into a 2D array that will populate the csv 
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
    
    #write the csv  
    print("Writing Rows")
    with open(f'new_{screen_name}_tweets.csv', 'w', encoding="utf-8") as f:
        print (outtweets[0])
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)
    
    pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("realDonaldTrump")