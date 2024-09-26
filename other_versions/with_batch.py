from mpi4py import MPI
import json
import time as timer
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process(tweets_batch):
    hourly_sentiments = defaultdict(float)
    daily_sentiments = defaultdict(float)
    hourly_tweets = defaultdict(int)
    daily_tweets = defaultdict(int)

    for tweet in tweets_batch:
        # Check if the expected keys exist
        if 'doc' in tweet and 'data' in tweet['doc'] and 'created_at' in tweet['doc']['data']:
            time = tweet['doc']['data']['created_at']
            hour = time[0:13]
            day = time[0:10]

            # Check for the 'sentiment' key and handle it accordingly
            sentiment = tweet['doc']['data'].get('sentiment', 0)
            score = sentiment['score'] if isinstance(sentiment, dict) else sentiment

            hourly_sentiments[hour] += score
            daily_sentiments[day] += score
            hourly_tweets[hour] += 1
            daily_tweets[day] += 1
        else:
            # print("Tweet with unexpected structure encountered:", tweet)
            break;

    return hourly_sentiments, daily_sentiments, hourly_tweets, daily_tweets

# Main process reads the file and distributes tweets in batches
if rank == 0:
    startTime = timer.time()

    with open('twitter-50mb.json') as file:
        data = json.load(file)

    tweets = data['rows']

    if size == 1:
        result = process(tweets)
        endTime = timer.time()
        execution_times = endTime - startTime
        print("Execution Time: ", execution_times)
        print("Happiest Hour:", max(result[0], key=result[0].get))
        print("Happiest Day:", max(result[1], key=result[1].get))
        print("Most Tweets Hour:", max(result[2], key=result[2].get))
        print("Most Tweets Day:", max(result[3], key=result[3].get))

    else:
        batch_size = max(len(tweets) // (10 * (size - 1)), 1)  # Ensure at least 1 tweet per batch
        tweet_batches = [tweets[i:i + batch_size] for i in range(0, len(tweets), batch_size)]

        for i, batch in enumerate(tweet_batches):
            dest = (i % (size - 1)) + 1
            comm.send(batch, dest=dest)

        # Signal end of data
        for i in range(1, size):
            comm.send(None, dest=i)

# Worker processes
else:
    while True:
        tweet_batch = comm.recv(source=0)
        if tweet_batch is None:
            break
        results = process(tweet_batch)
        comm.send(results, dest=0)

# Collecting results
if rank == 0 and size != 1:
    hourly_sentiments = defaultdict(float)
    daily_sentiments = defaultdict(float)
    hourly_tweet_counts = defaultdict(int)
    daily_tweet_counts = defaultdict(int)

    for _ in range(len(tweet_batches)):
        results = comm.recv(source=MPI.ANY_SOURCE)
        for hour, score in results[0].items():
            hourly_sentiments[hour] += score
        for day, score in results[1].items():
            daily_sentiments[day] += score
        for hour, count in results[2].items():
            hourly_tweet_counts[hour] += count
        for day, count in results[3].items():
            daily_tweet_counts[day] += count

    endTime = timer.time()
    execution_times = endTime - startTime
    print("Execution Time: ", execution_times)
    print("Happiest Hour:", max(hourly_sentiments, key=hourly_sentiments.get))
    print("Happiest Day:", max(daily_sentiments, key=daily_sentiments.get))
    print("Most Tweets Hour:", max(hourly_tweet_counts, key=hourly_tweet_counts.get))
    print("Most Tweets Day:", max(daily_tweet_counts, key=daily_tweet_counts.get))

MPI.Finalize()
