from mpi4py import MPI
from collections import defaultdict
import json
import time as timer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

BUFFER_SIZE = 1024 * 1024 * 8 # 10MB
CHUNK_SIZE = 1024 * 8

def processData(chunk):
    hourly_sentiments = defaultdict(float)
    daily_sentiments = defaultdict(float)
    hourly_tweets = defaultdict(int)
    daily_tweets = defaultdict(int)

    tweets = chunk.strip().split(',\n')

    for tweet_line in tweets:
        try:
            tweet = json.loads(tweet_line)
            if 'doc' in tweet and 'data' in tweet['doc'] and 'created_at' in tweet['doc']['data']:
                time = tweet['doc']['data']['created_at']
                hour = time[0:13]
                day = time[0:10]

                sentiment = tweet['doc']['data'].get('sentiment', 0)
                score = sentiment['score'] if isinstance(sentiment, dict) else sentiment

                hourly_sentiments[hour] += score
                daily_sentiments[day] += score
                hourly_tweets[hour] += 1
                daily_tweets[day] += 1
        except json.JSONDecodeError:
            continue

    return hourly_sentiments, daily_sentiments, hourly_tweets, daily_tweets


def collectResults(results):
    if not isinstance(results, tuple) or len(results) != 4:
        # print("Unexpected results structure:", results)
        return
    for hour, score in results[0].items():
        hourly_sentiments[hour] += score
    for day, score in results[1].items():
        daily_sentiments[day] += score
    for hour, count in results[2].items():
        hourly_tweet_counts[hour] += count
    for day, count in results[3].items():
        daily_tweet_counts[day] += count

hourly_sentiments = defaultdict(float)
daily_sentiments = defaultdict(float)
hourly_tweet_counts = defaultdict(int)
daily_tweet_counts = defaultdict(int)


if rank == 0:
    startTime = timer.time()
    with open('twitter-50mb.json', 'r') as file:
        while True:
            chunk = file.read(BUFFER_SIZE)
            if not chunk:
                break

            if size == 1:
                results = processData(chunk)
                collectResults(results)
            else:
                status = MPI.Status()
                comm.recv(source=MPI.ANY_SOURCE, status=status)
                comm.send(chunk, dest=status.Get_source())

    if size != 1:
        for _ in range(1, size):
            status = MPI.Status()
            comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(None, dest=status.Get_source())
            results = comm.recv(source=status.Get_source())
            if results:
                collectResults(results)

    endTime = timer.time()
    execution_times = endTime - startTime
    print("Execution Time: ", execution_times)
    print("Happiest Hour:", max(hourly_sentiments, key=hourly_sentiments.get))
    print("Happiest Day:", max(daily_sentiments, key=daily_sentiments.get))
    print("Most Tweets Hour:", max(hourly_tweet_counts, key=hourly_tweet_counts.get))
    print("Most Tweets Day:", max(daily_tweet_counts, key=daily_tweet_counts.get))

    for i in range(1, size):
        comm.send(None, dest=i)

else:
    while True:
        comm.send(None, dest=0)
        data_chunk = comm.recv(source=0)
        if data_chunk is None:
            break
        results = processData(data_chunk)
        comm.send(results, dest=0)

MPI.Finalize()
