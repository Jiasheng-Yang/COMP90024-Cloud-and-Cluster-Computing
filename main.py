from mpi4py import MPI
import json
import time


def process_json_line(line):
    try:
        tweet = json.loads(line)
        hour = tweet['doc']['data']['created_at'][0:13]
        day = hour[0:10]
        sentiment = tweet['doc']['data'].get('sentiment')
        if sentiment is not None:
            if not isinstance(sentiment, (int, float)):
                sentiment = 0
            return hour, day, sentiment
        else:
            return None, None, 0

    except json.JSONDecodeError:
        return None, None, 0


def combine_result(result, dict):
    for key, value in dict.items():
        if key in result.keys():
            result[key] += value
        else:
            result[key] = value
    return result


start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


filename = 'twitter-1mb.json'

happiest_hour = {}
happiest_day = {}
most_tweet_hour = {}
most_tweet_day = {}

if rank == 0:
    print("Size: ", size)
    if size == 1:
        with open(filename, 'r') as f:
            for line in f:
                hour, day, sentiment = process_json_line(line[:-2])
                if hour is None:
                    continue
                if hour in happiest_hour.keys():
                    happiest_hour[hour] += sentiment
                    happiest_day[day] += sentiment
                    most_tweet_hour[hour] += 1
                    most_tweet_day[day] += 1
                else:
                    happiest_hour[hour] = sentiment
                    happiest_day[day] = sentiment
                    most_tweet_hour[hour] = 1
                    most_tweet_day[day] = 1
    else:
        with open(filename, 'r') as f:
            for line_number, line in enumerate(f):
                i = 1
                comm.send(line, dest=i, tag=1)
                if i > size:
                    i = 1
                else:
                    i += 1
        f.close()
    for i in range(1, size):
        comm.send(None, dest=i, tag=1)

else:
    while True:
        dict1 = {}
        dict2 = {}
        dict3 = {}
        dict4 = {}
        line = comm.recv(source=0, tag=1)
        if line is None:
            break
        hour, day, sentiment = process_json_line(line[:-2])
        if hour is None:
            continue
        if hour in happiest_hour.keys():
            dict1[hour] += sentiment
            dict2[day] += sentiment
            dict3[hour] += 1
            dict4[day] += 1
        else:
            dict1[hour] = sentiment
            dict2[day] = sentiment
            dict3[hour] = 1
            dict4[day] = 1
        result = {"dict1": dict1, "dict2": dict2, "dict3": dict3, "dict4": dict4}
        print("Rank", rank, ":",result)
        comm.send(result, tag=2, dest=0)
    comm.send(None, tag=2, dest=0)

if rank == 0:
    received_results = 0
    while received_results < size - 1:
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2)
        if result is None:
            received_results += 1
            continue
        happiest_hour = combine_result(happiest_hour, result["dict1"])
        happiest_day = combine_result(happiest_day, result["dict2"])
        most_tweet_hour = combine_result(most_tweet_hour, result["dict3"])
        most_tweet_day = combine_result(most_tweet_day, result["dict4"])

    end_time = time.time()
    duration = end_time - start_time
    print("Execution Time::", duration)
    print("happiestHour:", max(happiest_hour, key=happiest_hour.get))
    print("happiestDay:", max(happiest_day, key=happiest_day.get))
    print("mostTweetHour", max(most_tweet_hour, key=most_tweet_hour.get))
    print("mostTweetDay:", max(most_tweet_day, key=most_tweet_day.get))

MPI.Finalize()



