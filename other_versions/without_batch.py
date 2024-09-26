from mpi4py import MPI
import json
import time as timer


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines


if __name__ == '__main__':
    startTime = timer.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = 'twitter-100gb.json'

    total_lines = len(read_file(filename))
    lines_per_process = total_lines // size
    start_index = rank * lines_per_process
    end_index = (rank + 1) * lines_per_process if rank != size - 1 else total_lines

    lines = read_file(filename)[start_index:end_index]

    for line in lines:
        json_data = json.loads(line)
        print(json_data)
    dict1 = {}
    dict2 = {}
    dict3 = {}
    dict4 = {}

    if size == 1:
        for tweet in tweets:
            if tweet == {}:
                break
            time = tweet['doc']['data']['created_at'][0:13]
            day = time[0:10]
            sentiment = tweet['doc']['data'].get('sentiment')
            if sentiment is not None:
                if not isinstance(sentiment, (int, float)):
                    sentiment = 0
            else:
                continue
            if time in dict1.keys():
                dict1[time] += sentiment
                dict2[day] += sentiment
                dict3[time] += 1
                dict4[day] += 1
            elif time not in dict1.keys():
                dict1[time] = sentiment
                dict2[day] = sentiment
                dict3[time] = 1
                dict4[day] = 1

        endTime = timer.time()
        print("duration: ", endTime - startTime)
        print("happiestHour:", max(dict1, key=dict1.get))
        print("happiestDay:", max(dict2, key=dict2.get))
        print("mostTweetHour", max(dict3, key=dict3.get))
        print("mostTweetDay:", max(dict4, key=dict4.get))

    else:
        # Main core to split data and send to child core
        if rank == 0:
            for i in range(len(tweets)):
                dest = (i % (size - 1)) + 1
                tweet = tweets[i]
                comm.send(tweet, dest=dest)
                # receive result from child core
            for i in range(1, size):
                comm.send(None, dest=i)
        else:
            while True:
                # receive raw data from main core
                tweet = comm.recv(source=0)
                # when there is no new tweets, break loop.
                if tweet is None or tweet == {}:
                    break
                time = tweet['doc']['data']['created_at'][0:13]
                sentiment = tweet['doc']['data'].get('sentiment')
                if sentiment is not None:
                    if isinstance(sentiment, (int, float)):
                        result = {time: sentiment}
                    else:
                        result = {time: sentiment['score']}
                else:
                    result = {time: 0}
                comm.send(result, dest=0)

        if rank == 0:
            while True:
                result = comm.recv(source=MPI.ANY_SOURCE)
                if result is None:
                    break
                for time, score in result.items():
                    day = time[0:10]
                    # task 1
                    if time in dict1.keys():
                        dict1[time] += score
                    else:
                        dict1[time] = score
                    # task 2
                    if day in dict2.keys():
                        dict2[day] += score
                    else:
                        dict2[day] = score
                    # task 3
                    if time in dict3.keys():
                        dict3[time] += 1
                    else:
                        dict3[time] = 1

                    # task 4
                    if day in dict4.keys():
                        dict4[day] += 1
                    else:
                        dict4[day] = 1
            endTime = timer.time()
            print("duration: ", endTime - startTime)
            print("happiestHour:", max(dict1, key=dict1.get))
            print("happiestDay:", max(dict2, key=dict2.get))
            print("mostTweetHour", max(dict3, key=dict3.get))
            print("mostTweetDay:", max(dict4, key=dict4.get))

    MPI.Finalize()
