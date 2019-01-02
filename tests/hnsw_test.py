

import datetime
import numpy as np
import logging
# logging.basicConfig(level=logging.INFO)

import nmslib
for i in range(1):
# create a random matrix to index
    data = np.random.randn(200, 128).astype(np.float32)
    time1=datetime.datetime.now()
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')

    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=False)
    print("init", (datetime.datetime.now() - time1).microseconds)
    time1 = datetime.datetime.now()
    # index.addDataPointBatch(data)
    # index.createIndex({'post': 2}, print_progress=False)
    # print("add data", (datetime.datetime.now() - time1).microseconds)
    # time1 = datetime.datetime.now()
    # query for the nearest neighbours of the first datapoint
    # nmslib.freeIndex({'post': 2})
    # ids, distances = index.knnQueryBatch(data, k=150, num_threads=4)
    neighbours = index.knnQueryBatch(data, k=150, num_threads=4)
    print("time2",len(neighbours))

    for res in neighbours:
        print(res[0])
        print(res[1])
# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
# neighbours = index.knnQueryBatch(data, k=10, num_threads=4)
# print(neighbours)