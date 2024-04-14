import json
import gzip
import numpy as np

outfile = "test.json.gzip"
with gzip.open(outfile, "w") as f:

    for i in range(1000):
        
        r = np.random.sample(100)
        f.write(np.array2string(r).encode())

with gzip.open(outfile, "r") as f:

    for line in f.readlines():
        print(line.decode("utf-8"))
        break
