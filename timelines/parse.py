import os
import json


for dirname in os.listdir('.'):
    if os.path.isdir(dirname):
        infos = []
        for i in range(48*8):
            with open(dirname + '/' + str(i)) as f:
                infos.append(json.load(f))
        a = infos[0]
        for i in range(48*8):
            a["traceEvents"].extend(infos[i]["traceEvents"])
        with open(dirname + '.json', 'w') as f:
            json.dump(a, f)
