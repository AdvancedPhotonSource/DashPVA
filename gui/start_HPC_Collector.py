import os
import sys
import json
import subprocess


def main():

    metadata = {}

    with open("PVs.json", "r") as json_file:
        metadata = json.load(json_file)

    command = "pvapy-hpc-collector --collector-id 1 --producer-id-list 1 --input-channel pvapy:dp-ADSim:Pva1:Image --control-channel collector:*:control"
    command += " --status-channel collector:*:status --output-channel collector:*:output --processor-file hpcexample.py --report-period 10"
    command += " --server-queue-size 100 --collector-cache-size 100 --monitor-queue-size 1000"

    i=0
    if metadata and metadata is not None:
        command += " --metadata-channels "
        for pv in metadata:
            command += "pva://"+metadata[pv]
            if i < len(metadata)-1:
                command += ","  
            i +=1
    print(command)
    
    #subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()