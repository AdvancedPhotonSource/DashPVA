import json
import subprocess


def main():

    prefix = "dp-ADSim"
    metadata = {}

    with open("gui/PVs.json", "r") as json_file:
        metadata = json.load(json_file)

    command = "pvapy-hpc-collector --collector-id 1 --producer-id-list 1 --input-channel "+prefix+":Pva1:Image --control-channel collector:*:control"
    command += " --status-channel collector:*:status --output-channel collector:*:output --processor-file hpcexample.py --report-period 5"
    command += " --server-queue-size 100 --collector-cache-size 100 --monitor-queue-size 1000"

    i=0
    if metadata and metadata is not None:
        command += " --metadata-channels "
        for pv in metadata:
            command += "ca://"+metadata[pv]
            if i < len(metadata)-1:
                command += ","  
            i +=1
    print(command)
    
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()