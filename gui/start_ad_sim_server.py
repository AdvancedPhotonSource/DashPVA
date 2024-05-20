import json
import subprocess


def main():

    prefix = "dp-ADSim"
    metadata = {}

    with open("gui/PVs.json", "r") as json_file:
        metadata = json.load(json_file)

    command = "pvapy-ad-sim-server -cn pvapy:"+prefix+":Pva1:Image -nx 128 -ny 128 -fps 100 -rp 100 -rt 180 -mpv "

    i=0
    if metadata and metadata is not None:
        for pv in metadata:
            command += "ca://"+metadata[pv]
            if i < len(metadata)-1:
                command += ","  
            i +=1
    print(command)
    
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()