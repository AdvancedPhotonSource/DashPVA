import json
import subprocess


def main():

    prefix = "dp-ADSim"
    metadata = {}

    with open("gui/PVs.json", "r") as json_file:
        metadata = json.load(json_file)

    command = "pvapy-ad-sim-server -cn "+prefix+":Pva1:Image -nx 128 -ny 128 -fps 100 -rp 100 -rt 120"

    if metadata and metadata is not None:
        command += " -mpv "
        for i, pv in enumerate(metadata):
            command += f"ca://{prefix}:{metadata[pv]}"
            if i < len(metadata)-1:
                command += ","  
            i +=1
    print(command)
    
    #subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    main()