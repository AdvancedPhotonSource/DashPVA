import json
import subprocess


def main():

    prefix = "dp-ADSim"
    metadata = {}

    with open("gui/PVs.json", "r") as json_file:
        metadata = json.load(json_file)

    command = f"pvapy-hpc-collector --collector-id 1 --producer-id-list 1 --input-channel {prefix}:Pva1:Image --control-channel collector:*:control"
    command += " --status-channel collector:*:status --output-channel collector:*:output --processor-file hpc_dpADSim.py --processor-class HpcAdMetadataProcessor --report-period 10"
    command += " --server-queue-size 100 --collector-cache-size 100 --monitor-queue-size 1000"

    if metadata and metadata is not None:
        command += " --metadata-channels "
        for i, pv in enumerate(metadata):
            command += f"ca://{prefix}:{metadata[pv]}"
            if i < len(metadata)-1:
                command += ","  
            i +=1
    # command += " --log-level debug"
    print(command)
    
    subprocess.run(args=command, shell=True)

if __name__ == "__main__":
    main()