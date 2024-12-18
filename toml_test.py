import toml

def load_toml(path) -> dict:
    with open(path, 'r') as f:
        toml_data: dict = toml.load(f)
    return toml_data
    
if __name__ == "__main__":
    metadata_config_path = "/home/beams0/JULIO.RODRIGUEZ/Desktop/Lab Software/area_det_PVA_viewer/pv_configs/metadata_pvs.toml"
    if metadata_config_path != '':
            with open(metadata_config_path, 'r') as toml_file:
                # loads the pvs in the json file into a python dictionary
                pvs:dict = toml.load(toml_file)
                stats: dict = pvs["stats"]
                print(stats)


    # pv_config = load_toml(metadata_config_path)
    # roi_config: dict = pv_config.get("rois", {})
    # num_rois = len(roi_config)
    # roi_pvs = ""
    # for roi in roi_config.keys():
    #     roi_specific_pvs: dict = roi_config.get(roi, {})
    #     if roi_specific_pvs:
    #         for pv in roi_specific_pvs.keys():
    #             pv_channel = roi_specific_pvs.get(pv, "")
    #             roi_pvs += f"ca://{pv_channel},"

    # roi_pvs = roi_pvs.strip(',')


    # print(f"{roi_config} \n\nnum rois: {num_rois}\npv channels: {roi_pvs}")

    # metadata_config : dict = pv_config.get("metadata", {})
    # if metadata_config:
    #         ca = metadata_config.get("ca", {})
    #         pva = metadata_config.get("pva", {})
    #         ca_pvs = ""
    #         pva_pvs = ""

    #         if ca:
    #             for value in list(ca.values()):
    #                 ca_pvs += f"ca://{value},"
    #         if pva:
    #             for value in list(pva.values()):
    #                 pva_pvs += f"pva://{value},"

    # all_pvs = ca_pvs.strip(',') if not(pva_pvs) else ca_pvs + pva_pvs.strip(',')
    # print(all_pvs)