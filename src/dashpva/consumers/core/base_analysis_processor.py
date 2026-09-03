from dashpva.consumers.core.base_hpc import BaseHpcProcessor


class BaseAnalysisProcessor(BaseHpcProcessor):
    """Shared base for HPC analysis consumers (RSM, vectorized, spontaneous).

    Extends :class:`BaseHpcProcessor` with the image / ROI scaffolding and
    NDAttribute parsing common to analysis processors, so a new consumer only
    has to implement ``process``. Statistics, codec maps, compression helpers
    and the pvapy stats hooks are all inherited from the base.

    Example:
        class MyAnalysisProcessor(BaseAnalysisProcessor):
            def process(self, pvObject):
                self.parse_pva_ndattributes(pvObject)
                self.parse_image_data_type(pvObject)
                # ... compute analysis, append attributes ...
                self.updateOutputChannel(pvObject)
                return pvObject
    """

    DEFAULT_ROI_X = 0
    DEFAULT_ROI_Y = 0
    DEFAULT_ROI_WIDTH = 50
    DEFAULT_ROI_HEIGHT = 50

    def __init__(self, configDict={}):
        super().__init__(configDict)
        self.attributes = {}
        self.image = None
        self.shape = (0, 0)
        self.data_type = None
        self.roi_x = self.DEFAULT_ROI_X
        self.roi_y = self.DEFAULT_ROI_Y
        self.roi_width = self.DEFAULT_ROI_WIDTH
        self.roi_height = self.DEFAULT_ROI_HEIGHT

    def parse_image_data_type(self, pva_object):
        """Store the incoming image datatype (the active union field name)."""
        if pva_object is not None:
            self.data_type = list(pva_object['value'][0].keys())[0]

    def parse_pva_ndattributes(self, pva_object):
        """Parse the NDAttributes from the PVA object into ``self.attributes``."""
        if pva_object is None:
            return
        obj_dict = pva_object.get()
        attributes = {}
        for attr in obj_dict.get("attribute", []):
            attributes[attr['name']] = attr['value']
        self.attributes = attributes
