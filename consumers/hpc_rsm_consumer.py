import numpy as np
import pvaccess as pva
from pvaccess import PvObject
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
import xrayutilities as xu
import time

class HpcRsmProcessor(AdImageProcessor):
    def __init__(self, configDict={}):
        super(HpcRsmProcessor, self).__init__(configDict)
        
        # Statistics
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.processingTime = 0
        
        # HKL parameters
        self.attributes = None
        self.hkl_data = None
        self.q_conv = None
        self.shape = None
        self.qx = None
        self.qy = None
        self.qz = None  
        
        # Configure from dictionary
        self.configure(configDict)
        
    def configure(self, configDict):
        """Configure processor settings and initialize HKL parameters"""
        self.logger.debug(f'Configuration update: {configDict}')
        
        if "HKL" in configDict:
            self.hkl_data = configDict["HKL"]
            self.init_hkl()

    def parse_pva_ndattributes(self, pva_object):
        """
        Parse the NDAttributes from the PVA Object into a python dict.
        Store attributes in self.attributes for easy reference.
        """
        if pva_object is None:
            return
        obj_dict = pva_object.get()
        attributes = {}
        for attr in obj_dict.get("attribute", []):
            name = attr['name']
            value = attr['value']
            attributes[name] = value

        # Include additional values commonly found at top-level for completeness.
        for value_key in ["codec", "uniqueId", "uncompressedSize"]:
            if value_key in pva_object:
                attributes[value_key] = pva_object[value_key]

        self.attributes = attributes

    def init_hkl(self):
        """Initialize HKL parameters from config"""
        if self.hkl_data:
            # Get sample circle parameters
            sample_circle_keys = [key for key in self.hkl_data.keys() if key.startswith('SampleCircle')]
            self.sample_circle_directions = []
            self.sample_circle_positions = []
            for key in sample_circle_keys:
                self.sample_circle_directions.append(self.hkl_data[key]['DirectionAxis'])
                self.sample_circle_positions.append(self.hkl_data[key]['Position'])

            # Get detector circle parameters
            det_circle_keys = [key for key in self.hkl_data.keys() if key.startswith('DetectorCircle')]
            self.det_circle_directions = []
            self.det_circle_positions = []
            for key in det_circle_keys:
                self.det_circle_directions.append(self.hkl_data[key]['DirectionAxis'])
                self.det_circle_positions.append(self.hkl_data[key]['Position'])

            # Get beam and reference directions
            self.primary_beam_directions = [self.hkl_data['PrimaryBeamDirection'][axis] for axis in self.hkl_data['PrimaryBeamDirection'].keys()]
            self.inplane_beam_direction = [self.hkl_data['InplaneReferenceDirection'][axis] for axis in self.hkl_data['InplaneReferenceDirection'].keys()]
            self.sample_surface_normal_direction = [self.hkl_data['SampleSurfaceNormalDirection'][axis] for axis in self.hkl_data['SampleSurfaceNormalDirection'].keys()]

            # Initialize QConversion
            self.q_conv = xu.experiment.QConversion(
                self.sample_circle_directions,
                self.det_circle_directions,
                self.primary_beam_directions
            )

            # Get UB matrix and energy
            self.ub_matrix = np.reshape(self.hkl_data['UBMatrix']['Value'], (3,3))
            self.energy = self.hkl_data['Energy']['Value'] * 1000

    def create_rsm(self, shape):
        """Calculate reciprocal space mapping"""
        if not self.hkl_data or not self.q_conv:
            return None, None, None

        hxrd = xu.HXRD(self.inplane_beam_direction, 
                      self.sample_surface_normal_direction, 
                      en=self.energy, 
                      qconv=self.q_conv)

        # Set up detector parameters
        roi = [0, shape[0], 0, shape[1]]
        pixel_dir1 = self.hkl_data['DetectorSetup']['PixelDirection1']
        pixel_dir2 = self.hkl_data['DetectorSetup']['PixelDirection2']
        cch1 = self.hkl_data['DetectorSetup']['CenterChannelPixel'][0]
        cch2 = self.hkl_data['DetectorSetup']['CenterChannelPixel'][1]
        nch1 = shape[0]
        nch2 = shape[1]
        pixel_width1 = self.hkl_data['DetectorSetup']['Size'][0] / nch1
        pixel_width2 = self.hkl_data['DetectorSetup']['Size'][1] / nch2
        distance = self.hkl_data['DetectorSetup']['Distance']

        hxrd.Ang2Q.init_area(
            pixel_dir1, pixel_dir2,
            cch1=cch1, cch2=cch2,
            Nch1=nch1, Nch2=nch2,
            pwidth1=pixel_width1,
            pwidth2=pixel_width2,
            distance=distance,
            roi=roi
        )

        angles = [*self.sample_circle_positions, *self.det_circle_positions]
        return hxrd.Ang2Q.area(*angles, UB=self.ub_matrix)

    def process(self, pvObject):
        t0 = time.time()
        
        try:
            # Get frame dimensions
            if 'dimension' in pvObject:
                dims = pvObject['dimension']
                self.shape = tuple([dim['size'] for dim in dims])
            
                # Calculate RSM
                qx, qy, qz = self.create_rsm(self.shape)
                
                if qx is not None:
                    # Create RSM data structure
                    rsm_data = {
                        'qx': qx.tolist(),
                        'qy': qy.tolist(),
                        'qz': qz.tolist()
                    }
                    
                    # Create attribute
                    rsm_attribute = {
                        'name': 'RSM',
                        'value': [{'value': rsm_data}]
                    }
                    
                    # Get or create attribute list
                    if 'attribute' not in pvObject:
                        pvObject['attribute'] = []
                    
                    # Add RSM attribute
                    pvObject['attribute'].append(rsm_attribute)
                    
                    self.nFramesProcessed += 1
                else:
                    self.nFrameErrors += 1
            
            # Update output channel
            self.updateOutputChannel(pvObject)
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            self.nFrameErrors += 1
            
        t1 = time.time()
        self.processingTime += (t1 - t0)
        
        return pvObject

    def resetStats(self):
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.processingTime = 0

    def getStats(self):
        processedFrameRate = 0
        frameErrorRate = 0
        if self.processingTime > 0:
            processedFrameRate = self.nFramesProcessed / self.processingTime
            frameErrorRate = self.nFrameErrors / self.processingTime
        return {
            'nFramesProcessed': self.nFramesProcessed,
            'nFrameErrors': self.nFrameErrors,
            'processingTime': FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate': FloatWithUnits(processedFrameRate, 'fps'),
            'frameErrorRate': FloatWithUnits(frameErrorRate, 'fps')
        }

    def getStatsPvaTypes(self):
        return {
            'nFramesProcessed': pva.UINT,
            'nFrameErrors': pva.UINT,
            'processingTime': pva.DOUBLE,
            'processedFrameRate': pva.DOUBLE,
            'frameErrorRate': pva.DOUBLE
        } 