import time
import numpy as np
import pvaccess as pva
from pvaccess import PvObject
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
import logging

import xrayutilities as xu
import toml

class HpcRsmProcessor(AdImageProcessor):

    def __init__(self, configDict={}):
        super(HpcRsmProcessor, self).__init__(configDict)

        # Config Variables
        self.path = None
        self.hkl_config = None
        
        # Statistics
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.nMetadataProcessed = 0
        self.nMetadataDiscarded = 0
        self.processingTime = 0

        # PV attributes
        self.shape : tuple = (0,0)
        self.type_dict = {'value':{
            'qx': [pva.DOUBLE],
            'qy': [pva.DOUBLE],
            'qz': [pva.DOUBLE]}}                   
        
        # HKL parameters
        self.attributes = {}
        self.hkl_pv_channels = set()
        self.hkl_attributes = {}
        self.q_conv = None
        self.qx = None
        self.qy = None
        self.qz = None   

        self.configure(configDict)
       
        
    def configure(self, configDict):
        """Configure processor settings and initialize HKL parameters"""
        self.logger.debug(f'Configuration update: {configDict}')

        if 'path' in configDict:
            self.path = configDict["path"]
            with open(self.path, "r") as config_file:
                self.config = toml.load(config_file)
                
            if 'HKL' in self.config:
                self.hkl_config : dict = self.config['HKL']
                for section in self.hkl_config.values(): # every section holds a dict
                    for channel in section.values(): # the values of each seciton is the pv name string
                        self.hkl_pv_channels.add(channel)

    def parse_hkl_ndattributes(self, pva_object):
        """
        Parse the NDAttributes from the PVA Object into a python dict.
        Store attributes in self.attributes for easy reference.
        """
        if pva_object is None:
            return
        obj_dict : dict = pva_object.get()
        attributes : list = obj_dict.get('attribute', [])
        hkl_attributes = {}
        for attr in attributes: # list of attribute dictionaries
            name = attr['name']
            value = attr['value'][0]['value']
            self.attributes[name] = value
            if name in self.hkl_pv_channels:
                hkl_attributes[name] = value
        return hkl_attributes
    
    def get_sample_and_detector_circles(self, hkl_attr: dict):
        # lists for sample circle parameters
        sample_circle_directions = []
        sample_circle_positions = []
        # lists for detector circles
        det_circle_directions = []
        det_circle_positions = []

        if len(hkl_attr) == len(self.hkl_pv_channels):
            # loop sorting pv channels
            for section, pv_dict in self.hkl_config.items():
                if section.startswith('SAMPLE_CIRCLE'):
                    for pv_name in pv_dict.values():
                        if pv_name.endswith('DirectionAxis'):
                            sample_circle_directions.append(hkl_attr[pv_name])
                        elif pv_name.endswith('Position'):
                            sample_circle_positions.append(hkl_attr[pv_name])
                elif section.startswith('DETECTOR_CIRCLE'):
                    for pv_name in pv_dict.values():
                        if pv_name.endswith('DirectionAxis'):
                            det_circle_directions.append(hkl_attr[pv_name])
                        elif pv_name.endswith('Position'):
                            det_circle_positions.append(hkl_attr[pv_name])

        return sample_circle_directions, sample_circle_positions, det_circle_directions, det_circle_positions
    
    def get_axis_directions(self, hkl_attr: dict):
         # Get beam and reference directions
        if len(hkl_attr) == len(self.hkl_pv_channels):
            primary_beam_directions = [hkl_attr.get(f'PrimaryBeamDirection:AxisNumber{i}', None) for i in range(1,4)]
            inplane_beam_direction = [hkl_attr.get(f'PrimaryBeamDirection:AxisNumber{i}', None) for i in range(1,4)]
            sample_surface_normal_direction = [hkl_attr.get(f'SampleSurfaceNormalDirection:AxisNumber{i}', None) for i in range(1,4)]

            return primary_beam_directions, inplane_beam_direction, sample_surface_normal_direction
        else:
            return None, None, None

        
    def get_ub_matrix(self, hkl_attr: dict):
        ub_matrix_key = self.hkl_config['SPEC'].get('UB_MATRIX_VALUE', '')

        return hkl_attr[ub_matrix_key]
    
    def get_energy(self, hkl_attr: dict):
        energy_key = self.hkl_config['SPEC'].get('ENERGY_VALUE', '')

        return hkl_attr[energy_key]
      

    def create_rsm(self, hkl_attr: dict, shape: tuple):
        """Calculate reciprocal space mapping"""
        try:
            # get Sample and Detection Circle positions and directions from hkl attributes
            sample_circle_directions, sample_circle_positions, det_circle_directions, det_circle_positions = self.get_sample_and_detector_circles(hkl_attr)
            # get all axis directions for primary beam, inplane beam, and sample surface normal from hkl attributes
            primary_beam_directions, inplane_beam_direction, sample_surface_normal_direction = self.get_axis_directions(hkl_attr)
            # get UB matrix and energy
            ub_matrix = self.get_ub_matrix(hkl_attr)
            ub_matrix = np.reshape(ub_matrix, (3,3))
            energy = self.get_energy(hkl_attr) * 1000

            # Initialize QConversion
            q_conv = xu.experiment.QConversion(
                sample_circle_directions,
                det_circle_directions,
                primary_beam_directions
            )
            # Initialize HXRD
            hxrd = xu.HXRD(inplane_beam_direction, 
                        sample_surface_normal_direction, 
                        en=energy, 
                        qconv=q_conv)
            
            # Set up detector parameters
            roi = [0, shape[0], 0, shape[1]]
            pixel_dir1 = hkl_attr['DetectorSetup:PixelDirection1']
            pixel_dir2 = hkl_attr['DetectorSetup:PixelDirection2']
            cch1 = hkl_attr['DetectorSetup:CenterChannelPixel'][0]
            cch2 = hkl_attr['DetectorSetup:CenterChannelPixel'][1]
            nch1 = shape[0]
            nch2 = shape[1]
            pixel_width1 = hkl_attr['DetectorSetup:Size'][0] / nch1
            pixel_width2 = hkl_attr['DetectorSetup:Size'][1] / nch2
            distance = hkl_attr['DetectorSetup:Distance']

            hxrd.Ang2Q.init_area(
                pixel_dir1, pixel_dir2,
                cch1=cch1, cch2=cch2,
                Nch1=nch1, Nch2=nch2,
                pwidth1=pixel_width1,
                pwidth2=pixel_width2,
                distance=distance,
                roi=roi
            )

            angles = [*sample_circle_positions, *det_circle_positions]
            return hxrd.Ang2Q.area(*angles, UB=ub_matrix)
        except Exception as e:
            with open("error_output.txt", "w") as f:
                f.write(str(e))
            return None, None, None

    def process(self, pvObject):
        t0 = time.time()

        dims = pvObject['dimension']
        nDims = len(dims)
        if not nDims:
            # Frame has no image data
            return pvObject

        if 'timeStamp' not in pvObject:
            # No timestamp, just return the object
            return pvObject
        
        self.hkl_attributes = self.parse_hkl_ndattributes(pvObject)
        for key, val in self.hkl_attributes.items():
            self.logger.log(level=1, msg=f'{key}: {val}')
        
        self.shape = tuple([dim['size'] for dim in dims])
    
        qx, qy, qz = self.create_rsm(self.hkl_attributes, self.shape)
        
        if qx is not None:
            # Create RSM data structure
            rsm_data = {
                'qx': qx.flatten().tolist(),
                'qy': qy.flatten().tolist(),
                'qz': qz.flatten().tolist()
            }
            try:
                # Create attribute
                rsm_attribute = {'value': rsm_data}
            
                rsm_object = pva.PvObject(self.type_dict, rsm_attribute) 
                rsm_attribute = pva.NtAttribute('RSM', rsm_object)
                pvObject['attribute'] = [rsm_attribute,]
            except Exception as e:
                with open("error_output.txt", "w") as f:
                    f.writelines([str(np.shape(rsm_data['qx']))+'\n', str(type(rsm_data['qx']))+'\n', str(e)])
                return pvObject
            
            
            self.nFramesProcessed += 1
        else:
            self.nFrameErrors += 1



        # Update stats
        frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        self.lastFrameTimestamp = frameTimestamp
        self.nFramesProcessed += 1

        # Update output channel if needed
        self.updateOutputChannel(pvObject)

        # Update processing time
        t1 = time.time()
        self.processingTime += (t1 - t0)

        return pvObject


    def resetStats(self):
        """
        Reset processor statistics.
        """
        self.nFramesProcessed = 0 
        self.nFrameErrors = 0 
        self.nMetadataProcessed = 0 
        self.nMetadataDiscarded = 0 
        self.processingTime = 0

    def getStats(self):
        """
        Get current statistics of processing.
        """
        processedFrameRate = 0
        frameErrorRate = 0
        if self.processingTime > 0:
            processedFrameRate = self.nFramesProcessed / self.processingTime
            frameErrorRate = self.nFrameErrors / self.processingTime
        return { 
            'nFramesProcessed' : self.nFramesProcessed,
            'nFrameErrors' : self.nFrameErrors,
            'nMetadataProcessed' : self.nMetadataProcessed,
            'nMetadataDiscarded' : self.nMetadataDiscarded,
            'processingTime' : FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate' : FloatWithUnits(processedFrameRate, 'fps'),
            'frameErrorRate' : FloatWithUnits(frameErrorRate, 'fps')
        }

    def getStatsPvaTypes(self):
        """
        Define PVA types for different stats variables.
        """
        return { 
            'nFramesProcessed' : pva.UINT,
            'nFrameErrors' : pva.UINT,
            'nMetadataProcessed' : pva.UINT,
            'nMetadataDiscarded' : pva.UINT,
            'processingTime' : pva.DOUBLE,
            'processedFrameRate' : pva.DOUBLE,
            'frameErrorRate' : pva.DOUBLE
        }