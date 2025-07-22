import time
import copy
import toml
import bitshuffle
import blosc2
import lz4.block
import numpy as np
import pvaccess as pva
import xrayutilities as xu
from pvaccess import PvObject, NtAttribute
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
# logging
import traceback

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

        # Type Mapping
        self.CODEC_PARAMETERS_MAP = {
            np.dtype('uint8'): pva.UBYTE,
            np.dtype('int8'): pva.BYTE,
            np.dtype('uint16'): pva.USHORT,
            np.dtype('int16'): pva.SHORT,
            np.dtype('uint32'): pva.UINT,
            np.dtype('int32'): pva.INT,
            np.dtype('uint64'): pva.ULONG,
            np.dtype('int64'): pva.LONG,
            np.dtype('float32'): pva.FLOAT,
            np.dtype('float64'): pva.DOUBLE,

        }

        # PV attributes
        self.shape : tuple = (0,0)
        self.type_dict = {
            'codec':{
                'name': pva.STRING, 
                'parameters': pva.INT},
            'qx': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.DOUBLE]},
            'qy': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.DOUBLE]},
            'qz': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.DOUBLE]}
            }
        
        compressed_dtype = self.CODEC_PARAMETERS_MAP[np.dtype('uint8')] 
        
        self.type_dict_compressed = {
            'codec':{
                'name': pva.STRING, 
                'parameters': pva.INT},
            'qx': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[compressed_dtype]},
            'qy': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[compressed_dtype]},
            'qz': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[compressed_dtype]}
            }                   
        
        # HKL parameters
        self.all_attributes = {}
        self.hkl_pv_channels = set()
        self.hkl_attributes = {}
        self.old_attrbutes : dict = None
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
        Store attributes in self.all_attributes for easy reference.
        """
        if pva_object is None:
            return
        # obj_dict : dict = pva_object.get()
        attributes : list = pva_object['attribute']
        hkl_attributes = {}
        for attr in attributes: # list of attribute dictionaries
            name = attr['name']
            value = attr['value'][0]['value']
            self.all_attributes[name] = value
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
            with open("error_output1.txt", "w") as f:
                f.write(str(e))
            return None, None, None
        
    def attributes_diff(self, hkl_attr: dict, old_attr: dict) -> bool:
        # if len(previous_data) != len(metadata):
        #         dicts_equal = False
        #     else:   
        for key, value in hkl_attr.items():
            if isinstance(value, np.ndarray):
                arrs_equal = np.array_equal(value, old_attr[key])
                if not arrs_equal:
                    return True
            elif old_attr[key] != hkl_attr[key]:
                return True
        else:
            return False
        
    def compress_array(self, hkl_array: np.ndarray, codec_name: str) -> np.ndarray:
        if not isinstance(hkl_array, np.ndarray):
            raise TypeError("hkl_array must be a numpy array")
        if hkl_array.ndim != 1:
            raise ValueError("hkl_array must be a 1D numpy array")
        byte_data = hkl_array.tobytes()
        typesize = hkl_array.dtype.itemsize

        if codec_name == 'lz4':
            compressed = lz4.block.compress(byte_data)
        elif codec_name == 'bslz4':
            compressed = bitshuffle.compress_lz4(hkl_array)
        elif codec_name == 'blosc':
            compressed = blosc2.compress(
                byte_data,
                typesize=typesize
            )
        else:
            raise ValueError(f"Unsupported codec: {codec_name}")

        # Convert compressed bytes to a uint8 numpy array
        return np.frombuffer(compressed, dtype=np.uint8)

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
        
        if 'attribute' not in pvObject:
            print('attributes not in pvObject')
            return pvObject
                
        self.hkl_attributes = self.parse_hkl_ndattributes(pvObject)
        self.shape = tuple([dim['size'] for dim in dims])
    
        if self.old_attrbutes is not None:
            attributes_diff = self.attributes_diff(self.hkl_attributes, self.old_attrbutes)
        else:
            attributes_diff = True
        self.old_attrbutes = copy.deepcopy(self.hkl_attributes)

        if attributes_diff:
            # Only recalculate qxyz if there are new attributes
            qxyz = self.create_rsm(self.hkl_attributes, self.shape)
            self.qx = np.ravel(qxyz[0])
            self.qy = np.ravel(qxyz[1])
            self.qz = np.ravel(qxyz[2])
            self.codec_name = pvObject['codec']['name']
            self.original_dtype = self.qx.dtype if self.qx.dtype == self.qy.dtype == self.qz.dtype else np.dtype('float64')
            self.codec_parameters = int(self.CODEC_PARAMETERS_MAP.get(self.original_dtype, None)) if self.codec_name else -1
            self.uncompressed_size = np.prod(self.shape) * self.original_dtype.itemsize
            self.compressed_size_qx = self.uncompressed_size
            self.compressed_size_qy = self.uncompressed_size
            self.compressed_size_qz = self.uncompressed_size
            
            if self.codec_name != '':
                self.qx = self.compress_array(self.qx, self.codec_name)
                self.qy = self.compress_array(self.qy, self.codec_name)
                self.qz = self.compress_array(self.qz, self.codec_name)
                self.compressed_size_qx = self.qx.shape[0]
                self.compressed_size_qy = self.qy.shape[0]
                self.compressed_size_qz = self.qz.shape[0]

   
        try:
            # Create RSM data structure
            rsm_data = {
                        'codec':{
                            'name': self.codec_name, 
                            'parameters': self.codec_parameters},
                        'qx': {
                            'compressedSize': int(self.compressed_size_qx),
                            'uncompressedSize': int(self.uncompressed_size),
                            'value':self.qx},
                        'qy': {
                            'compressedSize': int(self.compressed_size_qy),
                            'uncompressedSize': int(self.uncompressed_size),
                            'value':self.qy},
                        'qz': {
                            'compressedSize': int(self.compressed_size_qz),
                            'uncompressedSize': int(self.uncompressed_size),
                            'value':self.qz},
                        } 
            
            if self.codec_name != '':
                rsm_object = {'name': 'RSM', 'value': PvObject({'value': self.type_dict_compressed}, {'value': rsm_data})}
            else:
                rsm_object = {'name': 'RSM', 'value': PvObject({'value': self.type_dict}, {'value': rsm_data})}

                

            # pv_attribute = NtAttribute('RSM', rsm_object)

            frameAttributes = pvObject['attribute']
            frameAttributes.append(rsm_object)
            pvObject['attribute'] = frameAttributes
            self.nFramesProcessed += 1
 
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

        except Exception as e:
            self.nFrameErrors += 1
            with open("error_output2.txt", "w") as f:
                f.writelines([''.join(traceback.format_exception(None, e, e.__traceback__))])
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