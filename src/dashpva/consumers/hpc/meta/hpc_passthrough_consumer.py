# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
import time

from pvapy.utility.timeUtility import TimeUtility

from dashpva.consumers.core.base_meta_associator import BaseMetaAssociator


class HpcPassthroughProcessor(BaseMetaAssociator):

    def __init__(self, configDict={}):
        super().__init__(configDict)
        self.logger.debug('Created HpcPassthroughProcessor')

    def process(self, pvObject):
        t0 = time.time()
        frameId = pvObject['uniqueId']
        dims = pvObject['dimension']
        nDims = len(dims)
        if not nDims:
            self.logger.debug(f'Frame id {frameId} contains an empty image.')
            return pvObject

        if 'timeStamp' not in pvObject:
            self.logger.error(f'Frame id {frameId} does not have field "timeStamp"')
            return pvObject

        frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        self.logger.debug(f'Frame id {frameId} timestamp: {frameTimestamp}')

        self.updateOutputChannel(pvObject)
        self.lastFrameTimestamp = frameTimestamp
        t1 = time.time()
        self.processingTime += (t1 - t0)
        return pvObject
