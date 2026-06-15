"""
Single HPC consumer that chains the deterministic analysis pipeline:

    BlobTracker → FeatureExtractor

Both stages run in one process to avoid inter-process PVA channel hops.
The sub-processors' updateOutputChannel calls are suppressed so only the
pipeline's final enriched pvObject is published.

Output attributes added to each frame
--------------------------------------
BlobDetections, BlobDetCount, BlobTracks, BlobTrackCount  (from BlobTracker)
FeatureVector  (from FeatureExtractor — JSON with per-blob + global frame stats)

Usage:
    python -m pvapy.cli.hpcConsumer \\
        --input-channel  pvapy:image \\
        --output-channel sim:Pva1:Image \\
        --processor-file src/dashpva/consumers/hpc/analysis/hpc_analysis_pipeline.py \\
        --processor-class HpcAnalysisPipelineProcessor \\
        --n-consumers 1 --report-period 5
"""

from pvapy.hpc.adImageProcessor import AdImageProcessor

from dashpva.consumers.hpc.analysis.hpc_blob_tracking_consumer import (
    HpcBlobTrackingProcessor,
)
from dashpva.consumers.hpc.analysis.hpc_feature_extraction_consumer import (
    HpcFeatureExtractionProcessor,
)


def _noop(_pv):
    """Suppress mid-chain updateOutputChannel calls in sub-processors."""


class HpcAnalysisPipelineProcessor(AdImageProcessor):
    """
    Deterministic HPC pipeline: BlobTracker → FeatureExtractor.
    Fast, no external dependencies, no LLM calls.
    """

    def __init__(self, configDict={}):
        super().__init__(configDict)
        self._blob    = HpcBlobTrackingProcessor(configDict)
        self._feature = HpcFeatureExtractionProcessor(configDict)
        self._blob.updateOutputChannel    = _noop
        self._feature.updateOutputChannel = _noop

    def configure(self, configDict):
        self._blob.configure(configDict)
        self._feature.configure(configDict)

    def process(self, pvObject):
        # Broad try/except prevents any uncaught exception from propagating into
        # pvapy's C++ subscriber, which cannot handle Python exceptions safely.
        try:
            pvObject = self._blob.process(pvObject)
            pvObject = self._feature.process(pvObject)
        except Exception as e:
            try:
                self.logger.error(f"Pipeline error: {e}", exc_info=True)
            except Exception:
                print(f"[Pipeline] ERROR: {e}", flush=True)
        self.updateOutputChannel(pvObject)
        return pvObject

    def getStats(self):
        b = self._blob.getStats()
        f = self._feature.getStats()
        return {
            'blob_nFramesProcessed':    b['nFramesProcessed'],
            'blob_nFrameErrors':        b['nFrameErrors'],
            'feature_nFramesProcessed': f['nFramesProcessed'],
            'feature_nFrameErrors':     f['nFrameErrors'],
        }

    def resetStats(self):
        self._blob.resetStats()
        self._feature.resetStats()
