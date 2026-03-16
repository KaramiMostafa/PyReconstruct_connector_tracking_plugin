"""
pyrecon_connector
=================
PyReconstruct plugin that bridges the cell-tracker-core library with
PyReconstruct's series/section data model.

Public API
----------
PyReconConnector   -- reads PyReconstruct ROIs, runs tracking, renames
                      contours and writes outputs back.
"""

from .connector import PyReconConnector

__version__ = "0.1.0"
__all__ = ["PyReconConnector"]
