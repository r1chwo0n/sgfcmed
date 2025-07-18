"""
"""
from .sgfcmed_parallel import SGFCMedParallel

__all__ = ['SGFCMedParallel']

def __version__():
    """Return the version of the SGFCMed algorithm"""
    return "0.0.1"

def describe():
    """Print a description of the package and its features."""
    description = (
        "SGFCMed algorithm Library\n"
        "Version: {}\n"
        "Provides basic statistical calculations including:\n"
        "  - Mean\n"
        "  - Median\n"
        "  - Mode\n"
        "  - Standard Deviation\n"
    ).format(__version__())
    print(description)