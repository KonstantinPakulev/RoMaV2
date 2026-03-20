from .scannet1500 import ScanNet1500 as ScanNet1500
from .mega1500 import Mega1500 as Mega1500
try:
    from .wxbs import WxBSBenchmark as WxBSBenchmark
except ImportError:
    pass
try:
    from .satast import SatAst as SatAst
except ImportError:
    pass
