import runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
runpy.run_path(str(Path(__file__).with_name('independent_4k.py')),run_name='__main__')
