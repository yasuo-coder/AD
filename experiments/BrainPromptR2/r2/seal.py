"""Freeze a reviewable implementation manifest before launching any formal fit."""
import platform
import numpy,torch,sklearn,scipy
from .common import *
def main():
    files={}
    for directory in ['r2','configs','tests','legacy']:
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and path.suffix in ['.py','.json','.md']:files[str(path.relative_to(ROOT))]=digest(path)
    write(ROOT/'public/source_manifest.json',dict(files=files,source_sha256=objhash(files),
      environment=dict(python=platform.python_version(),torch=torch.__version__,cuda_build=torch.version.cuda,numpy=numpy.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__),
      protocol_sha256=digest(ROOT/'configs/study.json'),data_manifest_sha256=digest(ROOT/'prepared/manifest.json')))
if __name__=='__main__':main()
