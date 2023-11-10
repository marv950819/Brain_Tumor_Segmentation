from pathlib import Path
import gzip
import shutil


def extractzipfiles(p):
    allzipfiles=[x for x in p.rglob('*.gz')]

    for x in allzipfiles:
        if x.exists():
            with gzip.open(x.as_posix(), "rb") as f_in, open(x.parent.joinpath(x.stem).as_posix(), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            x.unlink()

# Path to zipfiles (can be train or validation data paths, change accordingly)
zippath = Path('../../Data/MICCAI_BraTS2020_ValidationData')
extractzipfiles(zippath)