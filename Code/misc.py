from pathlib import Path
import gzip
import shutil


# def extractzipfiles(p):
#     allzipfiles=[x for x in p.rglob('*.gz')]
#
#     for x in allzipfiles:
#         if x.exists():
#             with gzip.open(x.as_posix(), "rb") as f_in, open(x.parent.joinpath(x.stem).as_posix(), "wb") as f_out:
#                 shutil.copyfileobj(f_in, f_out)
#             x.unlink()
#
# # Path to zipfiles (can be train or validation data paths, change accordingly)
# zippath = Path('../../Data/MICCAI_BraTS2020_ValidationData')
# extractzipfiles(zippath)

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(10)+1
y = [1.023,0.897,0.888,0.856,0.852,0.804,0.816,0.797,0.766,0.773]
plt.figure()
plt.plot(x, y, "*-", linewidth=1.5)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Total Dice Loss')
plt.xlabel('Epochs')
plt.savefig("trainingloss.pdf", format='pdf', dpi=300, bbox_inches='tight')
pass