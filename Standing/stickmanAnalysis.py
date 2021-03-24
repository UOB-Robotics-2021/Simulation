# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 02:40:41 2021

@author: remib
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import pandas as pd

df = pd.read_csv('StickmanData.csv')



plt.figure()
plt.plot(df["total_energy"], linestyle="None", marker="x")
plt.minorticks_on()
plt.grid(which="both")
#plt.savefig("Log-Log.png", dpi=1200, format="png")
plt.legend()
plt.show()

