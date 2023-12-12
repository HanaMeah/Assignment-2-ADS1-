# -*- coding: utf-8 -*-
"""
Spyder Editor

MSc Data Science
Applied Data Science 1
Assignment 2 - 30 %
Hana Meah
16048117

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# functions


# main program

df_GDP_per_country = pd.read_csv("GDPpercentage.csv")

df_GDP_per_country.dropna()

df_GDP_per_country.drop(
    df_GDP_per_country.columns[[0, 2, 3]], axis=1, inplace=True)

n = df_GDP_per_country.iloc[0:211]

print(n.head(211))
#print(df_GDP_per_country.head(160))


