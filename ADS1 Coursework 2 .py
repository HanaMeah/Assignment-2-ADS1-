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

# def readmydata(df):
#     """

# This is a function to read my data files

    
# """
#     pd.read_csv(df)
#     return

# main program

df_GDP_per_country = pd.read_csv("GDPpercentage.csv")

df_GDP_per_country.dropna()

df_GDP_per_country.drop(
    df_GDP_per_country.columns[[0, 2, 3, 4, 5, -1]], axis=1, inplace=True)

#make a subset with the countries I want to look at in GDPpercentage
GDP_df = df_GDP_per_country.iloc[1:11]

print(GDP_df.head(11))

#second data set I am comparing
#Access to Clean Fuels and Technologies for Cooking and Rural percentage
cleanfuel_df = pd.read_csv("CleanFuelsTechpercentage.csv")
print(cleanfuel_df.head(11))




