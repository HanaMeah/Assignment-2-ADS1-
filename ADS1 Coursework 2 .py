# -*- coding: utf-8 -*-
"""
Spyder Editor

MSc Data Science
Applied Data Science 1
Assignment 2 - 30 %
Hana Meah
16048117

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew

# functions


def GDPpercountry(datafilename):
    """

    Function to read my data csv file which also gives the output of my 2 
    transposed data frames

    """
    df_orig = pd.read_csv(datafilename)   
    columns = df_orig.columns[1:]
 #   print(columns)
    df_orig[columns] = df_orig[columns].apply(pd.to_numeric)
    
    
 #   print(df_orig)
 #   print(df_orig.dtypes)

    # Transposing my dataframe
    df_transposed = df_orig.transpose()
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed.iloc[1:]
    df_transposed = df_transposed.apply(pd.to_numeric)
 #  print(df_transposed.columns)
    
 #  print(df_transposed.dtypes)
    
    
    return df_transposed, df_orig

# main program
df_years, df_countries = GDPpercountry("GDPpercentage.csv")


# second data set I am comparing
# IMprtzs percentage
imports_df = pd.read_csv("ImportsofGoodsiNservices.csv")
#print(imports_df.head(11))
