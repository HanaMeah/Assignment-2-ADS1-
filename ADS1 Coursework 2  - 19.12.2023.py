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


def reading_my_data(datafilename):
    """

    Genertal function to read all of my data csv files which also 
    gives the output of my original dataframe and my transposed data frame 
    as well as cleans it

    """
    df_orig = pd.read_csv(datafilename)
    columns = df_orig.columns[1:]

    df_orig[columns] = df_orig[columns].apply(pd.to_numeric)

    # Transposing my dataframe
    df_transposed = df_orig.transpose()
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed.iloc[1:]
    df_transposed = df_transposed.apply(pd.to_numeric)

    return df_transposed, df_orig


def summary_stats(df):
    """
    This function produces the summary statistics of my dataset anually per
    country when called using my transposed dataset for GDP.

    """

    summary_stats1 = df.describe()

    print(summary_stats1)

    return


def skew(df):
    """ 

    Calculates the centralised and normalised skewness of my data 

    """

    # calculating mean, std
    mean = np.mean(df)
    std = np.std(df)

    # calculating skewness per country
    value1 = np.sum(((df-mean) / std)**3) / len(df-2)
    print(value1)

    return value1


def kurtosis(df):
    """ 

    Calculates the centralised and normalised excess kurtosis of my data 

    """

    # calculates average and std, dev for centralising and normalising
    mean = np.mean(df)
    std = np.std(df)

    # now calculate the kurtosis
    value2 = np.sum(((df-mean) / std)**4) / len(df-3) - 3.0
    print(value2)

    return value2


def plot_kurt(df):
    """
    This function produces the kurtosis plot for the dataset.

    """

    plt.figure(1)
    plt.hist([kurtosis(df)], bins=10, edgecolor='black')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Data')

    plt.legend(['kurtosis'])

    return


def plot_skewness(df):
    """
    This function produces the kurtosis and skewness plot for the dataset.

    """

    plt.figure(2)
    plt.hist([skew(df)], bins=10, edgecolor='black')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Data')

    plt.legend(['skewness'])


def lineplotGDP(df):
    """

    This function creates a lineplot of my transposed data for GDP

    """

    plt.figure(3)
    df.reset_index().plot(x='index')

    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Year")
    plt.ylabel("GDP in USD $")
    plt.title("GDP in my selection of countries")
    return


def GDP_bar_plot_2015_GDP(df):
    """

    This function plots GDP data from the csv file and plots a 2015 bar graph.


    """

    plt.figure(4)
    plt.bar(df["Country Name"],  df["2015"])

    plt.xlabel("Countries")
    plt.ylabel("GDP ")
    plt.title("GDP in 2015")
    plt.xticks(rotation=45)

    return


def lineplot_pop_MEDC(df):
    """

    This function creates a lineplot More Economically Developed countries 
    for population data

    """

    plt.figure(6)

    plt.plot(df["Year"], df["France"], label="France")
    plt.plot(df["Year"], df["Germany"], label="Germany")
    plt.plot(df["Year"], df["United Kingdom"], label="United Kingdom")
    plt.plot(df["Year"], df["United States"], label="United States")

    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("Population in MEDC countries")
    return


def lineplot_pop_all(df):
    """

    This function creates a lineplot all countries for population data

    """

    plt.figure(7)

    plt.plot(df["Year"], df["Albania"], label="Albania")
    plt.plot(df["Year"], df["Ecuador"], label="Ecuador")
    plt.plot(df["Year"], df["Egypt"], label="Egypt")
    plt.plot(df["Year"], df["Ghana"], label="Ghana")
    plt.plot(df["Year"], df["Myanmar"], label="Myanmar")
    plt.plot(df["Year"], df["Vietnam"], label="Vietnam")
    plt.plot(df["Year"], df["France"], label="France")
    plt.plot(df["Year"], df["Germany"], label="Germany")
    plt.plot(df["Year"], df["United Kingdom"], label="United Kingdom")
    plt.plot(df["Year"], df["United States"], label="United States")

    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("Population in all countries")
    return


def piechart_labour_force(df, countries, year):
    """

    Pie Chart defnining proportion of Labour Force in each country for a 
    chosen year in my dataset

    """

    plt.figure(8)

    plt.pie(df, labels=countries, autopct='%1.1f%%')
    plt.title("Labour Force in " + year)
    return


def column_bar_plot_Arable_Land():
    """

    Function creates a column bar plot for Arable Land Square km data for each
    of my chosen countries

    """

    df = pd.read_csv("ArableLandSquareKM.csv")

    blue_bar_2015 = np.array(df.iloc[1, 1:8])
    orange_bar_2018 = np.array(df.iloc[4, 1:8])
    green_bar_2022 = np.array(df.iloc[-1, 1:8])

    N = 7
    ind = np.arange(N)
    plt.figure(figsize=(10, 5))
    width = 0.3

    plt.bar(ind, blue_bar_2015, width, label="2015")
    plt.bar(ind + width, orange_bar_2018, width, label="2018")
    plt.bar(ind + 2*width, green_bar_2022, width, label="2022")

    plt.xlabel("Countries in Selection")
    plt.ylabel("Land in Square km")
    plt.title("Change of Arable Land Square km in key countries")

    countries = ["Egypt", "France", "Germany", "Ghana", "Myanmar", "UK",
                 "Vietnam"]

    plt.xticks(ind + width, (countries))

    plt.legend(loc='best')
    plt.show()
    return


# main program starts here


df_GDP_transposed, df_GDP_orig = reading_my_data("GDPpercapitaUSD.csv")

summary_stats(df_GDP_transposed)  # summary statistics for GDP

value1 = skew(df_GDP_transposed)  # skewness for GDP
valu2 = kurtosis(df_GDP_transposed)  # kurtosis for GDP
plot_kurt(df_GDP_transposed)  # Kurtosis plot
plot_skewness(df_GDP_transposed)  # Skewness plot

GDP_bar_plot_2015_GDP(df_GDP_orig)

lineplotGDP(df_GDP_transposed)


df_pop_transposed, df_pop_orig = reading_my_data("PopulationTransposed.csv")

lineplot_pop_MEDC(df_pop_orig)  #call line plot function for MEDC countries
lineplot_pop_all(df_pop_orig)   ##call line plot function for all countries


# making the pie charts in my report. I ran each call line separately 
# as individual liness to view both pie charts

df_labourforce_transposed, df_labourforce_orig = reading_my_data(
    "LabourForceTotal.csv")

df_labour_force_2022 = df_labourforce_transposed.iloc[-1, 2:11] #subset 2022
df_labour_force_2015 = df_labourforce_transposed.iloc[1, 2:11]  #subset 2015

countries = ["Egypt", "France", "Germany", "Myanmar", "United Kingdom",
             "United States", "Vietnam"]

piechart_labour_force(df_labour_force_2022, countries, "2022")
piechart_labour_force(df_labour_force_2015, countries, "2015")
