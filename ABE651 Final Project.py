# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:52:16 2024

@author: miche
"""

import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
    
def read_precip_data( PrecipfileName ):
    '''
    This function takes a Precipitation file as input, reads the raw data in the file, 
    and returns two Panda DataFrames. The DataFrame index is the year, month 
    and day of the observation.

    Parameters
    ----------
    fileName : Input file = "Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv".

    Returns
    -------
    DataDF : A DataFrame that uses Date as the index.
    ReplacedValuesDF : The initial DataFrame that will summarize the number of 
    corrections made for all checks. The name of each check is used as the index.

    '''
    try:
        # open and read the file
        DataDF = pd.read_csv(PrecipfileName,
                             header='infer',
                             index_col= [5], 
                             parse_dates=[5])
        #print(DataDF)
        
        # define and initialize the missing data frame
        ReplacedValuesDF = pd.DataFrame(0, index=["1. No Data", "2. Gross Error", "3. Swapped","4. Range Fail"], columns=["HPCP"])
        #print(ReplacedValuesDF)
    
        return(DataDF, ReplacedValuesDF)

    except FileNotFoundError:
        print(f"File not found: {PrecipfileName}")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file at {PrecipfileName} is empty.")
        return None

def ClipPrecipData( DataDF, startDate, endDate ):
    """This function clips the given Preciptaiton time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    lenDataDF = len(DataDF)
    DataDF = DataDF['HPCP'].loc[startDate:endDate]
    
    print(DataDF)
    
    # quantify the number of missing values
    MissingValues = lenDataDF - len(DataDF)
    #print("Missing Values:" , MissingValues)
    
    return DataDF, MissingValues

def read_tide_data( TidefileName ):
    '''
    This function takes a Tide file as input, reads the raw data in the file, 
    and returns two Panda DataFrames. The DataFrame index is the year, month 
    and day of the observation.

    Parameters
    ----------
    fileName : Input file = "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv"

    Returns
    -------
    DataDF : A DataFrame that uses Date as the index.
    ReplacedValuesDF : The initial DataFrame that will summarize the number of 
    corrections made for all checks. The name of each check is used as the index.

    '''
    try:
        # open and read the file
        DataDF = pd.read_csv(TidefileName,
                             header='infer',
                             index_col= [0],
                             parse_dates=[0])
        #print(DataDF)
        
        # define and initialize the missing data frame
        ReplacedValuesDF = pd.DataFrame(0, index=["1. No Data", "2. Gross Error", "3. Swapped","4. Range Fail"], columns=["Verified Tide (ft)"])
        #print(ReplacedValuesDF)
    
        return(DataDF, ReplacedValuesDF)

    except FileNotFoundError:
        print(f"File not found: {TidefileName}")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file at {TidefileName} is empty.")
        return None


    
    
# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':
    PrecipfileName = "Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv"
    PrecipDataDF, ReplacedPrecipValuesDF = read_precip_data(PrecipfileName)
    
    Clipped_Precip_DataDF = ClipPrecipData( PrecipDataDF, "2005-01-01", "2005-12-31")
    
    TidefileName = "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv"
    TideDataDF, ReplacedTideValuesDF = read_tide_data( TidefileName )