# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:52:16 2024

@author: miche
@author2: maazy
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
    
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

    '''
    try:
        # open and read the file
        DataDF = pd.read_csv(PrecipfileName,
                             header='infer',
                             index_col= [5], 
                             parse_dates=[5])
        #print(DataDF)
        return(DataDF)

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
        
        
    
        return(DataDF)

    except FileNotFoundError:
        print(f"File not found: {TidefileName}")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file at {TidefileName} is empty.")
        return None

""" ------------------- Replacement of the Trace values to 0.005 inches --------------- """

def replace_trace_precip_values(precip_data):
    '''
    This function replaces precipitation values where the 'Measurement Flag' 
    is 'T' with 0.005 inches, and counts the number of such replacements.

    Parameters
    ----------
    precip_data : str
        The precipitation dataframe returned from the read_precip_data function.

    Returns
    -------
    precip_data
        cleaned precipitation dataframe.
    trace_count
        number of trace replacements

    '''
    try:
        # Identify rows where the Measurement Flag is 'T'
        trace_rows = precip_data['Measurement Flag'] == 'T'

        # Count how many 'T' values are there
        trace_count = trace_rows.sum()

        # Replace 'HPCP' values where 'Measurement Flag' is 'T' with 0.005 inches
        precip_data.loc[trace_rows, 'HPCP'] = 0.005

        return(precip_data, trace_count)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0


""" ----------------------------- Data Quality Checks -------------------------------- """

def DataQuality_check999(precip_data):
    '''
    Reads data from precipitation and tide files, removes entries where 
    specific columns have the value 999.99, and keeps a count of how many 
    such entries were removed.

    Parameters
    ----------
    precip_file_path : str
        The precipitation dataframe returned from the read_precip_data function.

    Returns
    -------
    precip_data_cleaned : int
        Cleaned dataframe with removed 999.99.
    precip_removed_count : int
        A tuple containing the counts of removed 999.99 entries from precipitation.

    '''
    # Convert data to float to avoid string comparison issues
    precip_data['HPCP'] = pd.to_numeric(precip_data['HPCP'], errors='coerce')
    
    # Count and remove 999.99s in precipitation data
    precip_original_count = len(precip_data)
    precip_data_cleaned = precip_data[precip_data['HPCP'] != 999.99]
    precip_removed_count = precip_original_count - len(precip_data_cleaned)


    return (precip_data_cleaned, precip_removed_count)

def DataQuality_checkBlanks(precip_data, tide_data):
    '''
    This function checks for checks for blank (NaN or empty) entries 
    in specific columns in precipitation and tide data files,
    and returns the count of such blanks for each file.

    Parameters
    ----------
    precip_data : str
        The precipitation dataframe returned from the read_precip_data function.
    tide_data : str
        The tide dataframe returned from the read_tide_data function.

    Returns
    -------
    tuple of (int, int)
        A tuple containing the counts of blank entries in specific columns of the precipitation and tide data, respectively.

    '''
    try:
        # Check for blanks in precipitation data in 'HPCP' column
        precip_blanks_count = precip_data['HPCP'].isna().sum()

        # Check for blanks in tide data in 'Verified (ft)' column
        tide_blanks_count = tide_data['Verified (ft)'].isna().sum()


        return ( precip_blanks_count, tide_blanks_count )

    except Exception as e:
        print(f"An error occurred: {e}")
        return (0, 0)
    

def DataQuality_checkGross(precip_data, tide_data):
    '''
    This function checks for gross errors in specific columns based on 
    defined conditions in precipitation and tide data files, and returns the 
    count of such errors for each file.

    Parameters
    ----------
    precip_data : str
        The precipitation dataframe returned from the read_precip_data function.
    tide_data : str
        The tide dataframe returned from the read_tide_data function.

    Returns
    -------
    tuple of (int, int)
        A tuple containing the counts of gross error entries in specific 
        columns of the precipitation and tide data, respectively.

    '''
    try:
        # Check for gross errors in precipitation data
        # Gross error defined as values outside the range 0 inches to 4 inches
        precip_data['HPCP'] = pd.to_numeric(precip_data['HPCP'], errors='coerce')  # Ensure data is float
        precip_gross_errors = precip_data[(precip_data['HPCP'] < 0) | (precip_data['HPCP'] > 4)].shape[0]

        # Check for gross errors in tide data
        # Gross error defined as values outside the range -3 feet to 3 feet
        tide_data['Verified (ft)'] = pd.to_numeric(tide_data['Verified (ft)'], errors='coerce')  # Ensure data is float
        tide_gross_errors = tide_data[(tide_data['Verified (ft)'] < -3) | (tide_data['Verified (ft)'] > 3)].shape[0]

        return (precip_gross_errors, tide_gross_errors)

    except Exception as e:
        print(f"An error occurred: {e}")
        return (0, 0)  

""" ----------------------------------- Data Quality Check function Ends-------------------------------------- """
""" ---------------------------- Summary table with data quality checking results ---------------------------- """

def ReplacedValuesDF( Precip_trace, Precip_check999, Precip_checkBlanks, Tide_checkBlanks, Precip_gross_errors , Tide_gross_errors):
    # define and initialize the missing data frame
    ReplacedValuesDF = pd.DataFrame(0, index=["1. Replace Trace Values", "2. 999.99", "3. Blanks","4. Gross Error"], columns=["Precipitation", "Verified Tide"])
    #print(ReplacedValuesDF)
    ReplacedValuesDF.loc["1. Replace Trace Values", "Precipitation"] = Precip_trace
    ReplacedValuesDF.loc["1. Replace Trace Values", "Verified Tide"] = 0

    ReplacedValuesDF.loc["2. 999.99", "Precipitation"] = Precip_check999
    ReplacedValuesDF.loc["2. 999.99", "Verified Tide"] = 0

    ReplacedValuesDF.loc["3. Blanks", "Precipitation"] = Precip_checkBlanks
    ReplacedValuesDF.loc["3. Blanks", "Verified Tide"] = Tide_checkBlanks
    
    ReplacedValuesDF.loc["4. Gross Error", "Precipitation"] = Precip_gross_errors
    ReplacedValuesDF.loc["4. Gross Error", "Verified Tide"] = Tide_gross_errors
    
    ReplacedValuesDF.to_csv('ReplacedValuesDF.txt', sep='\t', index=True)
    
    return(ReplacedValuesDF)
""" ------------------------------------ Data Quality Graphical Analysis -------------------------------------- """

def plot_precipitation( plotData , title , outFileName ):
    '''
    This function plots each precipitation dataset before and after correction 
    has been made. 

    Parameters
    ----------
    precip_file_path : str
    The file path for the precipitation data.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10, 5))
    plotData.plot(use_index=True, color = 'b')
    plt.xlabel('Date', fontsize = 15) #x-axis label
    plt.ylabel('Precipitation (in)', fontsize = 15) #y-axis label
    plt.title(title, fontsize = 20) #title of graph
    plt.savefig( outFileName )

def plot_check999( original_plotData , new_plotData , title , outFileName ):
    '''
    This function plots each precipitation dataset before and after correction 
    has been made. 

    Parameters
    ----------
    precip_file_path : str
    The file path for the precipitation data.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10, 5))
    original_plotData.plot(use_index=True, color = 'b', label = 'Original Precipitation')
    new_plotData.plot(use_index=True, color = 'r', label = 'Removed 999.99')
    plt.xlabel('Date', fontsize = 15) #x-axis label
    plt.ylabel('Precipitation (in)', fontsize = 15) #y-axis label
    plt.legend(fontsize = 8, loc = "upper right") #legend label
    plt.title(title, fontsize = 20) #title of graph
    plt.savefig( outFileName )    
    
def plot_tide( plotData , title , outFileName ):
    '''
    This function plots each precipitation dataset before and after correction 
    has been made. 

    Parameters
    ----------
    precip_file_path : str
    The file path for the precipitation data.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10, 5))
    plotData.plot(use_index=True, color = 'b')
    plt.xlabel('Date', fontsize = 15) #x-axis label
    plt.ylabel('Tide (ft)', fontsize = 15) #y-axis label
    plt.title(title, fontsize = 20) #title of graph
    plt.savefig( outFileName )
    
# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':
    PrecipfileName = "Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv"
    PrecipDataDF = read_precip_data(PrecipfileName)
    
    Clipped_Precip_DataDF = ClipPrecipData( PrecipDataDF, "2005-01-01", "2005-12-31")

    TidefileName = "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv"
    TideDataDF = read_tide_data( TidefileName )
    
    # 1. Data Quality Check for 999.99 Values in Precipitation Data
    precip_data_check999, precip_data_removed =  DataQuality_check999(PrecipDataDF)
    #print(f"Removed {precip_data_removed} entries from precipitation data")
    
    # 2. Data Quality Check for Blank Values in (Both the files)
    precip_blanks, tide_blanks = DataQuality_checkBlanks(precip_data_check999, TideDataDF)
    #print(f"Blank entries found: {precip_blanks} in precipitation data, {tide_blanks} in tide data.")

    # 3. Data Quality Check for Gross Values in (Both the files)
    precip_gross_errors, tide_gross_errors = DataQuality_checkGross(precip_data_check999, TideDataDF)
    #print(f"Gross error entries found: {precip_gross_errors} in precipitation data, {tide_gross_errors} in tide data.")

    # Checking whether the replacement worked
    modified_precip_data, trace_replacements = replace_trace_precip_values(precip_data_check999)
    #print(f"Trace values replaced: {trace_replacements}")

    # modified_precip_data.to_csv("Datasets/Modified_Hourly_Precipitation_Data_Fort_Myers_FL.csv", index=False) 
    """ ----------------------------- Summary table with data quality checking results---------------------------- """

    ReplacedValuesDF( trace_replacements, precip_data_removed, precip_blanks, tide_blanks, precip_gross_errors , tide_gross_errors)
    
    """ ----------------------------- Data Quality Graphical Analysis Starts-------------------------------------- """
    # Original Precipitation Plots 
    plot_precipitation( PrecipDataDF['HPCP'], 'Precipitation_All', 'Precipitation_All.png' )       
    plot_precipitation( Clipped_Precip_DataDF[0] , 'Precipitation_2015', 'Precipitation_2015.png' )

    plot_tide( TideDataDF['Verified (ft)'] , 'Tide - 2015' , 'Tide_2015.png' )
    
    # Original Precipitation vs. Precipitation - Post 999 Check
    plot_check999( PrecipDataDF['HPCP'] , precip_data_check999['HPCP'] , 'Precipitation Check 999.99', 'Precipitation_check999.png' )
    plot_precipitation( precip_data_check999['HPCP'] , 'Precipitation - Post 999 Check', 'Precipitation - Post 999 Check.png' )

    # Original vs. Post Blank Values Check (Both the files)
