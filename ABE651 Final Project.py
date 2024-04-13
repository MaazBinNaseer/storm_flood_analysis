# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:52:16 2024

@author: miche
@author2: maazy
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


""" ------------------- Replacement of the Trace values to 0.005 inches --------------- """

def replace_trace_precip_values(precip_file_path):
    """
    Reads the precipitation data file, replaces precipitation values where the 'Measurement Flag' is 'T' with 0.005 inches,
    and counts the number of such replacements.

    Parameters
    ----------
    precip_file_path : str
        The file path for the precipitation data.

    Returns
    -------
    tuple
        A tuple containing the modified DataFrame and the count of trace replacements.
    """
    try:
        # Read the data file
        precip_data = pd.read_csv(precip_file_path)

        # Identify rows where the Measurement Flag is 'T'
        trace_rows = precip_data['Measurement Flag'] == 'T'

        # Count how many 'T' values are there
        trace_count = trace_rows.sum()

        # Replace 'HPCP' values where 'Measurement Flag' is 'T' with 0.005 inches
        precip_data.loc[trace_rows, 'HPCP'] = 0.005

        return precip_data, trace_count

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0


""" ----------------------------- Data Quality Checks -------------------------------- """

def DataQuality_check999(precip_file_path, tide_file_path):
    """
    Reads data from precipitation and tide files, removes entries where specific columns have the value 999,
    and keeps a count of how many such entries were removed.

    Parameters
    ----------
    precip_file : str
        The file path for the precipitation data.
    tide_file : str
        The file path for the tide data.

    Returns
    -------
    tuple of (int, int)
        A tuple containing the counts of removed 999 entries from precipitation and tide data, respectively.

    """
    # Read data files
    precip_data = pd.read_csv(precip_file_path)
    tide_data = pd.read_csv(tide_file_path)

    # Convert data to float to avoid string comparison issues
    precip_data['HPCP'] = pd.to_numeric(precip_data['HPCP'], errors='coerce')
    
    # Count and remove 999.99s in precipitation data
    precip_original_count = len(precip_data)
    precip_data_cleaned = precip_data[precip_data['HPCP'] != 999.99]
    precip_removed_count = precip_original_count - len(precip_data_cleaned)

    # Count and remove 999.99s in tide data
    # Assuming no change needed here as tide data was previously correct
    tide_original_count = len(tide_data)
    tide_data['Verified (ft)'] = pd.to_numeric(tide_data['Verified (ft)'], errors='coerce')
    tide_data_cleaned = tide_data[tide_data['Verified (ft)'] != 999.99]
    tide_removed_count = tide_original_count - len(tide_data_cleaned)

    return (precip_removed_count, tide_removed_count)

def DataQuality_checkBlanks(precip_file_path, tide_file_path):
    """
    Reads precipitation and tide data files, checks for blank (NaN or empty) entries in specific columns,
    and returns the count of such blanks for each file.

    Parameters
    ----------
    precip_file_path : str
        The file path for the precipitation data.
    tide_file_path : str
        The file path for the tide data.

    Returns
    -------
    tuple of (int, int)
        A tuple containing the counts of blank entries in specific columns of the precipitation and tide data, respectively.
    """
    try:
        # Read the data files
        precip_data = pd.read_csv(precip_file_path)
        tide_data = pd.read_csv(tide_file_path)

        # Check for blanks in precipitation data in 'HPCP' column
        precip_blanks_count = precip_data['HPCP'].isna().sum()

        # Check for blanks in tide data in 'Verified (ft)' column
        tide_blanks_count = tide_data['Verified (ft)'].isna().sum()

        return (precip_blanks_count, tide_blanks_count)

    except Exception as e:
        print(f"An error occurred: {e}")
        return (0, 0) 

def DataQuality_checkGross(precip_file_path, tide_file_path):
    """
    Reads precipitation and tide data files, checks for gross errors in specific columns based on defined conditions,
    and returns the count of such errors for each file.

    Parameters
    ----------
    precip_file_path : str
        The file path for the precipitation data.
    tide_file_path : str
        The file path for the tide data.

    Returns
    -------
    tuple of (int, int)
        A tuple containing the counts of gross error entries in specific columns of the precipitation and tide data, respectively.
    """
    try:
        # Read the data files
        precip_data = pd.read_csv(precip_file_path)
        tide_data = pd.read_csv(tide_file_path)

        # Check for gross errors in precipitation data
        # Gross error defined as values outside the range 0.0 inches to 0.4 inches
        precip_data['HPCP'] = pd.to_numeric(precip_data['HPCP'], errors='coerce')  # Ensure data is float
        precip_gross_errors = precip_data[(precip_data['HPCP'] < 0.0) | (precip_data['HPCP'] > 0.4)].shape[0]

        # Check for gross errors in tide data
        # Gross error defined as values outside the range -3 feet to 3 feet
        tide_data['Verified (ft)'] = pd.to_numeric(tide_data['Verified (ft)'], errors='coerce')  # Ensure data is float
        tide_gross_errors = tide_data[(tide_data['Verified (ft)'] < -3) | (tide_data['Verified (ft)'] > 3)].shape[0]

        return (precip_gross_errors, tide_gross_errors)

    except Exception as e:
        print(f"An error occurred: {e}")
        return (0, 0)  


""" ----------------------------------- Data Quality Check function Ends-------------------------------------- """
    
# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':
    PrecipfileName = "Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv"
    PrecipDataDF, ReplacedPrecipValuesDF = read_precip_data(PrecipfileName)
    
    Clipped_Precip_DataDF = ClipPrecipData( PrecipDataDF, "2005-01-01", "2005-12-31")
    
    TidefileName = "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv"
    TideDataDF, ReplacedTideValuesDF = read_tide_data( TidefileName )


    # 1. Data Quality Check for 999 Values in (Both the files)
    percip_data_removed, tide_data_reomved =  DataQuality_check999("Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv", "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv")
    #print(f"Removed {percip_data_removed} entries from precipitation data and {tide_data_reomved} entries from tide data.")

    # 2. Data Quality Check for Blank Values in (Both the files)
    precip_blanks, tide_blanks = DataQuality_checkBlanks("Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv", "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv")
    # print(f"Blank entries found: {precip_blanks} in precipitation data, {tide_blanks} in tide data.")

    # 3. Data Quality Check for Gross Values in (Both the files)
    precip_gross_errors, tide_gross_errors = DataQuality_checkGross("Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv", "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv")
    # print(f"Gross error entries found: {precip_gross_errors} in precipitation data, {tide_gross_errors} in tide data.")

    # Checking whether the replacement worked
    modified_precip_data, trace_replacements = replace_trace_precip_values("Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv")
    # print(f"Trace values replaced: {trace_replacements}")

    # modified_precip_data.to_csv("Datasets/Modified_Hourly_Precipitation_Data_Fort_Myers_FL.csv", index=False) 


