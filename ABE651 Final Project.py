# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:52:16 2024

@author: miche
@author2: maazy
"""
#import geopandas as gpd
import matplotlib.pyplot as plt
#import contextily as ctx
#from shapely.geometry import Point
import pandas as pd
from scipy import stats
import numpy as np
import os  #os = operating system, which will be used to open files

    
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
        DataDF.sort_index() # Sort the index in ascending order
        #print(DataDF)
        return(DataDF)

    except FileNotFoundError:
        print(f"File not found: {PrecipfileName}")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file at {PrecipfileName} is empty.")
        return None

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    # Ensure the DatetimeIndex is sorted
    DataDF = DataDF.sort_index()
    
    DataDF = DataDF.loc[startDate:endDate]
    
    return DataDF

def read_tide_data( TidefileName ):
    '''
    This function takes a Tide file as input, reads the raw data in the file, 
    and returns two Panda DataFrames. The DataFrame index is the year, month 
    and day of the observation.

    Parameters
    ----------
    fileName : Input .CSV file including tide data

    Returns
    -------
    DataDF : A DataFrame that uses Date as the index, sorted in ascending order.
    ReplacedValuesDF : The initial DataFrame that will summarize the number of 
    corrections made for all checks. The name of each check is used as the index.

    '''
    try:
        # open and read the file
        DataDF = pd.read_csv(TidefileName,
                             header='infer',
                             index_col= [0],
                             parse_dates=[0])
        DataDF = DataDF.sort_index() # Sort the index in ascending order
        #print(DataDF)
    
        return(DataDF)

    except FileNotFoundError:
        print(f"File not found: {TidefileName}")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file at {TidefileName} is empty.")
        return None

def modified_tide_data_csv( DataDF, outFileName):
    '''
    This function will write the modified tide data to a .CSV file

    Parameters
    ----------
    DataDF : DataFrame from the function, read_tide_data

    outFileName : Output .CSV file name

    Returns
    -------
    None.

    '''
    if os.path.isfile(outFileName): #this if statement checks to see if the output file exists
        DataDF.to_csv(outFileName, mode='a', index=True, header=False)
        
    else:     
        DataDF.to_csv(outFileName)

def assign_season(date):
    '''
    This function returns a string of the season determined by the month of the date
    
    Inputs
    ----------
    date : Equivalent to the month of the date

    Returns
    -------
    str: Season
    '''
    
    month = date.month
    if month in [12, 1, 2]: # December to February
        #print('winter')
        return 'Winter'
    elif month in [3, 4, 5]: # March to May
        #print('spring')
        return 'Spring'
    elif month in [6, 7, 8]: # June to August 
        #print('summer')
        return 'Summer'
    else: # months 9, 10, 11 or September to November
        #print('fall')
        return 'Fall'
    
def precip_station_info( DataDF ):
    '''
    This function groups the station information for all the precipitation 
    stations, including latitude, longitude, and elevation
    
    Inputs
    ----------
    DataDF : The daily precipitation dataframe from all stations

    Returns
    -------
    df : The station info Pandas Dataframe grouped by station location
    
    '''
    #Step 1) Create a blank dataframe with station info
    station_info_df = pd.DataFrame(columns=["Station", "Latitude", "Longitude", "Elevation"])
    #print(df)

    # Step 1) Organize by Station # in DataDF Dataframe using the Grouby Method
    precip_data_df = DataDF.groupby('STATION')
    #print(DataDF)
    
    # Annual Average Total Water Depth(mm)
    # Step 1) Populate Station Info Dataframe
    station_info_df["Latitude"]= precip_data_df['LATITUDE'].mean()
    station_info_df["Longitude"]= precip_data_df['LONGITUDE'].mean()
    station_info_df["Elevation"]= precip_data_df['ELEVATION'].mean()
    
    #print(df)
    
    # Seasons
    # Step 1) Resample maximum daily precipitation by month
    seasonal_resample = precip_data_df['PRCP'].resample("MS").max()
    seasonal_df = pd.DataFrame(seasonal_resample)
    seasonal_df.reset_index(inplace=True) # removes the index
    seasonal_df['Date'] = pd.to_datetime(seasonal_df['DATE']) # Convert 'Date' column into panda date format
    seasonal_df.set_index('Date', inplace=True) # index the 'Date' column
    print(seasonal_df)
    
    # Step 2) Create a new column in seasonal_df DF to assign seasons for each date
    seasonal_df['Season'] = seasonal_df.index.map(assign_season)
    print("Seasonal DF:", seasonal_df)

    # Step 3) Initialize dictionaries to hold maximum precipitation values for each station
    seasonal_precip = {'Winter': {}, 'Spring': {}, 'Summer': {}, 'Fall': {}}

    # Step 4) Loop through each station and season to compute annual seasonal average Total VWC
    for station in station_info_df['Station'].unique():
        print("Station Info", station_info_df)
        station_data = seasonal_df[seasonal_df['STATION'] == station]
        print(station_data)
        
        if not station_data.empty: 
            for season in ['Winter', 'Spring', 'Summer', 'Fall']: 
                season_data = station_data[station_data['Season'] == season] 
                max_seasonal_precip = round(season_data['PRCP'].max(), 4) 
                seasonal_precip[season][station] = max_seasonal_precip 
                  
    
    # Step 5) Add the maximum seasonal precipitation into the station info df     
    station_info_df.loc[:,'Winter Precipitation (in.)'] = seasonal_precip['Winter']
    station_info_df.loc[:,'Spring Precipitation (in.)'] = seasonal_precip['Spring']
    station_info_df.loc[:,'Summer Precipitation (in.)'] = seasonal_precip['Summer']
    station_info_df.loc[:,'Fall Precipitation (in.)'] = seasonal_precip['Fall']
    station_info_df.reset_index(inplace=True) # removes the index and returns it as a column in the df
    
    print(station_info_df.head(10))
    print(station_info_df.tail(10))
    
    return station_info_df
""" ----------------------------- Data Quality Checks -------------------------------- """

def DataQuality_check9999(precip_data):
    '''
    Reads data from precipitation files and removes entries where 
    PRCP_ATTRIBUTES columns have the value , and keeps a count of how many 
    such entries were removed.

    Parameters
    ----------
    precip_file_path : str
        The precipitation dataframe returned from the read_precip_data function.

    Returns
    -------
    precip_data_cleaned : int
        Cleaned dataframe with removed 9999.
    precip_removed_count : int
        A tuple containing the counts of removed 9999 entries from precipitation.

    '''
    # Convert data to float to avoid string comparison issues
    precip_data['PRCP'] = pd.to_numeric(precip_data['PRCP'], errors='coerce')
    
    # Count and remove 999.99s in precipitation data
    precip_original_count = len(precip_data)
    precip_data_cleaned = precip_data[precip_data['PRCP'] != 9999]
    precip_removed_count = precip_original_count - len(precip_data_cleaned)


    return (precip_data_cleaned, precip_removed_count)

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
        #trace_rows = precip_data['PRCP_ATTRIBUTES'] == 'T'
        trace_rows = precip_data['PRCP_ATTRIBUTES'].isin(['T,,N'])

        # Count how many 'T' values are there
        trace_count = trace_rows.sum()

        # Replace 'PRCP' values where 'Measurement Flag' is 'T' with 0.005 inches
        precip_data.loc[trace_rows, 'PRCP'] = 0.005

        return(precip_data, trace_count)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0

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
        # Check for blanks in precipitation data in 'PRCP' column
        precip_blanks_count = precip_data['PRCP'].isna().sum()

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
        precip_data['PRCP'] = pd.to_numeric(precip_data['PRCP'], errors='coerce')  # Ensure data is float
        precip_gross_errors = precip_data[(precip_data['PRCP'] < 0) | (precip_data['PRCP'] > 6)].shape[0]

        # Check for gross errors in tide data
        # Gross error defined as values outside the range -3 feet to 3 feet
        tide_data['Verified (ft)'] = pd.to_numeric(tide_data['Verified (ft)'], errors='coerce')  # Ensure data is float
        tide_gross_errors = tide_data[(tide_data['Verified (ft)'] < -3) | (tide_data['Verified (ft)'] > 3)].shape[0]

        return (precip_gross_errors, tide_gross_errors)

    except Exception as e:
        print(f"An error occurred: {e}")
        return (0, 0)  
    
def DataQuality_checkZScore(precip_data, tide_data):
    '''
    This function removes outliers with Z Score in specific columns based on 
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
    None.

    '''
    #threshold_z = 2
    try:
        # Check for outliers in precipitation data
        precip_data['PRCP'] = pd.to_numeric(precip_data['PRCP'], errors='coerce')  # Ensure data is float
        z = np.abs(stats.zscore(precip_data['PRCP']))
        
        print("Z Score Test:", z) 
        #outlier_indices = np.where(z > threshold_z)[0]
        #no_outliers = precip_data.drop(outlier_indices)
        #print("Original DataFrame Shape:", precip_data.shape)
        #print("DataFrame Shape after Removing Outliers:", precip_data.shape)
    
        # Check for outliers in tide data
        # Gross error defined as values outside the range -3 feet to 3 feet
        tide_data['Verified (ft)'] = pd.to_numeric(tide_data['Verified (ft)'], errors='coerce')  # Ensure data is float
        modified_precip_data = 1
        modified_tide_data = 1
        precip_Zscore_count = 1
        tide_Zscore_count = 1
        return (modified_precip_data, modified_tide_data, precip_Zscore_count, tide_Zscore_count)

    except Exception as e:
        print(f"An error occurred: {e}")
        return (0, 0)  

         
    
    
    
    
""" ---------------------------- Summary table with data quality checking results ---------------------------- """

def ReplacedValuesDF( Precip_trace, Precip_check9999, Precip_checkBlanks, Tide_checkBlanks, Precip_gross_errors , Tide_gross_errors, Precip_checkZscore, Tide_checkZscore):
    # define and initialize the missing data frame
    ReplacedValuesDF = pd.DataFrame(0, index=["1. Replace Trace Values", "2. 9999", "3. Blanks","4. Gross Error", "5. Z-Score"], columns=["Precipitation", "Verified Tide"])
    #print(ReplacedValuesDF)
    ReplacedValuesDF.loc["1. Replace Trace Values", "Precipitation"] = Precip_trace
    ReplacedValuesDF.loc["1. Replace Trace Values", "Verified Tide"] = 0

    ReplacedValuesDF.loc["2. 9999", "Precipitation"] = Precip_check9999
    ReplacedValuesDF.loc["2. 9999", "Verified Tide"] = 0

    ReplacedValuesDF.loc["3. Blanks", "Precipitation"] = Precip_checkBlanks
    ReplacedValuesDF.loc["3. Blanks", "Verified Tide"] = Tide_checkBlanks
    
    ReplacedValuesDF.loc["4. Gross Error", "Precipitation"] = Precip_gross_errors
    ReplacedValuesDF.loc["4. Gross Error", "Verified Tide"] = Tide_gross_errors
    
    ReplacedValuesDF.loc["5. Z-Score", "Precipitation"] = Precip_checkZscore
    ReplacedValuesDF.loc["5. Z-Score", "Verified Tide"] = Tide_checkZscore
    
    
    
    ReplacedValuesDF.to_csv('ReplacedValuesDF.txt', sep='\t', index=True)
    
    return(ReplacedValuesDF)
""" ------------------------------------ Data Quality Graphical Analysis -------------------------------------- """

def plot_precipitation( plotData , title , outFileName ):
    '''
    This function plots the raw precipitation data values.

    Parameters
    ----------
    plotData : float
        The clipped tide data in a Pandas Dataframe.
    title : str
        Title of the plot
    outFileName : str
        Title of the output .PNG file name

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

def plot_tide( plotData , title , outFileName ):
    '''
    This function plots the raw tide data values.

    Parameters
    ----------
    plotData : float
        The clipped tide data in a Pandas Dataframe.
    title : str
        Title of the plot
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    plotData = pd.to_numeric(plotData, errors='coerce')  # Ensure data is float
    
    plt.figure(figsize=(10, 5))
    plotData.plot(use_index=True, color = 'b')
    plt.xlabel('Date', fontsize = 15) #x-axis label
    plt.ylabel('Tide (ft)', fontsize = 15) #y-axis label
    plt.title(title, fontsize = 20) #title of graph
    plt.savefig( outFileName )
    
def plot_check9999( original_plotData , new_plotData , title , outFileName ):
    '''
    This function plots each precipitation dataset before and after the check999
    correction has been made. 

    Parameters
    ----------
    original_plotData : float
        The raw data in a Pandas Dataframe.
    new_plotData : float
        The corrected data in a Pandas Dataframe.
    title : str
        Title of the plot
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10, 5))
    original_plotData.plot(use_index=True, color = 'b', label = 'Original Precipitation')
    new_plotData.plot(use_index=True, color = 'r', label = 'Removed 9999')
    plt.xlabel('Date', fontsize = 15) #x-axis label
    plt.ylabel('Precipitation (in)', fontsize = 15) #y-axis label
    plt.legend(fontsize = 8, loc = "upper right") #legend label
    plt.title(title, fontsize = 20) #title of graph
    plt.savefig( outFileName )    
    
def plot_trace( original_plotData , new_plotData , title , outFileName ):
    '''
    This function plots each precipitation dataset before and after the trace
    correction has been made. 

    Parameters
    ----------
    original_plotData : float
        The raw data in a Pandas Dataframe.
    new_plotData : float
        The corrected data in a Pandas Dataframe.
    title : str
        Title of the plot
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10, 5))
    original_plotData.plot(use_index=True, color = 'b', label = 'Original Precipitation')
    new_plotData.plot(use_index=True, color = 'r', label = 'Replaced Trace Values')
    plt.xlabel('Date', fontsize = 15) #x-axis label
    plt.ylabel('Precipitation (in)', fontsize = 15) #y-axis label
    plt.legend(fontsize = 8, loc = "upper right") #legend label
    plt.title(title, fontsize = 20) #title of graph
    plt.savefig( outFileName )    
    
    

    

    
"----------------------------------Plot the Map ---------------------------------------------------------------"

def plot_latitude_longitudeMap(data_frame):
    '''
    Plot points defined by latitude and longitude on a map using GeoPandas and Contextily.

    Parameters
    ----------
    data_frame : float
        DataFrame containing latitude and longitude columns

    Returns
    -------
    None.

    '''
    # Create a GeoDataFrame from the latitude and longitude data
    gdf = gpd.GeoDataFrame(
        data_frame,
        geometry=[Point(xy) for xy in zip(data_frame['LONGITUDE'], data_frame['LATITUDE'])],
        crs="EPSG:4326"  # WGS 84
    )

    # Convert to Web Mercator for contextily compatibility
    gdf = gdf.to_crs(epsg=3857)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, marker='o', color='red', markersize=50)

    # Add basemap
    ctx.add_basemap(ax)
    ax.set_axis_off()
    plt.savefig("MapDrawn.png")

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':
    # File location of Daily precipitation file
    #DailyPrecipfileName = "Datasets/Daily Precipitation Data/Daily Precipitation Data_Fort Myers_FL.csv"
    

    
   
    
    # File location of hourly precipitation file
    #PrecipfileName = "Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv"

    
    
    # Create Pandas Dataframe that contains tide values from 1998 to 2013
    TideDataDF_1998_2023 = read_tide_data( "Datasets/Hourly Tide Data/CO-OPS_8725520_1998-2023.csv")
    
    # Create Pandas Dataframe that contains daily precipitation values from 2003 to 2023
    DailyPrecipDataDF_2003_2023 = read_precip_data("Datasets/Daily Precipitation Data/Daily Precipitation Data_Fort Myers_FL.csv")
    
    
    # Group Precipitation Data by Station
    precip_station_info( DailyPrecipDataDF_2003_2023 )
    
    # Clip Precipitation and Tide Dataframes to given time periods (January 2004 to December 2023)
    PrecipDataDF = ClipData( DailyPrecipDataDF_2003_2023 , "2005-01-01", "2023-12-31")
    TideDataDF = ClipData( TideDataDF_1998_2023, "2004-01-01", "2023-12-31")
        
    # 1. Data Quality Check for 9999 Values in Precipitation Data
    precip_data_check9999, precip_data_removed =  DataQuality_check9999(PrecipDataDF)
    #print(f"Removed {precip_data_removed} entries from precipitation data")
    
    # 2. Replace trace precipitation values
    precip_data_trace_replace, trace_replacements = replace_trace_precip_values(precip_data_check9999)
    #print(f"Trace values replaced: {trace_replacements}")
    
    # 3. Data Quality Check for Blank Values in (Both the files)
    precip_blanks, tide_blanks = DataQuality_checkBlanks(precip_data_check9999, TideDataDF)
    #print(f"Blank entries found: {precip_blanks} in precipitation data, {tide_blanks} in tide data.")

    # 4. Data Quality Check for Gross Values in (Both the files)
    precip_gross_errors, tide_gross_errors = DataQuality_checkGross(precip_data_check9999, TideDataDF)
    #print(f"Gross error entries found: {precip_gross_errors} in precipitation data, {tide_gross_errors} in tide data.")


    #5. Data Quality Check for Outlier Values in (Both the files)
    modified_precip_data, modified_tide_data, precip_Zscore_count, tide_Zscore_count = DataQuality_checkZScore(precip_data_check9999, TideDataDF)    
    
    # modified_precip_data.to_csv("Datasets/Modified_Hourly_Precipitation_Data_Fort_Myers_FL.csv", index=False) 
    
    # modified_tide_data.to_csv("Datasets/Modified_Hourly_Tide_Data_Fort_Myers_FL.csv", index=False) 
    """ ----------------------------- Summary table with data quality checking results---------------------------- """

    #ReplacedValuesDF( trace_replacements, precip_data_removed, precip_blanks, tide_blanks, precip_gross_errors , tide_gross_errors )
    
    """ ----------------------------- Data Quality Graphical Analysis Starts-------------------------------------- """
    # Original Precipitation Plot
    #plot_precipitation( PrecipDataDF['PRCP'], 'Raw Precipitation Data', 'Raw Precipitation Data.png' )       
    
    # Original Tide Plot
    plot_tide(TideDataDF['Verified (ft)'], 'Raw Tide Data', 'Raw Tide Data.png')
    
    # Original Precipitation vs. Precipitation - Post 999 Check
    #plot_check9999(PrecipDataDF['PRCP'], precip_data_check9999['PRCP'], 'Precipitation Check 9999', 'Precipitation_Check9999.png' )
    plot_precipitation(precip_data_check9999['PRCP'],'Precipitation - Post 9999 Check', 'Precipitation - Post 9999 Check.png' )

    # Precipitation - Post 999 Check vs. Precipitation - Post Trace Replacement
    #plot_trace( precip_data_check9999['PRCP'] , precip_data_trace_replace['PRCP'] , 'Replace Trace Precipitation Values' , 'Precipitation_Trace_Replace.png' )
    plot_precipitation(precip_data_trace_replace['PRCP'],'Replace Trace Precipitation Values', 'Precipitation - Post Trace Replace.png' )

    # Original vs. Post Blank Values Check (Both the files)

    "----------------------------------Plot the Map ---------------------------------------------------------------"
    
   # data_sample = pd.read_csv('Datasets/Hourly Precipitation Data/Hourly Precipitation Data_Fort Myers_FL.csv')
   # unique_locs = data_sample.drop_duplicates(subset=['LATITUDE', 'LONGITUDE'])

    # plot_latitude_longitudeMap(unique_locs)
    
    '''# Combine yearly tide data into one .CSV file
    tide_fileName = {"Datasets/Hourly Tide Data/CO-OPS_8725520_1998.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_1999.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2000.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2001.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2002.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2003.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2004.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2005.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2006.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2007.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2008.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2009.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2010.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2011.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2012.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2013.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2014.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2015.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2016.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2017.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2018.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2019.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2020.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2021.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2022.csv",
                     "Datasets/Hourly Tide Data/CO-OPS_8725520_2023.csv"}
    
    for file in tide_fileName:
        tideDF = read_tide_data( file )
        modified_tide_data = modified_tide_data_csv( tideDF , "Datasets/Hourly Tide Data/CO-OPS_8725520_1998-2023.csv")'''
    
