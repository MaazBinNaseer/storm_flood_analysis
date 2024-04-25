# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:52:16 2024

@author: miche
@author2: maazy
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.dates import DateFormatter
from shapely.geometry import Point
import pandas as pd
from scipy import stats
import numpy as np
import os  #os = operating system, which will be used to open files
import seaborn as sns #visualization library for statistical graphics
    
def read_precip_data( PrecipfileName ):
    '''
    This function takes a Tide file as input, reads the raw data in the file, 
    and returns a Panda DataFrame. The DataFrame index is the year, month 
    and day of the observation.
    
    Parameters
    ----------
    PrecipfileName : Input .CSV file including daily precipitation data
    
    Returns
    -------
    DataDF : A DataFrame that uses Date as the index, sorted in ascending order. 

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
    '''
    This function clips the given time series dataframe to a given range 
    of dates

    Parameters
    ----------
    DataDF : Input dataframe with date indexed.
    
    startDate : Str: ('YYYY-MM-DD')
        Start date of clip data
    endDate : Str: ('YYYY-MM-DD')
        End date of clip data
    Returns
    -------
    DataDF : Clipped dataframe from the start to end date.

    '''
    # Ensure the DatetimeIndex is sorted
    DataDF = DataDF.sort_index()
    
    DataDF = DataDF.loc[startDate:endDate]
    
    return DataDF

def read_tide_data( TidefileName ):
    '''
    This function takes a Tide file as input, reads the raw data in the file, 
    and returns a Panda DataFrame. The DataFrame index is the year, month 
    and day of the observation.

    Parameters
    ----------
    TidefileName : Input .CSV file including tide data

    Returns
    -------
    DataDF : A DataFrame that uses Date as the index, sorted in ascending order.

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
    This function will write yearly tide data into a single .CSV file

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
    
def precip_station_info( data_df ):
    '''
    This function groups the station information for all the precipitation 
    stations, including latitude, longitude, and elevation.

    Parameters
    ----------
    data_df : DataFrame
        The daily precipitation dataframe from all stations

    Returns
    -------
    station_info : DataFrame
        The station info Pandas Dataframe grouped by station location.

    '''
    # Create new DataFrame called station_info_DF
    station_info = pd.DataFrame(columns = ['Latitude', 'Longitude', 'Elevation'])
    
    # Group data_df by station
    station_info['Latitude'] = data_df.groupby(["STATION"])['LATITUDE'].mean()
    station_info['Longitude'] = data_df.groupby(["STATION"])['LONGITUDE'].mean()
    station_info['Elevation'] = data_df.groupby(["STATION"])['ELEVATION'].mean()
    
    # Elevation Analysis
    '''avg_elevation = np.mean(station_info['Elevation'])
    max_elevation = np.max(station_info['Elevation'])
    min_elevation = np.min(station_info['Elevation'])
    
    
    print("The average elevation of all stations is:", avg_elevation)
    print("The minimum elevation of all stations is:", max_elevation)
    print("The maximum elevation of all stations is:", min_elevation)'''

    #print("Station Info:\n", station_info_DF)
    station_info.to_csv('Precipitation_Station_Info.csv', sep=',')
    
    
    return station_info

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
    
def seasonal_precip( data_df ):
    '''
    This function groups the precipitation data by station and mean precipitation

    Parameters
    ----------
    data_df : DataFrame
        The daily precipitation dataframe from all stations

    Returns
    -------
    seasonal_precip : DataFrame
        Pandas Dataframe grouped by station location and mean precipitation.

    '''
    # Convert index to datetime if not already
    data_df.index = pd.to_datetime(data_df.index)
    
    # Assign season for each row
    data_df['SEASON'] = data_df.index.to_series().apply(assign_season)
    
    # Aggregate precipitation by season and station
    seasonal_precip = data_df.groupby(['STATION', 'SEASON'])['PRCP'].sum().reset_index()
    
    return seasonal_precip

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
   
    precip_data = precip_data.copy()
    # Convert data to float to avoid string comparison issues
    precip_data.loc[:, 'PRCP'] = pd.to_numeric(precip_data['PRCP'], errors='coerce')

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
        # Ensure 'PRCP_ATTRIBUTES' is a string and handle missing data
        precip_data['PRCP_ATTRIBUTES'] = precip_data['PRCP_ATTRIBUTES'].astype(str)
        mask = precip_data['PRCP_ATTRIBUTES'].str.startswith('T')
        
        # Filter out where mask is true
        cleaned_data = precip_data[~mask]
        trace_count = mask.sum()

        return (cleaned_data, trace_count)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0

def DataQuality_checkBlanks(precip_data, tide_data):
    '''
    This function checks for blank (NaN or empty) entries in specific columns 
    in precipitation and tide data files, and returns the count of such 
    blanks for each file.

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

        return (precip_blanks_count, tide_blanks_count )

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
        # Gross error defined as values outside the range 0 inches to 25 inches
        precip_data.loc[:, 'PRCP'] = pd.to_numeric(precip_data['PRCP'], errors='coerce')
        precip_gross_errors = precip_data[(precip_data['PRCP'] < 0) | (precip_data['PRCP'] > 25)].shape[0]

        # Check for gross errors in tide data
        # Gross error defined as values outside the range -4 feet to 4 feet
        tide_data['Verified (ft)'] = pd.to_numeric(tide_data['Verified (ft)'], errors='coerce')  # Ensure data is float
        tide_gross_errors = tide_data[(tide_data['Verified (ft)'] < -4) | (tide_data['Verified (ft)'] > 4)].shape[0]

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
    try:
        precip_data = precip_data.copy()
        # Check and count outliers in precipitation data
        precip_data['PRCP'] = pd.to_numeric(precip_data['PRCP'], errors='coerce')
        precip_in_range = precip_data[(precip_data['PRCP'] >= 0) & (precip_data['PRCP'] <= 25)]
        z_precip = np.abs(stats.zscore(precip_in_range['PRCP'].dropna()))
        precip_outliers = np.where(z_precip > 3)[0]  # Using 3 as a threshold for Z-score
        precip_Zscore_count = len(precip_outliers)

        # Check and count outliers in tide data
        tide_data['Verified (ft)'] = pd.to_numeric(tide_data['Verified (ft)'], errors='coerce')
        tide_in_range = tide_data[(tide_data['Verified (ft)'] >= -4) & (tide_data['Verified (ft)'] <= 4)]
        z_tide = np.abs(stats.zscore(tide_in_range['Verified (ft)'].dropna()))
        tide_outliers = np.where(z_tide > 3)[0]  # Using 3 as a threshold for Z-score
        tide_Zscore_count = len(tide_outliers)

        return (precip_Zscore_count, tide_Zscore_count)

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
    This function plots precipitation data and saves the plot to a PNG file.

    Parameters
    ----------
    plotData : Dataframe
        The precipitation data in a Pandas Dataframe.
    title : str
        Title of the plot
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    plotData.plot(use_index=True, color = 'b')
    
    # Set plot details
    ax.set_xlabel('Date', fontsize = 15)
    ax.set_ylabel('Precipitation (in)', fontsize = 15) #y-axis label
    ax.set_title(title, fontsize = 20) #title of graph
    
    # Save the figure
    plt.savefig( outFileName )
    plt.close(fig)  # Close the plot figure to free up memory 

def plot_tide( plotData , title , outFileName ):
    '''
    This function plots tide data and saves the plot to a PNG file.

    Parameters
    ----------
    plotData : Dataframe
        The tide data in a Pandas Dataframe.
    title : str
        Title of the plot
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 6))
   
    # Plot data
    plotData = pd.to_numeric(plotData, errors='coerce')  # Ensure data is float
    plotData.plot(use_index=True, color = 'b')
    
    # Set plot details
    ax.set_xlabel('Date', fontsize = 15)
    ax.set_ylabel('Tide (ft)', fontsize = 15) #y-axis label
    ax.set_title(title, fontsize = 20) #title of graph
    
    # Save the figure
    plt.savefig( outFileName )
    plt.close(fig)  # Close the plot figure to free up memory 
    
def plot_checked_data( original_plotData , new_plotData , y_label, title , outFileName ):
    '''
    This function plots each precipitation dataset before and after the check999
    correction has been made and saves the plot to a PNG file.

    Parameters
    ----------
    original_plotData : Dataframe
        The raw data in a Pandas Dataframe.
    new_plotData : Dataframe
        The corrected data in a Pandas Dataframe.
    title : str
        Title of the plot
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    original_plotData.plot(use_index=True, color='g', label = 'Original Data')
    new_plotData.plot(use_index=True, color='b', alpha=0.75, label = 'Checked Data')
    
    # Set plot details
    ax.set_xlabel('Date', fontsize = 15)
    ax.set_ylabel(y_label, fontsize = 15) #y-axis label
    ax.set_title(title, fontsize = 20) #title of graph
    plt.legend(fontsize = 8, loc = "upper right") #legend label
    
    # Save the figure
    plt.savefig( outFileName )
    plt.close(fig)  # Close the plot figure to free up memory 

def plot_seasonaltrends(seasonal_precip, outFileName):
    '''
    This function plots the seasonal trends of precipitation data and 
    saves the plot to a PNG file.
    
    Parameters:
    ----------
    seasonal_precip : DataFrame
        DataFrame containing the total precipitation for each station and season.
    '''
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        # Filter data for the season
        season_data = seasonal_precip[seasonal_precip['SEASON'] == season]
        # Calculate average precipitation per season across all stations
        average_precip = season_data.groupby('SEASON')['PRCP'].sum()
        ax.bar(season, average_precip.values, label=season)

    # Set plot details
    ax.set_xlabel('Season', fontsize = 15)
    ax.set_ylabel('Precipitation (in)', fontsize = 15)
    ax.set_title('Sum of Seasonal Precipitation Across All Stations', fontsize = 20)

    # Save the figure
    plt.savefig( outFileName , format='png')
    plt.close(fig)  # Close the plot figure to free up memory

def plot_mean_monthly_precip( DataDF, station, outFileName):
    '''
    This function calculates the maximum monthly precipitation, creates a plot, and 
    saves the plot to a PNG file.

    Parameters
    ----------
    DataDF : Dataframe
        The Precipitation data in a Pandas Dataframe.
    station : str
        Prcipitation Station Name.
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    # Filter dataframe by station
    DataDF = DataDF[(DataDF["STATION"] == station)]
    
    # create empty dataframe "plotData"
    plotData = pd.DataFrame(columns=['Average Monthly Flow'])
    
    # Find the Average Monthly Precipitation
    plotData['Average Monthly Flow'] = DataDF.groupby(DataDF.index.month)['PRCP'].mean(numeric_only=True)

    # Plot data
    plotData['Average Monthly Flow'].plot(label = station , alpha=0.5) 
    
    # Set plot details
    plt.tick_params('x',labelrotation = 0)
    plt.xlabel('Month', fontsize = 15)
    plt.ylabel('Precipitation (in)', fontsize = 15) #y-axis label
    plt.title("Average Monthly Precipitation at All Stations", fontsize = 20) #title of graph
    plt.legend(fontsize = 8, bbox_to_anchor=(1.04, 1), borderaxespad=0)
    
    # Save the figure
    plt.savefig( outFileName, bbox_inches='tight' )
    #plt.close()  # Close the plot figure to free up memory
    

def plot_precip_hist( DataDF, outFileName ):
    '''
    This function plots the distribution of daily precipitation data and 
    saves the plot to a PNG file.

    Parameters
    ----------
    DataDF : Dataframe
        The precipitation data in a Pandas Dataframe.
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    plotData = DataDF.loc[DataDF['PRCP'] > 2]
    
    sns.histplot(plotData['PRCP'], bins = 100)
    
    # Setting the labels and title
    plt.xlabel('Precipitation (in)', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.suptitle("Distribution of Daily Precipitation > 2 in. at All Stations", fontsize=20)

    # Saving the figure
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()  # Close the plot figure to free up memory
    
def plot_tide_hist( DataDF, outFileName ):
    '''
    This function plots the distribution of hourly tide data and 
    saves the plot to a PNG file.

    Parameters
    ----------
    DataDF : Dataframe
        The tide data in a Pandas Dataframe.
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    sns.histplot(DataDF['Verified (ft)'], kde = True, bins = 100, color='g')
    
    # Setting the labels and title
    plt.xlabel('Tide (ft)', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.suptitle("Distribution of Hourly Tide", fontsize=20)

    # Saving the figure
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()  # Close the plot figure to free up memory    
    
def plot_extreme_precip( DataDF, outFileName):
    '''
    This function find the days where daily precipitation > 3 in, creates a plot, and 
    saves the plot to a PNG file.

    Parameters
    ----------
    DataDF : Dataframe
        The Precipitation data in a Pandas Dataframe.
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    # Create a dataframe that groups data by Station
    plotData = DataDF.groupby('STATION')

    # Rename the columns to reflect the structure after reset_index()
    plotData.columns = ['STATION', 'DATE', 'PRCP']
    plotData = DataDF.loc[DataDF['PRCP'] > 3].reset_index()

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for station, group in plotData.groupby('STATION'):
        plt.scatter(group['DATE'], group['PRCP'], label = station, alpha=0.5) 
        
    # Formatting the x-axis dates
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    fig.autofmt_xdate()

    # Setting the labels and title
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Precipitation (in)', fontsize=15)
    plt.title("Daily Precipitation > 3 in. at All Stations", fontsize=20)
    plt.legend(fontsize=8, bbox_to_anchor=(1.04,1), borderaxespad=0)

    # Saving the figure
    plt.savefig(outFileName, bbox_inches='tight')
    plt.close()  # Close the plot figure to free up memory
    
def plot_max_daily_tide( DataDF , outFileName):
    '''
    This function calculates the maximum monthly tide, creates a plot, and 
    saves the plot to a PNG file.

    Parameters
    ----------
    DataDF : Dataframe
        The tide data in a Pandas Dataframe.
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''    
    # Find the Maximum Monthly Tide
    plotData = DataDF['Verified (ft)'].resample("D").max(numeric_only=True).reset_index()
    
    # Rename the columns to reflect the structure after reset_index()
    plotData.columns = ['Date', 'Max Tide']

    # Ensure 'DATE' is in datetime format
    plotData['Date'] = pd.to_datetime(plotData['Date'])

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    plt.plot(plotData['Date'], plotData['Max Tide'], c = 'b')
    
    # Set plot details
    plt.xlabel('Date', fontsize = 15)
    plt.ylabel('Tide (ft.)', fontsize = 15) #y-axis label
    plt.title("Maximum Daily Tide", fontsize = 20) #title of graph
    
    # Save the figure
    plt.savefig( outFileName )
    plt.close(fig)  # Close the plot figure to free up memory

def plot_3Dprecip_rolling_sum( DataDF , outFileName ):
    '''
    This function calculates the 3-day rolling sum of the precipitation at 
    each station, creates a plot, and saves the plot to a PNG file.

    Parameters
    ----------
    DataDF : Dataframe
        The precipitation data in a Pandas Dataframe.
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    # Perform the rolling sum, then reset the index to turn the MultiIndex into columns
    rolling_df = DataDF.groupby('STATION')['PRCP'].rolling(window=3).sum().reset_index()

    # Rename the columns to reflect the structure after reset_index()
    rolling_df.columns = ['STATION', 'DATE', 'Rolling Sum']

    # Drop NaN values resulting from the rolling calculation
    rolling_df = rolling_df.dropna(subset=['Rolling Sum'])

    # Ensure 'DATE' is in datetime format
    rolling_df['DATE'] = pd.to_datetime(rolling_df['DATE'])

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for station, group in rolling_df.groupby('STATION'):
        plt.scatter(group['DATE'], group['Rolling Sum'],label = station, alpha=0.5)

    # Formatting the x-axis dates
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # Setting the labels and title
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Precipitation (in)', fontsize=15)
    plt.title("3-day Rolling Precipitation Sum at All Stations", fontsize=20)
    plt.legend(fontsize=8, bbox_to_anchor=(1.04,1), borderaxespad=0)

    # Saving the figure
    plt.savefig(outFileName, bbox_inches='tight')


def combined_tide_precip( tideDataDF, precipDataDF, outFileName ):
    '''
    This function calculates the 3-day rolling sum of the precipitation > 6 in. 
    at each station, and plots 2 subplots. The first subplot is of the daily 
    precipitation > 3 in. and rolling sum of precipitation > 6 in.
    The second subplot will be of maximum daily tide. The plot is saved to a 
    PNG file. 

    Parameters
    ----------
    tideDataDF : Dataframe
        The tide data in a Pandas Dataframe.
    precipDataDF : Dataframe
        The precipitation data in a Pandas Dataframe.
    outFileName : str
        Title of the output .PNG file name

    Returns
    -------
    None.

    '''
    # Step 1: Create a dataframe for daily precipitation that groups data by Station
    extreme_plotData = precipDataDF.groupby('STATION')

    # Rename the columns to reflect the structure after reset_index()
    extreme_plotData.columns = ['STATION', 'DATE', 'PRCP']
    
    # Filter the daily precipitation > 3 in.
    extreme_plotData = precipDataDF.loc[precipDataDF['PRCP'] > 3].reset_index() 
    
    # Step 2: Perform the rolling sum, then reset the index to turn the MultiIndex into columns
    rolling_df = precipDataDF.groupby('STATION')['PRCP'].rolling(window=3).sum().reset_index()
    
    # Rename the columns to reflect the structure after reset_index()
    rolling_df.columns = ['STATION', 'DATE', 'Rolling Sum']

    # Drop NaN values resulting from the rolling calculation
    rolling_df = rolling_df.dropna(subset=['Rolling Sum'])
    
    #Filter only Rolling Sum > 6
    rolling_df = rolling_df.loc[rolling_df['Rolling Sum'] > 6]
    
    # Ensure 'DATE' is in datetime format
    rolling_df['DATE'] = pd.to_datetime(rolling_df['DATE'])
    
    # Step 3: Find the Maximum Monthly Tide
    tideplotData = tideDataDF['Verified (ft)'].resample("D").max(numeric_only=True).reset_index()
    
    # Rename the columns to reflect the structure after reset_index()
    tideplotData.columns = ['Date', 'Max Tide']
    
    # Ensure 'DATE' is in datetime format
    tideplotData['Date'] = pd.to_datetime(tideplotData['Date'])
    
    # Start plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols= 1, sharex=True, figsize=(10, 6))
    ax1 = plt.subplot(211) #211 stands for: 2 rows, 1 column, and first (1) subplot
    for station, group in extreme_plotData.groupby('STATION'):
        plt.scatter(group['DATE'], group['PRCP'], marker = 'o', label = station, alpha=0.5)
    for station, group in rolling_df.groupby('STATION'):
        plt.scatter(group['DATE'], group['Rolling Sum'], marker = '^', label = station, alpha=0.5)
    
    # Setting the labels and title
    plt.ylabel('Precipitation (in)', fontsize=15)
    plt.suptitle("Daily Precipitation > 3 in (circle marker)", fontsize=20)
    plt.title("3-Day Cumulative Precipitation > 6 in (triangle marker)", fontsize=20)
    #plt.legend(fontsize=8, bbox_to_anchor=(1.2,1), borderaxespad=0)
    
    ax2 = plt.subplot(212) #211 stands for: 2 rows, 1 column, and second (2) subplot
    plt.plot(tideplotData['Date'], tideplotData['Max Tide'], c = 'b', alpha=0.5)
    
    # Setting the labels and title
    plt.xlabel('Date', fontsize = 15)
    plt.ylabel('Tide (ft.)', fontsize = 15) #y-axis label
    plt.title("Maximum Daily Tide", fontsize = 20) #title of graph
    
    ax1.set_xticklabels([])
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # Saving the figure
    plt.savefig(outFileName, bbox_inches='tight')


    "----------------------------------Plot the Map ---------------------------------------------------------------"

def plot_latitude_longitudeMap(data_frame):
    '''
    This function plots the location of the precipitation stations defined by 
    latitude and longitude on a map using GeoPandas and Contextily.

    Parameters
    ----------
    data_frame : DataFrame
        DataFrame containing latitude and longitude columns

    Returns
    -------
    None.

    '''
    if 'Latitude' not in data_frame.columns or 'Longitude' not in data_frame.columns:
        raise ValueError("DataFrame must contain 'Latitude' and 'Longitude' columns.")

    # Create a GeoDataFrame from the latitude and longitude data
    gdf = gpd.GeoDataFrame(
        data_frame,
        geometry=[Point(xy) for xy in zip(data_frame['Longitude'], data_frame['Latitude'])],
        crs="EPSG:4326"  # WGS 84
    )

    # Convert to Web Mercator for contextily compatibility
    gdf = gdf.to_crs(epsg=3857)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, marker='o', color='red', markersize=5)
    
    # Add basemap
    ctx.add_basemap(ax, zoom=12)
    ax.set_axis_off()
    plt.savefig("MapDrawn.png")
    plt.close(fig)  # Close the plot figure to free up memory


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':   
    """ ----------------------------- Original Tide Data -------------------------------------- """
    # Create Pandas Dataframe that contains tide values from 1998 to 2013
    TideDataDF_1998_2023 = read_tide_data( "Datasets/Hourly Tide Data/CO-OPS_8725520_1998-2023.csv")
    
    # Original Tide Plot
    plot_tide(TideDataDF_1998_2023['Verified (ft)'], 'Raw Tide Data', 'Figures/Raw Tide Data.png')
    
    # Clip Tide Dataframe to given time periods (January 2004 to December 2023)
    TideDataDF = ClipData( TideDataDF_1998_2023, "2004-01-01", "2023-12-31")
    
    # Clipped Tide Plot
    plot_tide(TideDataDF['Verified (ft)'], 'Tide Data from 2004-2023', 'Figures/Tide Data_2004_2023.png')
    
    """ ----------------------------- Original Precipitation Data-------------------------------------- """
    # Create Pandas Dataframe that contains daily precipitation values from 2003 to 2023
    DailyPrecipDataDF_2003_2023 = read_precip_data("Datasets/Daily Precipitation Data/Daily Precipitation Data_Fort Myers_FL.csv")

    # Original Precipitation Plot
    plot_precipitation( DailyPrecipDataDF_2003_2023['PRCP'], 'Raw Precipitation Data', 'Figures/Raw Precipitation Data.png' )
    
    # Create Pandas Dataframe that summarizes Precipitation Stations Info and creates .CSV file
    precip_station_info_df = precip_station_info( DailyPrecipDataDF_2003_2023 )
    
    # Clip Precipitation Dataframes to given time periods (January 2004 to December 2023)
    PrecipDataDF = ClipData( DailyPrecipDataDF_2003_2023 , "2004-01-01", "2023-12-31")
    
    # Clipped Precipitation Plot
    plot_precipitation( PrecipDataDF['PRCP'], 'Precipitation Data from 2004-2023', 'Figures/Precipitation Data_2004_2023.png' )
    
    """ ----------------------------- Data Quality Checking -------------------------------------- """
    # 1. Data Quality Check for 9999 Values in Precipitation Data
    precip_data_check9999, precip_data_removed =  DataQuality_check9999(PrecipDataDF)
    
    # 2. Replace trace precipitation values
    precip_data_trace_replace, trace_replacements = replace_trace_precip_values(precip_data_check9999)
    
    # 3. Data Quality Check for Blank Values in (Both the files)
    precip_blanks, tide_blanks = DataQuality_checkBlanks(precip_data_check9999, TideDataDF)

    # 4. Data Quality Check for Gross Values in (Both the files)
    precip_gross_errors, tide_gross_errors = DataQuality_checkGross(precip_data_check9999, TideDataDF)

    #5. Data Quality Check for Outlier Values in (Both the files)
    precip_Zscore_count, tide_Zscore_count = DataQuality_checkZScore(precip_data_check9999, TideDataDF)    
    
    # modified_precip_data.to_csv("Datasets/Modified_Hourly_Precipitation_Data_Fort_Myers_FL.csv", index=False) 
    
    # modified_tide_data.to_csv("Datasets/Modified_Hourly_Tide_Data_Fort_Myers_FL.csv", index=False) 
    """ ----------------------------- Summary table with data quality checking results---------------------------- """
    ''' Create Pandas Dataframe that summarizes number of data quality checks and creates .CSV file'''
    ReplacedValuesDF( trace_replacements, precip_data_removed, precip_blanks, tide_blanks, precip_gross_errors , tide_gross_errors, precip_Zscore_count, tide_Zscore_count)
    
    """ -----------------------------  Graphical Data Analysis -------------------------------------- """

    ''' Original Precipitation vs. Precipitation - Post Trace Replacement '''
    #plot_checked_data( PrecipDataDF['PRCP'] , precip_data_trace_replace['PRCP'] , 'Precipitation (in.)', 'Precipitation - Trace Replace' , 'Figures/Precipitation_Trace Replace.png' )
   
    ''' Original Precipitation vs. Precipitation - Post 9999 Check '''
    #plot_checked_data( PrecipDataDF['PRCP'] , precip_data_check9999['PRCP'] , 'Precipitation (in.)', 'Precipitation - 9999 Check' , 'Figures/Precipitation_9999 Check.png' )
    
    """ ----------------------------- Data Analysis -------------------------------------- """
    # Group Precipitation Seasonal Data by Station and create plot
    seasonal_precip_df = seasonal_precip( DailyPrecipDataDF_2003_2023 ) # remove this one once modified_precip_data is complete
    '''seasonal_precip_df = seasonal_precip( modified_precip_data ) # unhash this one once modified_precip_data is complete'''
    #plot_seasonaltrends(seasonal_precip_df, 'Figures/Seasonal_Precipitation_Trends.png')
    
    # Plot mean monthly precipitation at each station
    #for station in precip_station_info_df.index:
    #    plot_mean_monthly_precip( PrecipDataDF , station, 'Figures/Average Monthly Precipitation_All Stations.png')
    
    # Plot histograms for daily precipitation and tide to get distributions
    plot_precip_hist( PrecipDataDF, 'Figures/Daily Precipitation Histogram_All Stations.png' )
    
    plot_tide_hist( TideDataDF, 'Figures/Hourly Tide Histogram.png' )
    
    # Plot when daily precipitation > 4 in at all stations
    plot_extreme_precip( PrecipDataDF  , 'Figures/Extreme Precipitation_All Stations.png' )
    
    # Plot maximum monthly tide
    plot_max_daily_tide( TideDataDF , 'Figures/Maximum Daily Tide Data.png')
    
    # Plot 3-day rolling sum of precipitation
    plot_3Dprecip_rolling_sum( PrecipDataDF , 'Figures/Rolling_Precip.png')
    
    # Plot a combined graph of when daily precipitation > 4 in, 3-day rolling sum of precipitation > 4 in, and maximum tide
    combined_tide_precip( TideDataDF , PrecipDataDF , 'Figures/Combined_Tide_Precip.png')


    "----------------------------------Plot the Map ---------------------------------------------------------------"
    plot_latitude_longitudeMap(precip_station_info_df)
    

"""-------------------------------------Yearly/Hourly Tide data Combined -------------------------------------------------------"""
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
    