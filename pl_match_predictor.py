import pandas as pd


def get_cleaned_data():
    """
    Extract and clean Premier League match data from CSV file.
    
    Returns:
        pandas.DataFrame: Cleaned dataset with relevant match information
    """
    # Read the CSV file
    df = pd.read_csv('PL_24-25.csv')
    
    # Select only the columns we need to create the score strings
    columns_needed = [
        'HomeTeam',     # Home team name
        'AwayTeam',     # Away team name
        'Date',         # Match date
        'FTHG',         # Full time home team goals (for creating FT score)
        'FTAG',         # Full time away team goals (for creating FT score)
        'HTHG',         # Half time home team goals (for creating HT score)
        'HTAG',         # Half time away team goals (for creating HT score)
    ]
    
    # Extract only the needed columns
    temp_df = df[columns_needed].copy()
    
    # Create the final simplified dataframe with just the columns you want
    cleaned_df = pd.DataFrame()
    cleaned_df['HomeTeam'] = temp_df['HomeTeam']
    cleaned_df['AwayTeam'] = temp_df['AwayTeam']
    cleaned_df['Date'] = pd.to_datetime(temp_df['Date'], format='%d/%m/%Y')
    cleaned_df['FT'] = temp_df['FTHG'].astype(str) + '-' + temp_df['FTAG'].astype(str)
    cleaned_df['HT'] = temp_df['HTHG'].astype(str) + '-' + temp_df['HTAG'].astype(str)
    
    # Sort by date to have matches in chronological order
    cleaned_df = cleaned_df.sort_values('Date').reset_index(drop=True)

    cleaned_df.to_csv('PL_24-25_cleaned.csv', index=False)
    
    
    return cleaned_df


def display_data_info(df):
    """
    Display basic information about the dataset.
    
    Args:
        df (pandas.DataFrame): The cleaned dataset
    """
    print("Dataset Information:")
    print(f"Number of matches: {len(df)}")
    print(f"Date range: {df['Date'].min().strftime('%d/%m/%Y')} to {df['Date'].max().strftime('%d/%m/%Y')}")
    print(f"Number of unique teams: {len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))}")
    print("\nFirst few matches:")
    print(df.head())
    print("\nColumn data types:")
    print(df.dtypes)


def get_team_matches(df, team_name):
    """
    Get all matches for a specific team (both home and away).
    
    Args:
        df (pandas.DataFrame): The cleaned dataset
        team_name (str): Name of the team
    
    Returns:
        pandas.DataFrame: All matches involving the specified team
    """
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    return team_matches.sort_values('Date')


if __name__ == "__main__":
    # Example usage
    try:
        # Load and clean the data
        data = get_cleaned_data()
        
        # Display basic information
        display_data_info(data)
        
        # Example: Get matches for a specific team
        print("\n" + "="*50)
        print("Example: Liverpool matches")
        liverpool_matches = get_team_matches(data, 'Liverpool')
        print(liverpool_matches[['Date', 'HomeTeam', 'AwayTeam', 'FT', 'HT']].head())
        
    except FileNotFoundError:
        print("Error: PL_24-25.csv file not found. Please make sure the file is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
