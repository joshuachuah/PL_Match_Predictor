import pandas as pd
import os
import glob


def get_cleaned_data(input_file_path, output_file_path):
    """
    Extract and clean Premier League match data from CSV file.
    
    Args:
        input_file_path (str): Path to the input CSV file
        output_file_path (str): Path where the cleaned CSV will be saved
    
    Returns:
        pandas.DataFrame: Cleaned dataset with relevant match information
    """
    # Read the CSV file
    df = pd.read_csv(input_file_path)
    
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
    cleaned_df['Date'] = pd.to_datetime(temp_df['Date'], format='%d/%m/%Y')
    cleaned_df['HomeTeam'] = temp_df['HomeTeam']
    cleaned_df['AwayTeam'] = temp_df['AwayTeam']
    cleaned_df['HT'] = temp_df['HTHG'].astype(str) + '-' + temp_df['HTAG'].astype(str)
    cleaned_df['FT'] = temp_df['FTHG'].astype(str) + '-' + temp_df['FTAG'].astype(str)
    
    # Sort by date to have matches in chronological order
    cleaned_df = cleaned_df.sort_values('Date').reset_index(drop=True)

    cleaned_df.to_csv(output_file_path, index=False)
    
    
    return cleaned_df


def get_csv_files(csv_folder='csv'):
    """
    Get all CSV files from the specified folder.
    
    Args:
        csv_folder (str): Path to the folder containing CSV files
    
    Returns:
        list: List of CSV file paths
    """
    csv_pattern = os.path.join(csv_folder, '*.csv')
    csv_files = glob.glob(csv_pattern)
    return sorted(csv_files)


def process_all_csv_files(csv_folder='csv', output_folder='csv_cleaned'):
    """
    Process all CSV files in the csv folder and save cleaned versions.
    
    Args:
        csv_folder (str): Path to the folder containing input CSV files
        output_folder (str): Path to the folder where cleaned CSV files will be saved
    
    Returns:
        dict: Dictionary with file names as keys and cleaned DataFrames as values
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files
    csv_files = get_csv_files(csv_folder)
    
    if not csv_files:
        print(f"No CSV files found in {csv_folder} folder.")
        return {}
    
    processed_data = {}
    
    for csv_file in csv_files:
        try:
            # Get the base filename without extension
            base_filename = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Create output file path with _cleaned suffix
            output_filename = f"{base_filename}_cleaned.csv"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"Processing {csv_file}...")
            
            # Process the file
            cleaned_data = get_cleaned_data(csv_file, output_path)
            processed_data[base_filename] = cleaned_data
            
            print(f"Saved cleaned data to {output_path}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    return processed_data


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
    # Process all CSV files in the csv folder
    try:
        print("Processing all CSV files in the csv folder...")
        print("="*60)
        
        # Process all CSV files and get cleaned data
        all_processed_data = process_all_csv_files()
        
        if all_processed_data:
            print(f"\nSuccessfully processed {len(all_processed_data)} files:")
            for filename in all_processed_data.keys():
                print(f"  - {filename}")
            
            # Example: Display info for the most recent season (if available)
            if 'PL_24-25' in all_processed_data:
                print("\n" + "="*60)
                print("Example: Latest season (2024-25) data info")
                display_data_info(all_processed_data['PL_24-25'])
                
                # Example: Get matches for a specific team
                print("\n" + "="*50)
                print("Example: Liverpool matches from 2024-25 season")
                liverpool_matches = get_team_matches(all_processed_data['PL_24-25'], 'Liverpool')
                print(liverpool_matches[['Date', 'HomeTeam', 'AwayTeam', 'FT', 'HT']].head())
            else:
                # If 2024-25 data not available, use the first available dataset
                first_dataset_name = list(all_processed_data.keys())[0]
                first_dataset = all_processed_data[first_dataset_name]
                print(f"\n" + "="*60)
                print(f"Example: {first_dataset_name} data info")
                display_data_info(first_dataset)
        else:
            print("No CSV files were processed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
