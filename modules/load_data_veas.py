import pandas as pd
from modules.data_processing import rename_duplicate_columns

def load_data(data_path, column_names_path, sheets=[0, 1]):
    """
    Loads a CSV dataset, processes timestamps, renames columns using descriptions from an Excel file, 
    and ensures unique column names.

    Args:
        csv_path (str): Path to the CSV file.
        excel_path (str): Path to the Excel file with column descriptions.
        rename_sheets (list): List of sheet indices to extract name mappings.

    Returns:
        pd.DataFrame: Processed DataFrame with renamed and cleaned columns.
    """
    # Load CSV file
    data = pd.read_csv(data_path, index_col=0)
    
    # Convert "Time" column to datetime
    data["Time"] = pd.to_datetime(data["Time"])

    # Rename columns using multiple sheets from the Excel file
    for sheet in sheets:
        names_df = pd.read_excel(column_names_path, sheet_name=sheet, skiprows=2).iloc[:, :3].dropna()
        names_dict = dict(zip(names_df["Navn"], names_df["Beskrivelse"]))
        data.rename(columns=names_dict, inplace=True)

    # Ensure column names are unique
    data = rename_duplicate_columns(data)  # Assuming this function exists

    return data

# Example usage:
# data = load_data('data/veas_Hall_1_no-anomaly_010122-231023.csv', 
#                  'data/Tagliste_denit_prosesshaller.xlsx')
