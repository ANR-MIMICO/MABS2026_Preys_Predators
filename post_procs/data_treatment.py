import os
import pandas as pd
import re
import numpy as np

# Define the directory containing the results files
results_dir = './results'

# Initialize a list to store the aggregated data for each file
data = []

# Regex to separate parameter name (letters) from its value (numbers) in the filename
param_pattern = re.compile(r'([A-Za-z]+)([0-9]+)')
param_pattern = re.compile(r'([A-Za-z]+)(\d+\.?\d*)')
# Check if the directory exists
if not os.path.exists(results_dir):
    print(f"Directory '{results_dir}' not found. Please make sure it exists.")
else:
    print(f"Processing files in {results_dir}...")
    # Iterate through each file in the directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            # 1. Process Filename Parameters
            # Initialize a dictionary to store data for current file
            file_data = {'filename': filename}
            
            # Remove the extension and split the filename by underscores
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            
            # Iterate through the parts, skipping the first one ("Run")
            for part in parts[1:]:
                match = param_pattern.match(part)
                if match:
                    param_name, param_value = match.groups()
                    # Convert the value to an integer and add it to the dictionary
                    file_data[param_name] = (param_value)
            
            # 2. Process CSV Content (Final Row)
            filepath = os.path.join(results_dir, filename)
            try:
                # Read the CSV file into a temporary dataframe
                # Using header=0 implies the first row contains column names, like in image_4.png
                temp_df = pd.read_csv(filepath)
                
                # Check if the dataframe is not empty to avoid errors
                if not temp_df.empty:
                    # Select the last row using iloc[-1]
                    last_row_series = temp_df.iloc[-1]
                    
                    # Convert the Series to a dictionary and update the main file_data dictionary
                    # This automatically adds the CSV headers as keys and final values as values
                    file_data.update(last_row_series.to_dict())
                else:
                    print(f"Warning: {filename} was empty. Only filename parameters added.")

            except Exception as e:
                print(f"Error reading content of {filename}: {e}")
            
            # Add the combined data for this file to the main list
            data.append(file_data)

    # Create a pandas DataFrame from the list of dictionaries
    # Pandas will automatically handle varying keys, filling missing values with NaN if files have different headers
    df = pd.DataFrame(data)

# --- Drop specific columns ---
    columns_to_drop = ['IF', 'IG', 'IR', 'IH', 'IV']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # --- ADD REGIME COLUMN ---
    # Ensure the relevant columns are numeric just in case errors crept in
    df["Bandicoots"] = pd.to_numeric(df["Bandicoots"], errors='coerce').fillna(0)
    df["Foxes"] = pd.to_numeric(df["Foxes"], errors='coerce').fillna(0)

    # 1. Define the list of conditions
    # Note: Bitwise operators (&) are used for element-wise AND comparisons in pandas/numpy
    conditions = [
        (df["Bandicoots"] == 0),
        (df["Bandicoots"] > 0) & (df["Foxes"] == 0),
        (df["Bandicoots"] > 0) & (df["Foxes"] > 0)
    ]

    # 2. Define the corresponding values for each condition
    choices = [0, 1, 2]

    # 3. Use np.select to create the new column
    # default=-1 helps identify rows that didn't match any condition (though unlikely here)
    df["regime"] = np.select(conditions, choices, default=-1)

    print(f"Final DataFrame shape: {df.shape}")
    
    # Displaying the relevant columns to verify the new regime column
    print("\nPreview of Bandicoots, Foxes, and regime columns:")
    print(df[["Bandicoots", "Foxes", "regime"]].head(10))
    
    # --- SAVE THE DATAFRAME ---
    output_filename = 'final_aggregated_results_with_regime.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved data to '{output_filename}'")


print("DataFrame saved successfully to 'processed_results.csv'")






