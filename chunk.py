import pandas as pd
import matplotlib.pyplot as plt

# Define the file path
file_path = r"C:\Users\acer\Desktop\maternal mortality\NPGR82FL_output.csv"

# Use the actual column names from the CSV
columns = [
    "Year of interview",
    "Province",
    "Type of place of residence",
    "Total children ever born",
    "Sons who have died",
    "Daughters who have died",
    "Births in last five years",
    "Births in past year",
    "Ever had a terminated pregnancy",
    "Pregnancy losses"
]

# Read the data in chunks to handle large files
chunks = pd.read_csv(file_path, usecols=columns, chunksize=100000)

# Concatenate all chunks
df = pd.concat(chunks)

# Calculate total child deaths (as a proxy for maternal risk context)
df["child_deaths"] = df["Sons who have died"].fillna(0) + df["Daughters who have died"].fillna(0)

# Group by year and region, sum relevant columns
trend = df.groupby(["Year of interview", "Province"]).agg({
    "Total children ever born": "sum",
    "Births in last five years": "sum",
    "Births in past year": "sum",
    "child_deaths": "sum",
    "Pregnancy losses": "sum"
}).reset_index()

# Plot trends (example: child deaths and pregnancy losses over years)
plt.figure(figsize=(10, 6))
for region in trend["Province"].unique():
    region_data = trend[trend["Province"] == region]
    plt.plot(region_data["Year of interview"], region_data["child_deaths"], label=f"Child Deaths - Region {region}")
    plt.plot(region_data["Year of interview"], region_data["Pregnancy losses"], label=f"Pregnancy Losses - Region {region}", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Trends in Child Deaths and Pregnancy Losses (Proxy for Maternal Mortality Risk)")
plt.legend()
plt.show()