import pandas as pd

# Load both files
df1 = pd.read_csv(r'D:\Python_files\final_merged_for_model.csv')
df2 = pd.read_csv(r'D:\Python_files\extracted_articles.csv')

# Merge them — assuming you want to combine all rows
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the result
merged_df.to_csv(r'D:\Python_files\fully_merged.csv', index=False)

print("✅ Merge complete! Saved to 'fully_merged.csv'")
