import pandas as pd  
data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],  
        'Age': [28, 24, 35, 32],  
        'Country': ['USA', 'UK', 'Australia', 'Germany']}  

df = pd.DataFrame(data)  
print("Original DataFrame:")  
print(df)  

filtered_df = df[df['Age'] > 30]  
print("\nFiltered DataFrame:")  
print(filtered_df)
