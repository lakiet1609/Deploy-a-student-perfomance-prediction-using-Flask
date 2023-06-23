import pandas as pd

data = r'artifact\train.csv'

df = pd.read_csv(data)

# print(df.columns.values.tolist())
#['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score', 'writing score']

print(df['gender'].unique())
print('----------------------------------------------------------------')
print(df['race/ethnicity'].unique())
print('----------------------------------------------------------------')
print(df['parental level of education'].unique())
print('----------------------------------------------------------------')
print(df['lunch'].unique())
print('----------------------------------------------------------------')
print(df['test preparation course'].unique())


