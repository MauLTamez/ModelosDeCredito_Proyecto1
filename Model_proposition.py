import numpy as np
import pandas as pd
from cleanData import clean
from prediction import bracketed_data
import matplotlib.pyplot as plt

data = pd.read_csv('ModelosDeCredito_Proyecto1/train.csv')
data_clean = clean(data)
#print(data_clean)

X = data_clean[[
    'Monthly_Inhand_Salary','Num_Credit_Card','Num_of_Loan','Delay_from_due_date',
    'Num_of_Delayed_Payment','Num_Credit_Inquiries','Credit_Utilization_Ratio',
    'Credit_History_Age','Monthly_Balance'
    ]].columns
y = data_clean.columns[-1]

selected_data = data_clean[['Delay_from_due_date','Outstanding_Debt','Credit_History_Age','Num_Credit_Card','Num_of_Loan','Score']]

bracketed = bracketed_data(selected_data)

score = 1000 - ( 100*np.ones(5) * bracketed[bracketed.columns[:-1]].values.reshape([-1,1]) ).sum(axis = 1)

pd.Series(score).hist()
plt.show()