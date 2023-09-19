import numpy as np
import pandas as pd

def delay_bracket(x):
    brackets = {0:0,1:1,2:2}
    return brackets[np.floor(x/30) if np.floor(x/10) > 2 else 2]

def debt_bracket(x):
    if x <=1000:
        return 0
    elif x <=2500:
        return 1
    else:
        return 2

def credit_history_bracket(x):   
    brackets = {0:2,1:1,2:0}
    return brackets[np.floor(x/10) if np.floor(x/10) <= 2 else 2]

def credit_card_bracket(x):
    if x <=4:
        return 0
    elif x <=7:
        return 1
    else:
        return 2
def num_loan_bracket(x):
    if x <=2:
        return 0
    elif x <=5:
        return 1
    else:
        return 2
    
def bracketed_data(df: pd.DataFrame):
    functions = {0:delay_bracket,1:debt_bracket,2:credit_history_bracket,3:credit_card_bracket,4:num_loan_bracket}
    for i in range(len(df.columns)-1):
        df[df.columns[i]] = df[df.columns[i]].apply(functions[i])
    return df
