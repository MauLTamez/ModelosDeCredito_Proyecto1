import numpy as np
import pandas as pd
import pyswarms as ps
from cleanData import clean
import matplotlib.pyplot as plt
from prediction import bracketed_data
from pyswarms.utils.plotters import plot_cost_history

data = pd.read_csv('train.csv')
data_clean = clean(data)
selected_data = data_clean[['Delay_from_due_date','Outstanding_Debt','Credit_History_Age','Num_Credit_Card','Num_of_Loan','Score']]
bracketed = bracketed_data(selected_data).iloc[:10000]

# Important Functions

def credit_score_function(weights):
    #W = np.array([weights for i in range(len(bracketed))])
    X = bracketed[bracketed.columns[:-1]].values
    score = 900 - (weights * X).sum(axis = 1)
    score = pd.Series(score, name = 'Predicted').apply(lambda x: {0:1,1:2,2:3}[np.floor(x/300) if np.floor(x/300) <= 2 else 2] ).to_frame()
    score['Real'] = bracketed.reset_index()['Score']
    score['Error'] = (score['Real'] - score['Predicted']).apply(lambda x: x**2)
    error = score['Error'].sum()
    return error

def accuracy(weights):
    #W = np.array([weights for i in range(len(bracketed))])
    X = bracketed[bracketed.columns[:-1]].values
    score = 900 - (weights * X).sum(axis = 1)
    score = pd.Series(score, name = 'Predicted').apply(lambda x: {0:1,1:2,2:3}[np.floor(x/300) if np.floor(x/300) <= 2 else 2] ).to_frame()
    score['Real'] = bracketed.reset_index()['Score']
    score['Count'] = (score['Real'] - score['Predicted']).apply(lambda x: 0 if x!=0 else 1)
    return f"{round(score['Count'].sum()/len(score)*100,2)}%"


# PSO Parameters
swarm_size = len(bracketed)
dim = 5
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
constraints = (0, 200)
weights = 50 * np.ones((swarm_size, dim))  # Ensure the correct shape

# Call an instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                    dimensions=dim,
                                    options=options,
                                    bounds=constraints,
                                    init_pos=weights)

# Perform optimization
cost, joint_vars = optimizer.optimize(credit_score_function, iters=1000)
cost_history = optimizer.cost_history

plot_cost_history(cost_history)
plt.show()

print("Accuracy:",accuracy(joint_vars))
