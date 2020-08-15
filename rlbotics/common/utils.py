import pandas as pd


def get_return(log_file):
    logs = pd.read_csv(log_file)

    ep_sum = 0
    ep_returns = []

    for i in range(len(logs)):
        if logs.loc[i,'done'] == False:
            ep_sum += logs.loc[i, 'rewards']
            
        elif logs.loc[i,'done'] == True:
            ep_sum += logs.loc[i, 'rewards']
            ep_returns.append(ep_sum)
            ep_sum = 0

    return ep_returns
