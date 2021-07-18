import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

def friedman_test():
    df=pd.read_csv("results.csv")
    datasets = df['Datasets'].unique()
    algorithms = df['Algorithms'].unique()
    mean_values = []
    alpha = 0.05
    for dataset in datasets:
        mean_tmp=[]
        for algorithm in algorithms:
            df_dataset_algorithm=df.loc[(df['Datasets'] == dataset) & (df['Algorithms'] == algorithm)]
            mean=df_dataset_algorithm['AUC'].mean()
            mean_tmp.append(mean)
        mean_values.append(mean_tmp)
    statistics, p_value = friedmanchisquare(*mean_values)
    print('Statistics values gained in Friedman Test are :\n'
          'Statistics=%.10f, P_Value=%.10f' % (statistics, p_value))

    if p_value > alpha:
        print('All algorithms have the same distribution, that means that we did not reject H0')
    else:
        print('Algorithms have different distribution, that means that we reject H0')
        print(sp.posthoc_nemenyi_friedman(mean_values))


if __name__ == '__main__':
    friedman_test()