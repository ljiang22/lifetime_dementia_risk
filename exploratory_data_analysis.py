"""
Exploratory data analysis:
1) build correlation heat map;
2) remove features with low variance;
3) remove the features that have a low chance to have an effect on the dependent variable.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression
import seaborn as sns


def main():
    set_option('display.width', 2000)
    pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
    set_option('precision', 3)
    pd.options.mode.chained_assignment = None

    input_file = './raw data_edit/data_fd.csv'
    data_input_ori = pd.read_csv(input_file)
    data_input_ori = data_input_ori.drop(columns=['Subject'])

    # Create correlation heatmap
    cols = data_input_ori.keys()
    cols_edit = cols[1:]
    corr = data_input_ori[cols_edit].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    thd = 0.35
    corr_np = corr.values
    corr_edit = np.zeros_like(corr)
    corr_pairs = []
    for i in range(corr_np.shape[0]):
        for j in range(corr_np.shape[1]):
            if j > i:
                if corr_np[i, j] >= thd or corr_np[i, j] <= -thd:
                    corr_edit[i, j] = corr_np[i, j]
                    tmp1 = cols_edit[i]
                    tmp2 = cols_edit[j]
                    tmp = [tmp1, tmp2, corr_np[i, j]]
                    corr_pairs.append(tmp)
    print('Feature pairs with a high correlation (|cc| >=0.35):', corr_pairs)

    plt.figure(1)
    sns.heatmap(corr, annot=False, vmin=-1, vmax=1, xticklabels=1, yticklabels=1, mask= mask, cmap='seismic')

    data_input_ori = data_input_ori.drop(columns=['INDEPEND', 'TOBAC100', 'TOBAC30'])
    keys = data_input_ori.keys()

    # Remove the features with low variance
    var = 0.10
    sel = VarianceThreshold(threshold=var)
    data_edit = sel.fit_transform(data_input_ori)
    indices = sel.get_support(indices=True)
    keys0 = []  # The features with low variance
    for i in range(len(keys)):
        if i not in indices:
            keys0.append(keys[i])
    print('The features removed due to low variance', keys0)

    keys1 = keys[indices]
    # Exclude the 'sumbox'
    keys1 = keys1[2:]

    features = data_edit[:, 2:]
    label = data_edit[:, 1]

    N1 = 24
    fv, pv = f_regression(features, label)

    indices3 = np.argsort(pv)
    print('keys3 without removing feature', keys1[indices3])
    indices3c = indices3[0:N1]
    print('Features removed due to the high p-value', keys1[indices3[N1:]])
    features_new = features[:, indices3c]
    x_axis = np.linspace(1, len(pv), len(pv))
    pv = sorted(pv)

    # Merge the data together
    label_norm = np.reshape(label, (len(label), 1))
    # Make sure the dimension is the same for the data sets
    data_edit = np.concatenate((features_new, label_norm), axis=1)
    np.save('./raw data_edit/data_ml', data_edit)

    font = {'size': 16}
    plt.rc('font', **font)
    plt.figure(2)
    plt.scatter(x_axis, pv, color='black')
    plt.xlabel("Feature")
    plt.ylabel("P-value")
    plt.show()


if __name__ == "__main__":
    main()























