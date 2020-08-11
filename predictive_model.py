# Training the machine learning model and building the predictive model

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import sklearn
from sklearn.metrics import SCORERS
from mllib.utils import pca_analysis
from mllib.utils import data_split
from mllib.utils import evaluate
from mllib.utils import predictive_model
from mllib.utils import predictive_model_aug
from mllib.utils import data_augmentation


def main():

    raw_file_name = './raw data_edit/data_fd.csv'
    data_input_ori = pd.read_csv(raw_file_name)

    print(data_input_ori.keys())
    subject = pd.DataFrame(data_input_ori, columns=['Subject'])
    data_input = data_input_ori.drop(columns=['Unnamed: 0', 'Subject', 'INDEPEND', 'TOBAC100', 'TOBAC30'])

    data = pd.DataFrame(data_input, columns=['Age', 'BMI', 'Education', 'GDS', 'LIVSIT', 'INCONTU',
                                            'apoe', 'RESIDENC', 'INCONTF', 'SMOKYRS', 'B12DEF', 'DEP2YRS', 'dem_idx',
                                            'M/F', 'MARISTAT', 'PACKSPER', 'HYPERTEN', 'HYPERCHO', 'CVOTHR', 'DEPOTHR', 'CVANGIO', 'Hand',
                                            'PSYCDIS', 'sumbox'])
    keys_name = data.keys()

    training_method = 'RF'
    #training_method='xgboost'

    # PCA analysis
    """features = data[:, 0:-1]
    label = data[:, -1]
    thd = 0.95
    features_new = pca_analysis(features, thd=thd)
    features_new =features
    print(features_new.shape)"""

    #print(len(subject['Subject'].unique()))
    # Split the data
    rdst = 12
    X_train0, y_train0, X_train, X_val, X_test, y_train, y_val, y_test = data_split(data, subject, rdst=rdst)
    # Train the model without data augmentation
    predictive_model(training_method, X_train0, y_train0, X_train, X_val, X_test, y_train, y_val, y_test, keys_name)

    # Extrapolate the data to older ages using linear relationship between CDR and age
    # data_aug = data_augmentation(data_input_ori)
    input_file = './raw data_edit/data_fd_age.csv'  # The well name of an input file
    data_aug = pd.read_csv(input_file)
    subject_aug = pd.DataFrame(data_aug, columns=['Subject'])
    data_input = data_aug.drop(columns=['Unnamed: 0', 'Subject', 'INDEPEND', 'TOBAC100', 'TOBAC30'])

    data_aug_edit = pd.DataFrame(data_input, columns=['Age', 'BMI', 'Education', 'GDS', 'LIVSIT', 'INCONTU',
                                            'apoe', 'RESIDENC', 'INCONTF', 'SMOKYRS', 'B12DEF', 'DEP2YRS', 'dem_idx',
                                            'M/F', 'MARISTAT', 'PACKSPER', 'HYPERTEN', 'HYPERCHO', 'CVOTHR', 'DEPOTHR', 'CVANGIO', 'Hand',
                                            'PSYCDIS', 'sumbox'])
    # Split the data
    rdst = 42
    X_train0, y_train0, X_train, X_val, X_test, y_train, y_val, y_test = data_split(data_aug_edit, subject_aug, rdst=rdst)
    # Train the model with augmented data
    predictive_model_aug(training_method, X_train0, y_train0, X_train, X_val, X_test, y_train, y_val, y_test, keys_name)


if __name__ == '__main__':
    main()