__author__ = "Refilwe Kgoadi"
""""""
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.base import BaseEstimator, ClassifierMixin
from os.path import isfile
from sklearn.exceptions import NotFittedError
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
import pickle as pic

seed = 154


def tree_dataprep(df):
    """"""
    x = df.drop("Type", axis=1).astype("float32")
    y = df.Type
    return x, y


def feature_selection(x, y):
    # Train with random forests base classifier
    rf_model = XGBClassifier(tree_method="approx", objective="multi:softmax", booster="gbtree", n_estimators=100)
    rf_model.fit(x, y)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rfecv_model = RFECV(rf_model, step=1, min_features_to_select=10, cv=kfold, scoring='accuracy', n_jobs=-1)
    rfecv_model.fit(x, y)
    num_features = rfecv_model.n_features_
    print("--" * 20)
    print('Number of features selected: {}'.format(num_features))
    # Return selected with column names
    features = x.columns[rfecv_model.get_support(indices=True)].tolist()
    # save move using rf functions
    filename = "xgb_Features.pkl"
    pic.dump(features, open(filename, 'wb'))
    return features


def xgboptimise_params(x, y, valid_split=0.3):
    """"""
    feature_set = "xgb_Features.pkl"
    if isfile(feature_set):
        feature_subset = pd.read_pickle(feature_set)
    else:
        feature_subset = feature_selection(x, y)
    x = x.filter(feature_subset, axis=1)
    x_train, x_val, y_train, y_val = tts(x, y, test_size=valid_split, stratify=y, shuffle=True)

    def xgbobjective(params_set):
        xgbmodel = XGBClassifier(tree_method="approx", objective="multi:softmax", booster="gbtree",
                                 eval_metric="mlogloss", learning_rate=params_set["learning_rate"],
                                 n_estimators=int(params_set["n_estimators"]),
                                 min_child_weight=int(params_set["min_child_weight"]),
                                 colsample_bytree=int(params_set["colsample_bytree"]),
                                 scale_pos_weight=params_set["scale_pos_weight"],
                                 gamma=params_set["gamma"],
                                 reg_alpha=params_set["reg_alpha"],
                                 reg_lambda=params_set["reg_lambda"],
                                 colsample_bynode=params_set["colsample_bynode"],
                                 max_depth=int(params_set["max_depth"]),
                                 subsample=params_set["subsample"],
                                 )
        xgbmodel.fit(x_train, y_train, eval_metric='mlogloss', eval_set=[(x, y), (x_val, y_val)], verbose=0)
        xgbmodel.fit(x_train, y_train)
        y_predict = xgbmodel.predict_proba(x_val)
        auc = roc_auc_score(y_val, y_predict, multi_class="ovr", average="weighted")
        return {"loss": 1 - auc, "status": STATUS_OK, "model": xgbmodel}

    search_space = {"learning_rate": hp.quniform("learning_rate", 0.001, 0.32, 0.01),
                    "n_estimators": hp.quniform("n_estimators", 100, 2100, 100),
                    "colsample_bytree": hp.quniform("colsample_bytree", 0.65, 0.95, 0.05),
                    "colsample_bynode": hp.quniform("colsample_bynode", 0.65, 0.95, 0.05),
                    "min_child_weight": hp.quniform("min_child_weight", 1, 5, 1),
                    "gamma": hp.quniform("gamma", 0.65, 1.00, 0.05),
                    "reg_alpha": hp.quniform("reg_alpha", 0.0, 0.5, 0.02),
                    "reg_lambda": hp.quniform("reg_lambda", 0.0, 0.5, 0.02),
                    "max_depth": hp.quniform("max_depth", 2, 15, 2),
                    "scale_pos_weight": hp.quniform("scale_pos_weight", 0.5, 10, 0.5),
                    "subsample": hp.quniform("subsample", 0.65, 1.00, 0.05)
                    }
    trials = Trials()
    best_params = fmin(fn=xgbobjective, algo=tpe.suggest, max_evals=100, space=search_space, trials=trials)
    params = space_eval(search_space, best_params)
    file_nm = "xgb_bestparams.pkl"
    pic.dump(best_params, open(file_nm, "wb"))
    return params


class XGBOptimised(BaseEstimator, ClassifierMixin):

    def __init__(self, param_space="xgb_bestparams.pkl", clf_file="xgbmodel.pkl"):
        self.param_space = param_space
        self.clf_file = clf_file

    def fit(self, x, y, valid_split=.30):
        feature_set = "xgb_Features.pkl"
        if isfile(feature_set):
            feature_subset = pd.read_pickle(feature_set)
        else:
            feature_subset = feature_selection(x, y)
        x = x.filter(feature_subset, axis=1)
        # assert isinstance(x.values, object)
        x_train, x_val, y_train, y_val = tts(x, y, test_size=valid_split, stratify=y, shuffle=True)
        if not isfile(self.param_space):
            print("Optimised parameters not found ..\n Applying hyperopt.")
            optimal_params = xgboptimise_params(x, y)
        else:
            optimal_params = pd.read_pickle(self.param_space)
        # Check if classifier exists.
        if not isfile(self.clf_file):
            xgbmodel = XGBClassifier(tree_method="approx", objective="multi:softmax", booster="gbtree",
                                     eval_metric="mlogloss", learning_rate=optimal_params["learning_rate"],
                                     n_estimators=int(optimal_params["n_estimators"]),
                                     min_child_weight=int(optimal_params["min_child_weight"]),
                                     colsample_bytree=optimal_params["colsample_bytree"],
                                     gamma=optimal_params["gamma"],
                                     reg_alpha=optimal_params["reg_alpha"],
                                     reg_lambda=optimal_params["reg_lambda"],
                                     colsample_bynode=optimal_params["colsample_bynode"],
                                     max_depth=int(optimal_params["max_depth"]),
                                     scale_pos_weight=optimal_params["scale_pos_weight"])
            x, y = x_train, y_train
            x_val, y_val = x_val, y_val
            xgbmodel.fit(x, y, eval_metric='mlogloss', eval_set=[(x, y), (x_val, y_val)], verbose=0.)
            file_nm = "xgbmodel.pkl"
            pic.dump(xgbmodel, open(file_nm, "wb"))
        else:
            xgbmodel = pd.read_pickle(self.clf_file)
        return xgbmodel

    def predict(self, x_test):
        feature_set = "xgb_Features.pkl"
        feature_subset = pd.read_pickle(feature_set)
        x_test = x_test.filter(feature_subset, axis=1)
        if not isfile(self.clf_file):
            raise NotFittedError("Call 'fit' before predict")
        else:
            xgbmodel = pd.read_pickle(self.clf_file)
            y_predict = xgbmodel.predict(x_test)
        return y_predict

    def predict_proba(self, x_test):
        feature_set = "xgb_Features.pkl"
        feature_subset = pd.read_pickle(feature_set)
        x_test = x_test.filter(feature_subset, axis=1)
        if not isfile(self.clf_file):
            raise NotFittedError("Call 'fit' before predict")
        else:
            xgbmodel = pd.read_pickle(self.clf_file)
        y_probs = xgbmodel.predict_proba(x_test)
        return y_probs
