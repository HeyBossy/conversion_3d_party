import pandas as pd
import catboost as cb
from features.transforms import FeatureTransformer
from catboost import CatBoostRegressor, EShapCalcType, EFeaturesSelectionAlgorithm

import optuna
import catboost as cb
import pickle
from sklearn.metrics import roc_auc_score
from model_optimization.catboost_opt import optimize_model

#######################
#
# This is a (very) simple solution to the classification problem of
# conversion prediction by means of third-party conversions.
#
#######################

VIEWS_PATH = './data/private_info/train_views.parquet'
ACTIONS_PATH = './data/private_info/train_actions.parquet'
THIRD_PARTY_PATH = './data/private_info/third_party_conversions.parquet'
MODEL_PATH = 'trained_model.cb'

feature_transformer = FeatureTransformer()

selected_features = ['accept_encoding',
                     'processed_mobile_screen_size',
                     'landing_page_domain',
                     'bid_referer_domain',
                     'creative_type',
                     'floor_cpm',
                     'is_interstitial',
                     'gdpr_regulation',
                     'user_fraud_state',
                     'processed_bid_isp_name']


def run():
    # Some datasets are quite big. All together they take ~ 15Gb RAM
    train_views = pd.read_parquet(VIEWS_PATH)
    train_actions = pd.read_parquet(ACTIONS_PATH)
    # third_party_conv = pd.read_parquet(THIRD_PARTY_PATH)

    # Construct dataset.
    # In this simple solution we just merge views and target post-click conversions.
    # Validation dataset is created in the same way.
    df = train_views.merge(train_actions[(train_actions.conversion_name == 'cart') \
                                         & (train_actions.is_post_click == 1)][['ssp_event_id', 'is_post_click']],
                           how='left', on='ssp_event_id')
    df.is_post_click.fillna(0, inplace=True)
    df.drop('ssp_event_id', inplace=True, axis=1)

    # rename target column
    df['label'] = df['is_post_click']
    df.drop('is_post_click', inplace=True, axis=1)
    split_date = '2024-01-17'

    X_train = df[(df.time < split_date)].drop(['label'], axis=1)
    y_train = df.label[(df.time < split_date)]
    X_test = df[(df.time > split_date)].drop(['label'], axis=1)
    y_test = df.label[(df.time > split_date)]

    feature_transformer = FeatureTransformer()
    feature_transformer.fit(X_train)
    X_train = feature_transformer.transform(X_train)
    X_test = feature_transformer.transform(X_test)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    cat_features = list(X_train.columns[X_train.dtypes == 'object'])
    num_features = list(X_train.columns[~(X_train.dtypes == 'object')])

    X_train = pd.concat((X_train[cat_features].fillna('-1'), X_train[num_features].fillna(-1)), axis=1)
    X_test = pd.concat((X_test[cat_features].fillna('-1'), X_test[num_features].fillna(-1)), axis=1)

    optimize_model(X_train, y_train, X_test, y_test, cat_features, MODEL_PATH)


    # cb_clf = cb.CatBoostClassifier(iterations=100, cat_features=cat_features, eval_metric="AUC",
    #                                early_stopping_rounds=20,
    #                                learning_rate=0.0183832802841483,
    #                                l2_leaf_reg=4.24348355935796,
    #                                subsample=0.46645667943978464,
    #                                depth=7,
    #                                border_count=179,
    #                                grow_policy='SymmetricTree',
    #                                random_seed=42
    #                                )
    # cb_clf.fit(X_train, y_train, eval_set=(X_test, y_test))

    # predictions = cb_clf.predict_proba(X_test)[:, 1]
    #
    # print("Save model..")
    # cb_clf.save_model(MODEL_PATH)
    return predictions


def make_pedictions(test_df_path):
    val_df = pd.read_parquet(test_df_path)
    val_df = feature_transformer.transform(val_df)
    val_df = val_df[selected_features]
    cat_features = list(val_df.columns[val_df.dtypes == 'object'])
    num_features = list(val_df.columns[~(val_df.dtypes == 'object')])
    val_df = pd.concat((val_df[cat_features].fillna('-1'), val_df[num_features].fillna(-1)), axis=1)

    cb_clf = cb.CatBoostClassifier()
    cb_clf.load_model(MODEL_PATH)

    final_predictions = cb_clf.predict_proba(val_df)[:, 1]
    print(f"Predictions length: {len(final_predictions)}")
    return final_predictions


if __name__ == "__main__":
    run()
