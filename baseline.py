import pandas as pd
import catboost as cb
from features.transforms import FeatureTransformer

import optuna
import catboost as cb
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
    print(X_train.dtypes)
    y_train = df.label[(df.time < split_date)]
    X_test = df[(df.time >= split_date)].drop(['label'], axis=1)
    y_test = df.label[(df.time >= split_date)]

    feature_transformer.fit(X_train)
    X_train = feature_transformer.transform(X_train)
    X_test = feature_transformer.transform(X_test)
    cat_features = list(X_train.columns[X_train.dtypes == 'object'])
    print('cat_features', cat_features)
    num_features = list(X_train.columns[~(X_train.dtypes == 'object')])
    print('num_features', num_features)

    X_train = pd.concat((X_train[cat_features].fillna('-1'), X_train[num_features].fillna(-1)), axis=1)
    X_test = pd.concat((X_test[cat_features].fillna('-1'), X_test[num_features].fillna(-1)), axis=1)

    #    optimize_model(X_train, y_train, X_test, y_test, cat_features, MODEL_PATH)

    #    cb_clf = cb.CatBoostClassifier(cat_features=cat_features, eval_metric="AUC",
    #    early_stopping_rounds=20)

    cb_clf = cb.CatBoostClassifier(
        cat_features=cat_features,
        eval_metric="AUC",
        early_stopping_rounds=20,
        depth=6,  # Обновленный параметр
        l2_leaf_reg=7.482059679856143,  # Обновленный параметр
        border_count=53,
        grow_policy='Depthwise',
        learning_rate=0.2080395733274946,
        subsample=0.49804575140321844,
    )

    cb_clf.fit(X_train, y_train, eval_set=(X_test, y_test))

    feature_importances = cb_clf.get_feature_importance(prettified=True)



    print(f'Важность признаков  {feature_importances}')
    predictions = cb_clf.predict_proba(X_test)[:, 1]

    print("Save model..")
    cb_clf.save_model(MODEL_PATH)
    return predictions


def make_pedictions(test_df_path):
    val_df = pd.read_parquet(test_df_path)
    val_df = feature_transformer.transform(val_df)
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
