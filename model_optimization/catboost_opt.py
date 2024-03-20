import optuna
import catboost as cb
from sklearn.metrics import roc_auc_score
import json


class ModelOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, cat_features, save_path):
        self.best_auc = 0
        self.best_model = None
        self.save_path = save_path
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cat_features = cat_features

    def objective(self, trial):
        param = {
            'objective': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'depth': trial.suggest_int('depth', 4, 10),
            'border_count': trial.suggest_int('border_count', 50, 200),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'iterations': 100,
        }

        if param['grow_policy'] == 'Lossguide':
            param['max_leaves'] = trial.suggest_int('max_leaves', 31, 64)

        model = cb.CatBoostClassifier(**param, verbose=1)
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], cat_features=self.cat_features, early_stopping_rounds=20, verbose=1)

        preds = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, preds)

        if auc > self.best_auc:
            self.best_auc = auc
            self.best_model = model
            model.save_model(self.save_path)
            with open('best_model_params.json', 'w') as f:
                json.dump(param, f)

        return auc
        
    def save_results(self):
            # Сохраняем результаты в JSON
            with open('all_model_params.json', 'w') as f:
                json.dump(self.auc_results, f)

            # Конвертируем в DataFrame и сохраняем в CSV
            df = pd.DataFrame(self.auc_results)
            df_sorted = df.sort_values(by='auc', ascending=False)  # Сортировка по AUC
            df_sorted.to_csv('all_model_params.csv', index=False)

def optimize_model(X_train, y_train, X_test, y_test, cat_features, save_path):
    optimizer = ModelOptimizer(X_train, y_train, X_test, y_test, cat_features, save_path)
    study = optuna.create_study(direction='maximize')
    study.optimize(optimizer.objective, n_trials=100)
    
    # Сохраняем все результаты после оптимизации
    optimizer.save_results()

    print('Лучшие параметры: ', study.best_params)
