import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost
import os
import matplotlib.pyplot as plt
import copy

# load data, create X, y train test
df = pd.read_csv('analysis_caa.csv')
X = df.drop(['id', 'model_id', 'model_variant', 'expert_coherence_avg', 'expert_consistency_avg', 'expert_fluency_avg',
             'expert_relevance_avg'], axis=1)
info = df[['id', 'model_id', 'model_variant']]


def run_analysis(X, y, info, output):
    # train test split
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=42)

    # create model
    model = xgboost.XGBRegressor(objective='reg:squarederror', max_depth=2, learning_rate=0.1,
                                 n_estimators=6000, n_jobs=15, base_score=np.mean(y_train), subsample=0.7,
                                 colsample_bytree=0.9, reg_lambda=50)

    # fit model
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_validation, y_validation)],
              eval_metric=['rmse', 'mae'], early_stopping_rounds=200, verbose=False);

    print('Validation RMSE: ', np.sqrt(mean_squared_error(y_validation, model.predict(X_validation))))
    print('Validation MAE: ', np.sqrt(mean_absolute_error(y_validation, model.predict(X_validation))))

    # Run shapley explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_interaction_values = explainer.shap_interaction_values(X)

    def plot_scatter(metric):
        shap.plots.scatter(shap_values[:, shap_values.feature_names[metric]], color=shap_values, show=False)

        if not os.path.exists('plots/' + output + '/scatter/'):
            os.makedirs('plots/' + output + '/scatter/')

        plt.savefig('plots/' + output + '/scatter/' + shap_values.feature_names[metric].replace('*', 'star') + '.png',
                    dpi=200, bbox_inches="tight")
        plt.close()

    def plot_waterfall(idx):

        shap.plots.waterfall(shap_values[idx], show=False)

        doc_id = info.iloc[idx][0].strip('dm-test-')
        model_id = info.iloc[idx][1]
        variant_id = info.iloc[idx][2].split('.')[0].split('/')[-1]

        if not os.path.exists('plots/' + output + '/waterfall/'):
            os.makedirs('plots/' + output + '/waterfall/')

        plt.savefig('plots/' + output + '/waterfall/' + doc_id + model_id + variant_id + '.png', dpi=200,
                    bbox_inches="tight")
        plt.close()

    # # dependence plots
    # for i in range(shap_values.shape[1]):
    #     plot_scatter(i)

    # waterfall plots
    for i in range(shap_values.shape[0]):
        plot_waterfall(i)

    # # summary plot
    # shap.plots.beeswarm(copy.deepcopy(shap_values), max_display=20, show=False)
    # plt.savefig('plots/' + output + '/summary.png', dpi=200, bbox_inches="tight")
    # plt.close()
    #
    # # feature importance abs(shap)
    # shap.plots.bar(shap_values, max_display=20, show=False)
    # plt.savefig('plots/' + output + '/bar.png', dpi=200, bbox_inches="tight")
    # plt.close()


run_analysis(X, df['expert_coherence_avg'], info, output='coherence')
run_analysis(X, df['expert_consistency_avg'], info, output='consistency')
run_analysis(X, df['expert_fluency_avg'], info, output='fluency')
run_analysis(X, df['expert_relevance_avg'], info, output='relevance')


