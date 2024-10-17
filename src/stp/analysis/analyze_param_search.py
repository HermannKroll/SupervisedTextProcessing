import json
import os

import pandas as pd

from stp.config import BASELINE_MODELS


def load_json_files(folder_path, file_pattern):
    files_data = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_pattern in file_name:
                with open(os.path.join(root, file_name), 'r') as file:
                    data = json.load(file)
                    files_data.append(data)
    return files_data


def analyze_results(files_data):
    all_params = []
    all_scores = []

    for data in files_data:
        all_params.extend(data['all_params'])
        all_scores.extend(data['all_scores'])

    df = pd.DataFrame(all_params)
    df['score'] = all_scores

    grouped = df.groupby(list(df.columns[:-1])).mean().reset_index()

    best_params = grouped.sort_values(by='score', ascending=False)

    return best_params


def main(folder_path):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    svc_files = load_json_files(folder_path, 'svc_grid_search_results.json')
    rf_files = load_json_files(folder_path, 'rf_grid_search_results.json')
    xgb_files = load_json_files(folder_path, 'xgb_grid_search_results.json')

    if svc_files:
        print("Best parameters for SVC:")
        svc_best_params = analyze_results(svc_files)
        print(svc_best_params.head())

    if rf_files:
        print("\nBest parameters for RF:")
        rf_best_params = analyze_results(rf_files)
        print(rf_best_params.head())

    if xgb_files:
        print("\nBest parameters for XGB:")
        xgb_best_params = analyze_results(xgb_files)
        print(xgb_best_params.head())


if __name__ == "__main__":
    main(BASELINE_MODELS)
