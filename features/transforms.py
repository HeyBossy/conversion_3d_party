import pandas as pd
import json


class FeatureTransformer:
    def __init__(self):
        pass

    def fit(self, df):
        pass

    def transform(self, df):
        df = self._extract_time_features(df, 'time')
        df = self._replace_browser(df, 'ua_browser')
        df = self._create_data_by_zipcode(df, 'zip_code')
        return df

    def _extract_time_features(self, df, time_col):
        df[time_col] = pd.to_datetime(df[time_col])
        df['processed_hour_of_day'] = df[time_col].dt.hour
        df['processed_day_of_week'] = df[time_col].dt.weekday
        return df

    def _replace_browser(self, df, ua_browser_col):
        popular_browsers = ['CHROME', 'YANDEX', 'OTHER', 'SAFARI', 'OPERA', 'EDGE', 'FIREFOX', 'ANDROID_BROWSER']

        def __replace(browser):
            if browser not in (popular_browsers):
                return "Unknown"
            return browser

        df["processed_ua_browser"] = df[ua_browser_col].apply(__replace)
        return df

    def _replace_page_lang(self, df, lang):
        most_popular_page_langs = ['ru', 'es', 'en']

        def __replace(lang):
            if lang not in (most_popular_page_langs):
                return "Unknown"
            return lang

        df["processed_page_language"] = df[lang].apply(__replace)

    def _create_data_by_zipcode(self, df, zipcode):
        mapping_json_path = 'data/zipcode_to_data.json'
        with open(mapping_json_path, 'r', encoding='utf8') as f:
            mapping = json.load(f)

        temp_df = df['zip_code'].map(mapping)
        temp_df = pd.json_normalize(temp_df)
        columns = temp_df.columns
        for col in columns:
            temp_df = temp_df.rename({col: f"processed_{col}"}, axis='columns')
            df[f"processed_{col}"] = temp_df[f"processed_{col}"].values

        df = df.drop(zipcode, axis=1)

        return df
