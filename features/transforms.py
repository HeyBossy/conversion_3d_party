import pandas as pd
import json
from features.variables import common_screen_size


class FeatureTransformer:
    def __init__(self):
        pass

    def fit(self, df):
        pass

    def transform(self, df):
        df = self._extract_time_features(df, 'time')
        df = self._replace_browser(df, 'ua_browser')
        df = self._create_data_by_zipcode(df, 'zip_code')
        df = self._replace_page_lang(df, 'page_language')
        df = self._categorize_creative_size(df, 'creative_size')
        df = self._categorize_screen_size(df, 'mobile_screen_size')
        df = self._categorize_viewability(df, 'historical_viewability')

        return df

    def _extract_time_features(self, df, time_col):
        df[time_col] = pd.to_datetime(df[time_col])
        df['processed_hour_of_day'] = df[time_col].dt.hour
        df['processed_day_of_week'] = df[time_col].dt.weekday

        df['processed_period'] = (df[time_col].dt.hour % 24 + 4) // 4
        df['processed_period'].replace({1: 'Late Night',
                              2: 'Early Morning',
                              3: 'Morning',
                              4: 'Noon',
                              5: 'Evening',
                              6: 'Night'}, inplace=True)
        df = df.drop([time_col], axis=1)
        return df

    def _replace_browser(self, df, ua_browser_col):
        popular_browsers = ['CHROME', 'YANDEX', 'OTHER', 'SAFARI', 'OPERA', 'EDGE', 'FIREFOX', 'ANDROID_BROWSER']

        def __replace(browser):
            if browser not in (popular_browsers):
                return "Unknown"
            return browser

        df["processed_ua_browser"] = df[ua_browser_col].apply(__replace)
        df = df.drop([ua_browser_col], axis=1)

        return df

    def _replace_page_lang(self, df, lang):
        most_popular_page_langs = ['ru', 'es', 'en']

        def __replace(lang):
            if lang not in (most_popular_page_langs):
                return "Unknown"
            return lang

        df["processed_page_language"] = df[lang].apply(__replace)
        df = df.drop([lang], axis=1)

        return df

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

    def _categorize_creative_size(self, df, size):

        def __replace(size):
            '''Обработка рекламных баннеров'''
            if size in ['300x250', '240x400', '336x280', '320x50', '320x100', '300x600', '300x300', '250x250',
                        '300x50']:
                return 'Standard'
            elif size in ['728x90', '970x250', '970x90']:
                return 'Banner'
            elif size in ['480x320', '320x480']:
                return 'Mobile'
            elif size == '160x600':
                return 'Skyscraper'
            elif size == '580x400':
                return 'Large'
            else:
                return 'Other'

        df['processed_' + size] = df[size].apply(__replace)
        df = df.drop(size, axis=1)

        return df

    def _categorize_screen_size(self, df, mobile_screen_size):

        def __replace(size_str):
            '''Функция для категоризации размера экрана'''
            if (size_str is None) or (size_str not in common_screen_size):
                return 'Others'
            width, height = map(int, size_str.split('x'))
            # Рассчитываем диагональ в пикселях
            diagonal = (width ** 2 + height ** 2) ** 0.5

            # Определение категорий в соответствии с диагональю
            if diagonal < 1500:
                return 'Small'
            elif 1500 <= diagonal < 2000:
                return 'Medium'
            elif 2000 <= diagonal < 2500:
                return 'Large'
            else:
                return 'Extra Large'

        df['processed_' + mobile_screen_size] = df[mobile_screen_size].apply(lambda x: __replace(x))
        df = df.drop(mobile_screen_size, axis=1)

        return df

    def _categorize_viewability(self, df, viewability):

        def __replace(viewability):
            ''' Функция для категоризации видимость рекламного места на сайте'''
            if viewability is None:
                return 'Unknown'
            elif viewability < 50.0:
                return 'Low'
            elif 50.0 <= viewability < 75.0:
                return 'Medium'
            else:
                return 'High'

        df['processed_' + viewability] = df[viewability].apply(__replace)
        df = df.drop(viewability, axis=1)
        return df
