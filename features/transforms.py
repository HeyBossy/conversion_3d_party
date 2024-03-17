import pandas as pd
from joblib import load
import json
from features.variables import common_screen_size
from urllib.parse import urlparse
import pickle


class FeatureTransformer:
    def __init__(self):
        pass

    def fit(self, df):
        pass

    def transform(self, df):

        # that was bad idea xD
        df = self._extract_time_features(df, 'time')
        df = self._replace_browser(df, 'ua_browser')
        # df = self._create_data_by_zipcode(df, 'zip_code')

        df = self._replace_page_lang(df, 'page_language')
        df = self._categorize_creative_size(df, 'creative_size')
        df = self._categorize_screen_size(df, 'mobile_screen_size')
        df = self._categorize_viewability(df, 'historical_viewability')
        df = self.ua_browser_version_freq(df)
        df = self.tag_id_freq(df)
        df = self.bid_isp_name(df)
        df = self.select_domain_landing_page(df)
        df = self.select_domain_bid_url(df)
        df = self.select_domain_bid_referer(df)
        df = self.select_big_city(df)

        df = self._categorize_search_terms(df, 'search_terms')

        # select
        df = df[['ua_browser_version', 'tag_id', 'bid_isp_name',
                 'landing_page_domain', 'bid_url_domain', 'category_city',
                 'user_segments',
                 'processed_hour_of_day', 'processed_day_of_week', 'processed_period',
                 'processed_ua_browser',
                 'processed_page_language', 'processed_creative_size',
                 'processed_historical_viewability', 'processed_search_terms']]

        # that was bad idea too
        # df = self._create_user_seg(df)

        # df = self._create_3d_conv_features(df)
        return df

    def ua_browser_version_freq(self, df):
        freq = pd.read_pickle('features/ua_browser_version_freq.pkl')

        df['ua_browser_version'].fillna("unknown", inplace=True)
        df['ua_browser_version'] = df['ua_browser_version'].map(freq)
        df['ua_browser_version'].fillna("rare", inplace=True)
        df['ua_browser_version'] = df['ua_browser_version'].astype('str')

        return df

    def tag_id_freq(self, df):
        freq = pd.read_pickle('features/tag_id_freq.pkl')

        df['tag_id'].fillna("unknown", inplace=True)
        df['tag_id'] = df['tag_id'].map(freq)
        df['tag_id'].fillna("rare", inplace=True)

        df['tag_id'] = df['tag_id'].astype('str')
        return df

    def bid_isp_name(self, df):
        with open('features/bid_isp_name_freq_list.pkl', 'rb') as f:
            freq = pickle.load(f)

        def bid_isp_name_processing(bid_isp_name):
            if bid_isp_name is None:
                return "unknown"

            if bid_isp_name in freq:
                return bid_isp_name

            return "rare"

        df['bid_isp_name'] = df['bid_isp_name'].apply(bid_isp_name_processing).value_counts()

        return df

    def select_domain_landing_page(self, df):
        df['landing_page_domain'] = df['landing_page'].apply(
            lambda x: urlparse(x).netloc if pd.notnull(x) else 'unknown')

        df.drop(['landing_page'], axis=1, inplace=True)
        return df

    def select_domain_bid_url(self, df):
        df['bid_url_domain'] = df['bid_url'].apply(
            lambda x: urlparse(x).netloc if pd.notnull(x) else 'unknown')

        df.drop(['bid_url'], axis=1, inplace=True)
        return df

    def select_domain_bid_referer(self, df):
        df['bid_referer_domain'] = df['bid_referer'].apply(
            lambda x: urlparse(x).netloc if pd.notnull(x) else 'unknown')

        df.drop(['bid_referer'], axis=1, inplace=True)
        return df

    def select_big_city(self, df):
        million_cities = [
            "Moskva", "Sankt-Peterburg", "Novosibirsk", "Yekaterinburg",
            "Nizhniy Novgorod", "Kazan", "Chelyabinsk", "Omsk", "Samara",
            "Rostov-on-Don", "Ufa", "Krasnoyarsk", "Perm", "Voronezh", "Volgograd"
        ]

        city_counts = df['city'].value_counts()
        df['city_count'] = df['city'].map(city_counts)
        df['category_city'] = df.apply(
            lambda x: 'freq_million_city' if (x['city'] in million_cities and x['city_count'] >= 100) else 'other',
            axis=1
        )

        return df

    def _create_3d_conv_features(self, df):
        user_freq = pd.read_pickle('features/user_freq.pkl')
        df['user_3d_freq'] = df['user_id'].map(user_freq)
        df['user_3d_freq'] = df['user_3d_freq'].fillna(0)

        user_conv_types_and_count = pd.read_pickle('features/user_conversion_types_and_count.pkl')
        df = df.set_index('user_id').join(user_conv_types_and_count, how='left')
        df = df.reset_index()

        return df

    def _create_user_seg(self, df):
        users_segs = pd.read_parquet('features/users_segs.parquet')
        df = df.merge(users_segs, how='left', on='user_id')
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
        mapping_json_path = 'features/zipcode_to_data.json'
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

        def __to_numerical(size_str):
            if (size_str is None):
                return -1, -1
            width, height = map(int, size_str.split('x'))
            return [width, height]

        df['processed_' + mobile_screen_size] = df[mobile_screen_size].apply(lambda x: __replace(x))
        nums = df[mobile_screen_size].apply(lambda x: __to_numerical(x))
        df['processed_' + mobile_screen_size + 'w'] = nums.apply(lambda x: x[0])
        df['processed_' + mobile_screen_size + 'h'] = nums.apply(lambda x: x[1])
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

    def _categorize_search_terms(self, df, search_terms):
        model_path = 'features/svc_model.joblib'
        loaded_model = load(model_path)

        na_values = df[df[search_terms].isna()].index
        df[search_terms] = df[search_terms].fillna('')
        predictions_terms = loaded_model.predict(df[search_terms])
        df['processed_' + search_terms] = predictions_terms
        df.loc[na_values, 'processed_' + search_terms] = 'unknown'
        df = df.drop(search_terms, axis=1)
        return df
