import pandas as pd
import numpy as np

class BikeDemandDataProcessor:
    def __init__(self, path):
        self.path = path

    def load_data(self, filename):
        full_path = f"{self.path}/{filename}"
        print(f"Loading data from {full_path}")
        try:
            return pd.read_csv(full_path)
        except FileNotFoundError:
            print(f"File {filename} not found in path {self.path}.")
            return None

    def preprocess(self, df):
        df = df.copy()

        # Creating timestamp column
        df['timestamp'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str) + ':00:00', format='%d/%m/%Y %H:%M:%S')

        # Renaming columns
        df = df.rename(columns={'hr': 'hour', 'yr': 'year', 'mnth': 'month', 'cnt': 'count'})
        df = df.drop(['dteday', 'instant'], axis=1)
        
        # Extracting time features
        df['year'] = df['timestamp'].dt.year
        df['day'] = df['timestamp'].dt.day_of_year
        df['day_of_week'] = df['timestamp'].dt.day_of_week
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        # Cyclic encoding for time features
        df['year_sin'] = np.sin(2 * np.pi * df['year'])
        df['year_cos'] = np.cos(2 * np.pi * df['year'])
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        # Day type features
        df['working_day'] = df['day_of_week'].apply(lambda x: 1 if x < 5 else 0)
        df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Special periods
        df['moonphase'] = df['timestamp'].apply(lambda x: (x.day + x.month * 29.53) % 29.53)
        df['quarter'] = df['timestamp'].dt.quarter
        df['christmas_holiday_season'] = df['timestamp'].apply(lambda x: 1 if (x.month == 1 and x.day <= 14) or (x.month == 12 and x.day >= 24) else 0)
        df['summer_season'] = df['timestamp'].apply(lambda x: 1 if 6 <= x.month <= 8 else 0)
        df['spring_season'] = df['timestamp'].apply(lambda x: 1 if 3 <= x.month <= 5 else 0)
        df['fall_season'] = df['timestamp'].apply(lambda x: 1 if 9 <= x.month <= 11 else 0)
        df['winter_season'] = df['timestamp'].apply(lambda x: 1 if x.month in [12, 1, 2] else 0)
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Hour features (rush hour estimated from visualisations)
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['working_day'] == 1)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['rush_hour'] = df.apply(lambda x: 1 if ((x['hour'] >= 4 and x['hour'] <= 10) or (x['hour'] >= 15 and x['hour'] <= 21)) and x['working_day'] == 1 else 0, axis=1)

        # Creating interaction features
        df['hum_windspeed'] = df['hum'] * df['windspeed']
        df['temp_hum'] = df['temp'] * df['hum']
        df['temp_windspeed'] = df['temp'] * df['windspeed']
        df['windspeed_squared'] = df['windspeed'] ** 2
        df['hum_squared'] = df['hum'] ** 2
        
        # Weather interaction features
        df['weather_temp'] = df['weathersit'] * df['temp']
        df['weather_hum'] = df['weathersit'] * df['hum']
        df['weather_windspeed'] = df['weathersit'] * df['windspeed']
        
        # Atemp interaction features
        df['atemp_hum'] = df['atemp'] * df['hum']
        df['atemp_windspeed'] = df['atemp'] * df['windspeed']
        df['atemp_squared'] = df['atemp'] ** 2
        df['temp_atemp'] = df['temp'] * df['atemp']
        df['weather_atemp'] = df['weathersit'] * df['atemp']
        
        # Temperature difference feature
        df['temp_atemp_diff'] = np.abs(df['temp'] - df['atemp'])

        # Creating 28-day lag of the count
        df['count_lag_28d'] = df['count'].shift(28 * 24)

        # Creating 1-year lag of the count
        df['count_lag_1y'] = df['count'].shift(365 * 24)

        df['count_lag_28d'] = df['count_lag_28d'].ffill().bfill()
        df['count_lag_1y'] = df['count_lag_1y'].ffill().bfill()

        # Convert object columns to category
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

        return df.drop_duplicates()
    
    def feature_engineering(self, train_df, val_df):
        train_df = train_df.copy()
        val_df = val_df.copy()

        # Avoid division by zero
        casual_sum = train_df['casual'].sum()
        if casual_sum == 0:
            casual_sum = 1  # Prevent division by zero

        total_ratio_of_registered_uses = train_df['registered'].sum() / casual_sum

        # Average ratios for different time-based groupings
        average_hour_ratio = train_df.groupby('hour').agg(
            ratio=('registered', lambda x: x.sum() / max(x.sum(), 1))
        )['ratio']

        average_day_ratio = train_df.groupby('day_of_week').agg(
            ratio=('registered', lambda x: x.sum() / max(x.sum(), 1))
        )['ratio']

        average_week_ratio = train_df.groupby('week').agg(
            ratio=('registered', lambda x: x.sum() / max(x.sum(), 1))
        )['ratio']

        average_month_ratio = train_df.groupby('month').agg(
            ratio=('registered', lambda x: x.sum() / max(x.sum(), 1))
        )['ratio']

        average_season_ratio = train_df.groupby('season').agg(
            ratio=('registered', lambda x: x.sum() / max(x.sum(), 1))
        )['ratio']

        average_weekend_ratio = train_df.groupby('weekend').agg(
            ratio=('registered', lambda x: x.sum() / max(x.sum(), 1))
        )['ratio']

        average_working_day_ratio = train_df.groupby('working_day').agg(
            ratio=('registered', lambda x: x.sum() / max(x.sum(), 1))
        )['ratio']

        # Applying ratios to both train and validation sets
        train_df['total_registered_ratio'] = total_ratio_of_registered_uses
        val_df['total_registered_ratio'] = total_ratio_of_registered_uses

        train_df['hour_ratio'] = train_df['hour'].map(average_hour_ratio)
        val_df['hour_ratio'] = val_df['hour'].map(average_hour_ratio)

        train_df['day_ratio'] = train_df['day_of_week'].map(average_day_ratio)
        val_df['day_ratio'] = val_df['day_of_week'].map(average_day_ratio)

        # Mapping working day and weekend registered ratios
        train_df['working_day_or_weekend_ratio'] = train_df['working_day'].map(average_working_day_ratio).where(train_df['working_day'] == 1, 
                                                                                                                train_df['weekend'].map(average_weekend_ratio))
        val_df['working_day_or_weekend_ratio'] = val_df['working_day'].map(average_working_day_ratio).where(val_df['working_day'] == 1, 
                                                                                                            val_df['weekend'].map(average_weekend_ratio))

        train_df['week_ratio'] = train_df['week'].map(average_week_ratio)
        val_df['week_ratio'] = val_df['week'].map(average_week_ratio)

        train_df['month_ratio'] = train_df['month'].map(average_month_ratio)
        val_df['month_ratio'] = val_df['month'].map(average_month_ratio)

        train_df['season_ratio'] = train_df['season'].map(average_season_ratio)
        val_df['season_ratio'] = val_df['season'].map(average_season_ratio)

        # Dropping columns that won't be available for prediction
        train_df = train_df.drop(['casual', 'registered'], axis=1)
        val_df = val_df.drop(['casual', 'registered'], axis=1)

        # Aggregate counts to daily level
        daily_train_df = train_df.groupby(['year', 'month', 'day', 'day_of_year'])[['count']].sum().reset_index()

        # Calculate rolling mean and standard deviation (2-week window)
        rolling_mean = daily_train_df['count'].rolling(window=14, center=True).mean()
        rolling_std = daily_train_df['count'].rolling(window=14, center=True).std()

        # Identify 3-sigma outliers
        daily_train_df['sigma_3_outlier'] = (daily_train_df['count'] > rolling_mean + 3 * rolling_std) | \
                                            (daily_train_df['count'] < rolling_mean - 3 * rolling_std)

        # Find max outlier flag per day_of_year
        day_of_year_outlier = daily_train_df.groupby('day_of_year', as_index=False)['sigma_3_outlier'].max()

        # Merge back into train_df and val_df
        train_df = train_df.merge(day_of_year_outlier, on='day_of_year', how='left')
        val_df = val_df.merge(day_of_year_outlier, on='day_of_year', how='left')

        # Fill NaN values (if no outlier was detected for that day)
        train_df['sigma_3_outlier'] = train_df['sigma_3_outlier'].fillna(0)
        val_df['sigma_3_outlier'] = val_df['sigma_3_outlier'].fillna(0)

        return train_df.drop_duplicates(), val_df.drop_duplicates()
            
    def split_and_engineer_data(self, df, test_period=None):
        if test_period is None:
            test_period = 28 * 24  # Default to 28 days of hourly data
        sorted_df = df.sort_values('timestamp').copy()

        original_shape = sorted_df.drop(['casual', 'registered'], axis=1).shape  # Store original shape

        # Creating a Test DF
        train_df = sorted_df.iloc[:-test_period].copy()
        val_df = sorted_df.iloc[-test_period:].copy()

        train_df, val_df = self.feature_engineering(train_df, val_df)

        # Ensure row count is unchanged when reducing back to original columns
        original_columns = sorted_df.drop(['casual', 'registered'], axis=1).columns.tolist()
        
        reduced_train = train_df[original_columns]
        reduced_val = val_df[original_columns]

        combined_df = pd.concat([reduced_train, reduced_val])

        assert combined_df.shape == original_shape, "Row count mismatch after feature engineering!"

        return train_df, val_df
    
def inverse_boxcox(y_transformed, lambda_value):
    if lambda_value == 0:
        return np.exp(y_transformed) - 1
    else:
        return (y_transformed * lambda_value + 1) ** (1 / lambda_value) - 1
