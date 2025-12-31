import numpy as np
import pandas as pd
import datetime

def preprocess_input(user_data, encoder):
    """
    Transforms raw user input into the 170-feature vector for the model.
    """
    # 1. Calculate time-based features (matches feature_engineering.ipynb)
    now = datetime.datetime.now()
    hour = now.hour
    day_of_week = now.weekday()
    
    # Cyclic transformations
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # 2. Log transformation of amount
    amount_log = np.log1p(user_data['amount'])
    
    # 3. Categorical Encoding (Produces 61 features)
    # We create the 15 columns used in cat_cols in model_building script
    cat_df = pd.DataFrame([{
        'transaction type': user_data['type'],
        'merchant_category': user_data['category'],
        'transaction_status': user_data['status'],
        'sender_age_group': user_data['age'],
        'receiver_age_group': user_data['age'],
        'sender_state': user_data['state'],
        'sender_bank': user_data['bank'],
        'receiver_bank': user_data['bank'],
        'device_type': user_data['device'],
        'network_type': user_data['network'],
        'is_weekend': 'Yes' if day_of_week >= 5 else 'No',
        'year': now.year,
        'month': now.month,
        'day': now.day,
        'minute': now.minute
    }])
    
    encoded_cat = encoder.transform(cat_df)
    
    # 4. Padding to 170 features
    # Since we don't have all 109 numerical features, we fill the rest with zeros
    # but include our known engineered features
    numerical_features = np.zeros((1, 109))
    numerical_features[0, 0] = hour_sin
    numerical_features[0, 1] = hour_cos
    numerical_features[0, 2] = day_sin
    numerical_features[0, 3] = day_cos
    numerical_features[0, 4] = amount_log
    
    # Combine: 61 + 109 = 170
    final_input = np.hstack([encoded_cat, numerical_features])
    return final_input