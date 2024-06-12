import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # 연령 범주를 중앙값으로 인코딩
    encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                          '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                          '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                          '30-34':32,'25-29':27}
    df['AgeCategory'] = df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
    df['AgeCategory'] = df['AgeCategory'].astype('float')
    
    df_cleaned = df[['HeartDisease','Smoking','DiffWalking','Diabetic','PhysicalActivity','PhysicalHealth','AgeCategory']]
    train_data, test_data = train_test_split(df_cleaned, test_size=0.2, random_state=42)
    train_data.replace({'Yes':1,'No':0}, inplace=True)
    test_data.replace({'Yes':1, 'No':0}, inplace=True)

    y_train = train_data['HeartDisease']
    X_train = train_data.drop('HeartDisease', axis=1)
    y_test = test_data['HeartDisease']
    X_test = test_data.drop('HeartDisease', axis=1)

    one_hot_train = pd.get_dummies(X_train)
    one_hot_test = pd.get_dummies(X_test)
    one_hot_train, one_hot_test = one_hot_train.align(one_hot_test, join='outer', axis=1, fill_value=0)

    one_hot_train = one_hot_train.astype(float)
    one_hot_test = one_hot_test.astype(float)

    return one_hot_train, y_train, one_hot_test, y_test
