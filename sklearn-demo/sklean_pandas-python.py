from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


"""
    对 pandas 中指定的列使用指定的方法进行处理    
"""


# SimleImputer 可以进行缺失值补全
imputer_Pclass = SimpleImputer(strategy='most_frequent', add_indicator=True)
imputer_Age = SimpleImputer(strategy='median', add_indicator=True)
imputer_SibSp = SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)
imputer_Parch = SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)
imputer_Fare = SimpleImputer(strategy='median', add_indicator=True)
imputer_Embarked = SimpleImputer(strategy='most_frequent')

scaler_Age = MinMaxScaler()
scaler_Fare = StandardScaler()

onehotencoder_Sex = OneHotEncoder(drop=['male'], handle_unknown='error')
onehotencoder_Embarked = OneHotEncoder(handle_unknown='error')


mapper = DataFrameMapper([
    (['Age'], [imputer_Age, scaler_Age], {'alias':'Age_scaled'}),
    (['Pclass'], [imputer_Pclass]),
    (['SibSp'], [imputer_SibSp]),
    (['Parch'], [imputer_Parch]),
    (['Fare'], [imputer_Fare, scaler_Fare], {'alias': 'Fare_scaled'}),
    (['Sex'], [onehotencoder_Sex], {'alias': 'is_female'}),
    (['Embarked'], [imputer_Embarked, onehotencoder_Embarked])
], df_out=True)

mapper.fit(X=train, y=train['Survived'])