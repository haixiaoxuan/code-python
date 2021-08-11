from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import xgboost as xgb


"""
    有3个目标函数,Point Wise,Pairwise和Listwise,这3种方法都可以用来排序,每个方法都有其优缺点.对于pointwise而言,每次仅仅考虑一个样本,预估的是每一条和query的相关性,基于此进行排序.Pairwise是每次取一对样本,预估这一对样本的先后顺序,不断重复预估一对对样本,从而得到某条query下完整的排序.Listwise同时考虑多个样本,找到最优顺序.
    Point Wise虽然简单,但是存在不少问题.比如说赛马场景,马的输赢取决于对手.再比如搜索场景,我们确实可以预估每个query到每个document的点击率做为排序依据,但是点击率要考虑rank,例如排的越靠前的document点击率上占据优势,这些point-wise模型很难考虑进去.基于此,我们需要做learning to rank的模型.
    
"""


df = pd.DataFrame()
# 需要按query id进行分割,如果直接随机拆分,同一个query id下的数据就会被分开,这样会导致模型出问题
gss = GroupShuffleSplit(test_size=.40,
                        n_splits=1,
                        random_state=7).split(df, groups=df['query_id'])
X_train_inds, X_test_inds = next(gss)

train_data = df.iloc[X_train_inds]
X_train = train_data.loc[:, ~train_data.columns.isin(['id', 'rank'])]
y_train = train_data.loc[:, train_data.columns.isin(['rank'])]

# 模型需要输入按query_id排序后的样本
# 并且需要给定每个query_id下样本的数量
groups = train_data.groupby('id').size().to_frame('size')['size'].to_numpy()        # 计算每个query_id 下的样品数量


test_data = df.iloc[X_test_inds]
# We need to keep the id for later predictions
X_test = test_data.loc[:, ~test_data.columns.isin(['rank'])]
y_test = test_data.loc[:, test_data.columns.isin(['rank'])]


# 然后我们就可以建模了,可以用XGBRanker训练排序模型,在这个场景下,我们无法自定义objective,也无法自定义mertic了.
model = xgb.XGBRanker(
    tree_method='gpu_hist',
    booster='gbtree',
    objective='rank:pairwise',      # rank:ndcg rank:map
    random_state=42,
    learning_rate=0.1,
    colsample_bytree=0.9,
    eta=0.05,
    max_depth=6,
    n_estimators=110,
    subsample=0.75
    )
model.fit(X_train, y_train, group=groups, verbose=True)


# 训练完后我们就可以进行预估,因为预估方法并不会输入groups,所以我们需要做一些特殊处理:
def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['id'])])


predictions = (data.groupby('id')
               .apply(lambda x: predict(model, x)))


"""
    pair wise 方法相比pointwise有优势，可以学习到一些顺序。但是pairwise也有缺点：
        1.只能给出排序,并不能给出有多好,好多少.比如在搜索场景下,可能一条与query相关的doc都没,pointwise可以通过卡阈值得到这个信息,但是rank方式就不能区分.
        2.当一个query下有很多doc，会产生大量的pairs。
        3.对噪声的label 非常敏感。
"""