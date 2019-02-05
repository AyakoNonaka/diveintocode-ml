# ライブラリ・モジュールを読み込む
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='data file')
parser.add_argument('--classifier', type=str,
                    help='classifier:LogisticRegression, SVC, DecisionTreeClassifier')

# 分類器
def main(classifier, X_train, X_test, y_train):
    print("test1")
    
    if classifier == "LogisticRegression":       
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print("y_pred : {}".format(x))
        score = model.score(X_train, y_train) 
        print("score : {}".format(x))
        
    elif classifier == "SVC":
        svc = SVC()
        svc.fit(X_train, y_train) 
        y_pred = svc.predict(X_test)
        print("y_pred : {}".format(x))
        score = model.score(X_train, y_train) 
        print("score : {}".format(x))

    elif classifier == "DecisionTreeClassifier":
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        print("y_pred : {}".format(x))
        score = model.score(X_train, y_train) 
        print("score : {}".format(x))
        
    else:
        print('error:select LogisticRegression, SVC, DecisionTreeClassifier')

            
if __name__ == '__main__':
    # pyファイルの実行時にはまずここが実行される。

    # コマンドライン引数の読み込み
    args = parser.parse_args()
    pd.read_csv(args.data)
    X = args.data[0:1,0:1]
    y = args.data[0:1,0:1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=10)
    main(args.classifier)
    
    

    
# # ロジスティック回帰
# from sklearn.linear_model import LogisticRegression
# clf_lr = LogisticRegression()
# clf_lr.fit(X_train, y_train) 
# y_pred = clf_lr.predict(X_test)
# score = model.score(X_train, y_train) 
# print(score)


# # SVC
# from sklearn.svm import SVC
# clf_svc = SVC()
# clf_svc.fit(X_train, y_train) 
# y_pred = clf_svc.predict(X_test) 
# y_pred 



# # 決定木
# from sklearn.tree import DecisionTreeClassifier
# clf_dtc = DecisionTreeClassifier()
# clf_dtc.fit(X_train, y_train)
# y_pred = clf_dtc.predict(X_test)