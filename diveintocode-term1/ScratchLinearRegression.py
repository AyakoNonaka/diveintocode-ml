# 線形回帰classを実装

class ScratchLinearRegression():


    def __init__(self, num_iter=100, lr=0.000000001, bias=None, verbose=True):
 
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose

        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)        

        
# 学習

#     def fit(self, arX_train, ary_train, arX_test=None, ary_test=None):
    def fit(self, arX_train, ary_train, arX_test, ary_test):

        #パラメータ（＝weightの初期値）の配列を作る
        #self.weight = np.empty(arX_train.shape[1] + 1)
        self.weight = np.random.rand(arX_train.shape[1] + 1 )
        
        #biasを含めた特徴量の配列を作る
        arX_train_bias = np.insert(arX_train, 2, 1, axis=1)
        
        if (arX_test is not None) and (ary_test is not None):
            arX_test = np.insert(arX_test, -1, 1, axis=1)
        
        for i in range(self.iter):
            
            #最急降下法を実行
            self._gradient_descent(arX_train_bias, ary_train)      
            
            
            #学習データに対する損失を記録
            self.loss[i] = self._compute_cost(arX_train_bias, ary_train)
                        
            #検証用データに対する損失を記録
            if (arX_test is not None) and (ary_test is not None):
                self.val_loss[i] = self._compute_cost(arX_test, ary_test)
            

            #verboseをTrueにした際は学習過程を出力   
            if self.verbose:
                print("学習データの損失:" + str(self.loss[i]))
                print("検証用データの損失:" + str(self.val_loss[i]))
                print("coef:" + str(self.weight))
    
    
# 仮定関数を定義する関数
    
    def _linear_hypothesis(self,arX_train_bias):
        return np.dot(self.weight, arX_train_bias.T)    


# MSEを計算する関数を呼び出す

    def _compute_cost(self, arX_train_bias, ary_train):             
        return self.MSE(self._linear_hypothesis(arX_train_bias), ary_train)

    
# 最急降下法

    def _gradient_descent(self, arX_train_bias, ary_train):
        grad = np.mean((self._linear_hypothesis(arX_train_bias)[:, np.newaxis] - ary_train) * arX_train_bias, axis = 0)
        self.weight = self.weight - self.lr * grad


# 推定

    def predict(self, arX_test):
        arX_test_bias = np.insert(arX_test, -1, 1, axis=1)
        return self._linear_hypothesis(arX_test_bias)
        
    
# MSEを導く関数 
    
    def MSE(self, ary_pred, ary_train):
        mse = np.mean((ary_pred - ary_train)**2 / 2)
        return mse
    
    
#グラフをプロットする関数を作る

    def drow(self):
        fig = plt.figure(figsize=(10, 8))
        plt.title("Learning Records")
        plt.xlabel("Number of Iterrations")
        plt.ylabel("Loss")
        plt.plot(self.loss)
        plt.plot(self.val_loss)    
        plt.show()