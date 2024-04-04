import numpy as np
from random import sample
from sklearn.model_selection import train_test_split
from warnings import filterwarnings

class PLA ():
    def __init__(self, n_int : int = 1000) -> None:
        self.n_int = n_int
    
    def acuracia (self, X : np.array, Y : np.array, w_lista : np.array) -> float:
        lista_x = np.concatenate((np.ones((len(X), 1)), X), axis = 1)
        soma_PCC = 0

        for i in range (len(X)):
            aux = np.sign(np.matmul(w_lista, lista_x[i])) #y_predict

            if (aux == Y[i]):
                soma_PCC += 1
    
        return (soma_PCC/len(X))
    
    def __construtor_PCI (self, X : np.array, Y : np.array) -> np.array:
        lista_x = np.concatenate((np.ones((len(X), 1)), X), axis = 1)

        lista_PCI_x = []
        lista_PCI_y = []

        for i in range (len(X)):
            aux = np.sign(np.matmul(self.w_lista, lista_x[i]))

            if (aux != Y[i]):
                lista_PCI_x.append(lista_x[i])
                lista_PCI_y.append(Y[i])
        
        return np.array(lista_PCI_x), np.array(lista_PCI_y) 

    def fit (self, X : np.array, Y : np.array) -> None:
        lista_PCI_x = np.concatenate((np.ones((len(X), 1)), X), axis = 1) 
        lista_PCI_y = Y
        self.w_lista, w_otimo = np.zeros(lista_PCI_x.shape[1]), np.zeros(lista_PCI_x.shape[1])

        i = 0
        while (len(lista_PCI_x) > 0) and (i < self.n_int):
            ale_index = np.random.randint(0, len(lista_PCI_x)) #index aleatório
            ponto_x = lista_PCI_x[ale_index]
            ponto_y = lista_PCI_y[ale_index]
            
            aux = ponto_x * ponto_y
            w_novo = np.add(self.w_lista, aux)

            #Pocket:
            if (self.acuracia (X, Y, w_otimo) < self.acuracia (X, Y, w_novo)):
                w_otimo = w_novo

            self.w_lista = w_novo

            lista_PCI_x, lista_PCI_y = self.__construtor_PCI (X = X, Y = Y)
            i += 1
        
        self.w_lista = w_otimo

        return
    
    def predict (self, X : np.array) -> np.array:
        lista_x = np.concatenate((np.ones((len(X), 1)), X), axis = 1)
        predict_y = [np.sign(np.matmul(i, self.w_lista)) for i in lista_x]

        return predict_y

    def get_w (self) -> np.array:
        try:
            return self.w_lista
        
        except:
            print ("Não foi possível recuperar w. Por favor, se certifique de treinar o modelo antes.\n")

    def set_w (self, novo_w : np.array) -> None:
        self.w_lista = novo_w

class Reg_Lin ():
    def __init__ (self, n_int = None) -> None:
        self.n_int = n_int
    
    def acuracia (self, X : np.array, Y : np.array) -> float:
        predict_y = self.predict(X = X)

        soma_PCC = 0
        for i in range (len(Y)):
            if (Y[i] == predict_y[i]):
                soma_PCC += 1
        
        return (soma_PCC/len(Y))

    def fit (self, X : np.array, Y : np.array) -> None:
        lista_x = np.concatenate((np.ones((len(X), 1)), X), axis = 1)

        xTx = np.dot(lista_x.transpose(), lista_x)
        inverse = np.linalg.inv(xTx)
        self.w_lista = np.dot(np.dot(inverse, lista_x.transpose()), Y)

        return None
    
    def predict (self, X : np.array) -> int:
        lista_X = np.concatenate((np.ones((len(X), 1)), X), axis = 1)
        pred_y = [np.sign(np.dot(np.transpose(self.w_lista), xi)) for xi in lista_X] #sign(wTx)
        
        return np.array(pred_y)

    def get_w (self):
        try:
            return self.w_lista
        
        except:
            print ("Não foi possível recuperar w. Por favor, se certifique de treinar o modelo antes.\n")

    def set_w (self, novo_w : np.array) -> None:
        self.w_lista = novo_w

class Reg_Log ():
    def __init__ (self, eta = 0.1, n_int = 1000, tam_batch = 50, lamb = 0) -> None:
        self.eta = eta
        self.lamb = lamb
        self.n_int = n_int
        self.tam_batch = tam_batch

    def acuracia (self, X : np.array, Y : np.array) -> float:
        y_predict = self.predict(X = X)
        soma_PCC = 0

        for i in range (len(Y)):
            if (Y[i] == y_predict[i]):
                soma_PCC += 1
            
        return (soma_PCC/len(Y))
    
    def fit (self, X : np.array, Y : np.array, eps : float = 0.0001) -> None:
        lista_X = np.concatenate((np.ones((len(X), 1)), X), axis = 1) #O 1 será usado para a multiplicação com o bias posteriormente
        lista_y = Y
        
        n_dim = len(lista_X[0]) #nº de features
        n_elem = len(lista_X) #nº de pontos/dados
        w_lista = np.zeros(n_dim, dtype = float)

        #Cálculo dos gradientes pelo processo iterativo
        for i in range (self.n_int):
            vsoma = np.zeros(n_dim, dtype = 'float64')

            if (self.tam_batch) < n_elem:
                batch_X = []
                batch_Y = []
                indices = sample(range(n_elem), self.tam_batch)

                for j in indices:
                    batch_X.append(lista_X[j])
                    batch_Y.append(lista_y[j])
            
            else:
                batch_X = lista_X
                batch_Y = lista_y
            
            for xn, yn in zip(batch_X, batch_Y):
                aux = np.matmul(np.transpose(yn * w_lista), xn)
                vsoma += np.array((yn * xn) / (1 + np.exp(aux, dtype = "float128")), dtype = "float128")
            
            grad_t = (-1 / self.tam_batch) * vsoma

            #Aplicando o weight decay, se aplicável
            if (self.lamb != 0):
                grad_t += (2 * self.lamb * w_lista)

            #Testando o minimo
            if (np.linalg.norm(grad_t) <  eps):
                break

            w_lista = w_lista - (self.eta * grad_t)
        
        self.w_lista = w_lista

    def predict (self, X : np.array) -> np.array:
        lista_X = np.concatenate((np.ones((len(X), 1)), X), axis = 1)

        aux = [1 / (1 + np.exp(- np.matmul(np.transpose(self.w_lista), i))) for i in lista_X]
        predict_y = [1 if i >= 0.5 else -1 for i in aux]

        return predict_y

    def get_w (self) -> np.array:
        try:
            return self.w_lista
        
        except:
            print ("Não foi possível recuperar w. Por favor, se certifique de treinar o modelo antes.\n")

    def set_w (self, novo_w : np.array) -> None:
        self.w_lista = novo_w

class Weight_Decay (Reg_Log):
    def __init__(self, eta = 0.1, n_int = 1000, tam_batch = 50, lamb = 0) -> None:
        super().__init__ (eta, n_int, tam_batch, lamb)
    
    def fit (self, X : np.array, Y : np.array) -> None:
        X_train, X_val, y_train, y_val = train_test_split (X, Y, test_size = 0.2)
        self.lista_lambda = []
        
        #Atribuindo valores arbitrários aos lambdas:
        lambdas = [10 ** x for x in range(-5, 5)]
        lambdas.append(0) #Caso sem weight-decay
        lambdas = sorted(lambdas)

        Eout_otimo = 1 #Instanciando o melhor erro no pior cenário
 
        filterwarnings('error') #Evitando os warnings pelos lambdas serem valores pequenos
        for lamb in lambdas:
            classificador = Reg_Log (n_int = 1000, lamb = lamb)

            try:
                classificador.fit (X_train, y_train)

            except (RuntimeWarning):
                continue 

            #Computando o erro quadrático (Eout)
            EOut = self.get_Eout (classificador.predict (X_val), y_val)                
            
            if (EOut < Eout_otimo):
                Eout_otimo = EOut
                self.classificador = classificador
            
            self.lista_lambda.append([lamb, EOut])
                    
        self.w_lista = self.classificador.w_lista
    
    def get_Eout (self, pred_y : np.array, y_val : np.array):
        #Computando o erro quadrático (Eout)
        soma_PCI = 0

        for i in range (len(y_val)):

            if(pred_y[i] != y_val[i]):
                soma_PCI += 1

        return (soma_PCI/(len(y_val)))
    
    def get_lista_lambda (self) -> list:
        try:
            return self.lista_lambda
        
        except:
            return None