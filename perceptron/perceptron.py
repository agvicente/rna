import numpy as np

class Perceptron:
    def __init__(self, learning_rate, maxepocas, use_zero_one_labels=False):
        self.learning_rate = learning_rate
        self.maxepocas = maxepocas
        self.use_zero_one_labels = use_zero_one_labels  # True para usar labels 0/1, False para -1/1
        self.w_ = None
        self.errors_ = []

    def train_perceptron(self, X, y, tol, init_weights_zero=False):
        """
        Treina o classificador Perceptron
        
        Parametros:
        X : array, shape = [n_samples, n_features]
            Dados de treinamento
        y : array, shape = [n_samples]
            Valores alvo (-1 ou 1) ou (0 ou 1) dependendo de use_zero_one_labels
        tol : float
            Tolerancia para convergencia (numero de erros por epoca)
        init_weights_zero : bool
            Se True, inicializa pesos com zeros (como no código R)
            
        Retorna:
        self.w_ : array, shape = [n_features + 1]
            Pesos apos o treinamento
        self.errors_ : list
            Lista com o numero de erros de cada epoca
        """
        N = X.shape[0]
        n_features = X.shape[1]
        
        # Inicializa pesos
        if init_weights_zero:
            # Inicializa com zeros (como no código R)
            self.w_ = np.zeros(n_features + 1)
        else:
            # Inicializa com valores aleatórios pequenos
            np.random.seed(42)  # Para reproducibilidade
            self.w_ = np.random.uniform(-0.5, 0.5, n_features + 1)
        
        nepocas = 0
        self.errors_ = []
        
        while nepocas < self.maxepocas:
            # Embaralha as amostras
            xseq = np.random.permutation(N)
            eepoca = 0  # Contador de erros da epoca
            
            for i in range(N):
                iseq = xseq[i]
                # Cria vetor com bias (1) + features
                xvec = np.concatenate([[1.0], X[iseq]])
                
                # Calcula saida linear (net input)
                net_input = np.dot(self.w_, xvec)
                
                # Aplica funcao de ativacao step
                if self.use_zero_one_labels:
                    # Para labels 0/1 (como no código R)
                    yhat = 1.0 if net_input >= 0.0 else 0.0
                else:
                    # Para labels -1/1 (versão clássica)
                    yhat = 1.0 if net_input >= 0.0 else -1.0
                
                # Calcula erro usando saida binaria
                ei = y[iseq] - yhat
                
                # Atualiza pesos apenas se houver erro (ei != 0)
                if ei != 0:
                    dw = self.learning_rate * ei * xvec
                    self.w_ += dw
                    eepoca += 1  # Incrementa contador de erros
            
            # Armazena numero de erros da epoca
            self.errors_.append(eepoca)
            
            nepocas += 1
            
            # Verifica convergencia (sem erros ou abaixo da tolerancia)
            if eepoca <= tol:
                break
        
        return self.w_, self.errors_
    
    def net_input(self, X):
        """
        Calcula o net input (saida linear) do modelo
        
        Parametros:
        X : array, shape = [n_samples, n_features]
            Dados para calculo do net input
            
        Retorna:
        array, shape = [n_samples]
            Net input (saida linear antes da funcao de ativacao)
        """
        if self.w_ is None:
            raise ValueError("Modelo nao foi treinado ainda. Chame train_perceptron primeiro.")
        
        # Adiciona bias (coluna de 1s)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return np.dot(X_with_bias, self.w_)
    
    def predict(self, X):
        """
        Faz predicoes usando funcao de ativacao step
        
        Parametros:
        X : array, shape = [n_samples, n_features]
            Dados para predicacao
            
        Retorna:
        array, shape = [n_samples]
            Classes preditas (0 ou 1) ou (-1 ou 1) dependendo de use_zero_one_labels
        """
        net_inputs = self.net_input(X)
        if self.use_zero_one_labels:
            return np.where(net_inputs >= 0.0, 1, 0)
        else:
            return np.where(net_inputs >= 0.0, 1, -1)