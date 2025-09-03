import numpy as np
import random

def train_adaline(xin, yd, eta, tol, maxepocas, par=True):
    """
    Função de treinamento do Adaline
    
    Parâmetros:
    xin: matriz de entrada (N x dimensões)
    yd: vetor de saída desejada (N,)
    eta: taxa de aprendizagem
    tol: tolerância para o erro
    maxepocas: número máximo de épocas
    par: se True, adiciona bias (intercepto)
    
    Retorna:
    dict com 'pesos' e 'erros' (erro por época)
    """
    
    # Obter dimensões
    dimxin = xin.shape
    N = dimxin[0]
    n = dimxin[1]
    
    # Inicializar pesos
    if par:
        # Adicionar coluna de bias (uns)
        wt = np.column_stack([np.ones(N), xin])
        xin = np.column_stack([np.ones(N), xin])
        # Inicializar pesos aleatoriamente
        w = np.random.uniform(-0.5, 0.5, n + 1)
    else:
        wt = xin.copy()
        # Inicializar pesos aleatoriamente  
        w = np.random.uniform(-0.5, 0.5, n)
    
    # Inicializar contadores e lista de erros
    nepocas = 0
    ecpoca = tol + 1
    evec = np.zeros(maxepocas)
    
    # Loop principal de treinamento
    while (nepocas < maxepocas) and (ecpoca > tol):
        ei2 = 0  # erro quadrático da época
        
        # Criar sequência aleatória para embaralhar dados
        xseq = list(range(N))
        random.shuffle(xseq)
        
        # Treinar com cada exemplo na ordem embaralhada
        for i in range(N):
            irand = xseq[i]
            
            # Exemplo atual
            xvec = xin[irand, :]
            yhati = np.dot(xvec, w)  # saída da rede
            
            # Calcular erro
            ei = yd[irand] - yhati
            
            # Atualizar pesos usando regra delta
            dw = eta * ei * xvec
            w = w + dw
            
            # Acumular erro quadrático
            ei2 = ei2 + ei**2
        
        # Atualizar contadores
        nepocas = nepocas + 1
        evec[nepocas - 1] = ei2 / N  # erro médio quadrático
        ecpoca = evec[nepocas - 1]
    
    # Preparar resultado
    retlist = {
        'pesos': w,
        'erros': evec[:nepocas]  # retornar apenas as épocas utilizadas
    }
    
    return retlist

def predict_adaline(x, pesos, par=True):
    """
    Função para fazer predições com o Adaline treinado
    
    Parâmetros:
    x: dados de entrada
    pesos: pesos treinados
    par: se True, adiciona bias
    
    Retorna:
    predições
    """
    if par:
        # Adicionar coluna de bias
        if x.ndim == 1:
            x_bias = np.concatenate([[1], x])
        else:
            x_bias = np.column_stack([np.ones(x.shape[0]), x])
    else:
        x_bias = x
    
    return np.dot(x_bias, pesos)

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    np.random.seed(42)
    random.seed(42)
    
    # Gerar dados sintéticos
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * np.random.randn(100)
    
    # Treinar Adaline
    resultado = train_adaline(X, y, eta=0.01, tol=0.01, maxepocas=1000, par=True)
    
    print("Pesos finais:", resultado['pesos'])
    print("Número de épocas:", len(resultado['erros']))
    print("Erro final:", resultado['erros'][-1])
    
    # Fazer predição
    teste_x = np.array([1.0, 2.0])
    predicao = predict_adaline(teste_x, resultado['pesos'], par=True)
    print("Predição para [1.0, 2.0]:", predicao)
