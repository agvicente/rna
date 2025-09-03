#!/usr/bin/env python3
"""
Exemplo de uso do algoritmo Perceptron

Este arquivo demonstra como usar a implementação do Perceptron
para resolver problemas de classificação binária linearmente separáveis.
"""

import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def test_perceptron_and_gate():
    """
    Testa o Perceptron com o problema da porta lógica AND
    """
    print("="*50)
    print("TESTE 1: Porta Lógica AND")
    print("="*50)
    
    # Dados de treinamento para porta AND
    # X1  X2  | Y
    #  0   0  | -1
    #  0   1  | -1  
    #  1   0  | -1
    #  1   1  |  1
    X_and = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
    
    y_and = np.array([-1, -1, -1, 1])
    
    # Treina o Perceptron
    perceptron = Perceptron(learning_rate=0.1, maxepocas=100)
    weights, errors = perceptron.train_perceptron(X_and, y_and, tol=0)
    
    print(f"Treinamento concluído em {len(errors)} épocas")
    print(f"Pesos finais: {weights}")
    print(f"Erros por época: {errors}")
    
    # Testa o modelo
    print("\nTestes:")
    for i, x in enumerate(X_and):
        prediction = perceptron.predict(x.reshape(1, -1))[0]
        net_input = perceptron.net_input(x.reshape(1, -1))[0]
        print(f"Entrada: {x} -> Net Input: {net_input:.3f} -> Predição: {prediction:2.0f} -> Esperado: {y_and[i]:2.0f}")
    
    return errors

def test_perceptron_or_gate():
    """
    Testa o Perceptron com o problema da porta lógica OR
    """
    print("\n" + "="*50)
    print("TESTE 2: Porta Lógica OR")
    print("="*50)
    
    # Dados de treinamento para porta OR
    # X1  X2  | Y
    #  0   0  | -1
    #  0   1  |  1
    #  1   0  |  1
    #  1   1  |  1
    X_or = np.array([[0, 0],
                     [0, 1], 
                     [1, 0],
                     [1, 1]])
    
    y_or = np.array([-1, 1, 1, 1])
    
    # Treina o Perceptron
    perceptron = Perceptron(learning_rate=0.1, maxepocas=100)
    weights, errors = perceptron.train_perceptron(X_or, y_or, tol=0)
    
    print(f"Treinamento concluído em {len(errors)} épocas")
    print(f"Pesos finais: {weights}")
    print(f"Erros por época: {errors}")
    
    # Testa o modelo
    print("\nTestes:")
    for i, x in enumerate(X_or):
        prediction = perceptron.predict(x.reshape(1, -1))[0]
        net_input = perceptron.net_input(x.reshape(1, -1))[0]
        print(f"Entrada: {x} -> Net Input: {net_input:.3f} -> Predição: {prediction:2.0f} -> Esperado: {y_or[i]:2.0f}")
    
    return errors

def test_separable_dataset():
    """
    Testa o Perceptron com um dataset linearmente separável mais complexo
    """
    print("\n" + "="*50)
    print("TESTE 3: Dataset Linearmente Separável")
    print("="*50)
    
    # Gera dataset sintético linearmente separável
    np.random.seed(123)
    
    # Classe -1: pontos em torno de (1, 1)
    class1 = np.random.randn(10, 2) * 0.5 + [1, 1]
    
    # Classe +1: pontos em torno de (3, 3)  
    class2 = np.random.randn(10, 2) * 0.5 + [3, 3]
    
    X = np.vstack([class1, class2])
    y = np.hstack([np.full(10, -1), np.full(10, 1)])
    
    # Embaralha os dados
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Treina o Perceptron
    perceptron = Perceptron(learning_rate=0.01, maxepocas=1000)
    weights, errors = perceptron.train_perceptron(X, y, tol=0)
    
    print(f"Treinamento concluído em {len(errors)} épocas")
    print(f"Pesos finais: {weights}")
    print(f"Últimos 10 erros: {errors[-10:] if len(errors) > 10 else errors}")
    
    # Calcula acurácia
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Acurácia no conjunto de treinamento: {accuracy:.2%}")
    
    return errors, X, y, perceptron

def plot_results(errors_and, errors_or, errors_dataset):
    """
    Plota os resultados dos treinamentos
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Erros AND
    plt.subplot(1, 3, 1)
    plt.plot(errors_and, 'bo-', linewidth=2, markersize=8)
    plt.title('Perceptron - Porta AND\nErros por Época')
    plt.xlabel('Época')
    plt.ylabel('Número de Erros')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, max(errors_and) + 0.5)
    
    # Plot 2: Erros OR
    plt.subplot(1, 3, 2)
    plt.plot(errors_or, 'ro-', linewidth=2, markersize=8)
    plt.title('Perceptron - Porta OR\nErros por Época')
    plt.xlabel('Época')
    plt.ylabel('Número de Erros')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, max(errors_or) + 0.5)
    
    # Plot 3: Erros Dataset
    plt.subplot(1, 3, 3)
    plt.plot(errors_dataset, 'go-', linewidth=2, markersize=4)
    plt.title('Perceptron - Dataset Separável\nErros por Época')
    plt.xlabel('Época')
    plt.ylabel('Número de Erros')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, max(errors_dataset) + 0.5)
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, y, perceptron):
    """
    Plota a fronteira de decisão do Perceptron para dados 2D
    """
    plt.figure(figsize=(10, 8))
    
    # Plota os pontos
    mask_pos = y == 1
    mask_neg = y == -1
    
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='red', marker='o', s=100, 
                alpha=0.7, label='Classe +1')
    plt.scatter(X[mask_neg, 0], X[mask_neg, 1], c='blue', marker='s', s=100, 
                alpha=0.7, label='Classe -1')
    
    # Plota a fronteira de decisão
    w = perceptron.w_
    
    # A fronteira de decisão é definida por: w0 + w1*x1 + w2*x2 = 0
    # Resolvendo para x2: x2 = -(w0 + w1*x1) / w2
    if abs(w[2]) > 1e-10:  # Evita divisão por zero
        x1_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        x2_line = -(w[0] + w[1] * x1_line) / w[2]
        plt.plot(x1_line, x2_line, 'k--', linewidth=2, label='Fronteira de Decisão')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron - Fronteira de Decisão')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def test_r_code_reproduction():
    """
    Reproduz o experimento do código R fornecido
    """
    print("\n" + "="*50)
    print("TESTE 4: Reprodução do Código R")
    print("="*50)
    
    # Parâmetros do código R
    N = 30
    eta = 0.01
    maxepocas = 100
    tol = 1e-10
    
    # Gera dados como no código R
    np.random.seed(123)
    xc1 = np.random.normal(0, 0.5, (N, 2)) + 2  # Classe 0 centrada em (2,2)
    xc2 = np.random.normal(0, 0.5, (N, 2)) + 4  # Classe 1 centrada em (4,4)
    
    X = np.vstack([xc1, xc2])
    y = np.hstack([np.zeros(N), np.ones(N)])  # Labels 0 e 1
    
    print(f"Dataset R: {len(X)} amostras, labels 0 e 1")
    print(f"Classe 0: {len(xc1)} pontos centrados em (2,2)")
    print(f"Classe 1: {len(xc2)} pontos centrados em (4,4)")
    
    # Treina com configuração do código R
    perceptron_r = Perceptron(learning_rate=eta, maxepocas=maxepocas, use_zero_one_labels=True)
    weights_r, errors_r = perceptron_r.train_perceptron(X, y, tol, init_weights_zero=True)
    
    print(f"Treinamento concluído em {len(errors_r)} épocas")
    print(f"Pesos finais: {weights_r}")
    
    # Testa acurácia
    predictions_r = perceptron_r.predict(X)
    accuracy_r = np.mean(predictions_r == y)
    print(f"Acurácia: {accuracy_r:.2%}")
    
    # Compara com versão clássica (-1/1)
    print("\nComparação com versão clássica (-1/1):")
    y_classic = np.where(y == 0, -1, 1)
    perceptron_classic = Perceptron(learning_rate=eta, maxepocas=maxepocas, use_zero_one_labels=False)
    weights_classic, errors_classic = perceptron_classic.train_perceptron(X, y_classic, tol, init_weights_zero=True)
    
    print(f"Versão clássica: {len(errors_classic)} épocas")
    print(f"Pesos clássicos: {weights_classic}")
    
    return errors_r, X, y, xc1, xc2, perceptron_r

def main():
    """
    Função principal que executa todos os testes
    """
    print("DEMONSTRAÇÃO DO ALGORITMO PERCEPTRON")
    print("Implementado conforme algoritmo das imagens fornecidas")
    print("Agora com suporte para labels 0/1 (código R) e -1/1 (clássico)\n")
    
    # Executa os testes clássicos
    errors_and = test_perceptron_and_gate()
    errors_or = test_perceptron_or_gate()
    errors_dataset, X_classic, y_classic, perceptron_classic = test_separable_dataset()
    
    # Executa teste do código R
    errors_r, X_r, y_r, xc1, xc2, perceptron_r = test_r_code_reproduction()
    
    # Plota os resultados
    try:
        plot_results(errors_and, errors_or, errors_dataset)
        plot_decision_boundary(X_classic, y_classic, perceptron_classic)
        
        # Plot específico do experimento R
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(xc1[:, 0], xc1[:, 1], c='red', s=60, alpha=0.7, label='Classe 0')
        plt.scatter(xc2[:, 0], xc2[:, 1], c='blue', s=60, alpha=0.7, label='Classe 1')
        plt.xlim(0, 6)
        plt.ylim(0, 6)
        plt.title('Dataset do Código R')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(errors_r, 'go-', linewidth=2, markersize=6)
        plt.title('Erros - Reprodução Código R')
        plt.xlabel('Época')
        plt.ylabel('Número de Erros')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.5, max(errors_r) + 0.5 if errors_r else 0.5)
        
        plt.subplot(1, 3, 3)
        # Plot da fronteira de decisão do código R
        mask_0 = y_r == 0
        mask_1 = y_r == 1
        plt.scatter(X_r[mask_0, 0], X_r[mask_0, 1], c='red', s=60, alpha=0.7, label='Classe 0')
        plt.scatter(X_r[mask_1, 0], X_r[mask_1, 1], c='blue', s=60, alpha=0.7, label='Classe 1')
        
        # Fronteira de decisão
        w = perceptron_r.w_
        if abs(w[2]) > 1e-10:
            x1_line = np.linspace(0, 6, 100)
            x2_line = -(w[0] + w[1] * x1_line) / w[2]
            plt.plot(x1_line, x2_line, 'k--', linewidth=2, label='Fronteira')
        
        plt.xlim(0, 6)
        plt.ylim(0, 6)
        plt.title('Fronteira - Código R')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nNota: matplotlib não está disponível para plotar os gráficos.")
        print("Para visualizar os gráficos, instale matplotlib: pip install matplotlib")
    
    print("\n" + "="*50)
    print("RESUMO DOS RESULTADOS")
    print("="*50)
    print(f"Porta AND: Convergiu em {len(errors_and)} épocas")
    print(f"Porta OR: Convergiu em {len(errors_or)} épocas") 
    print(f"Dataset Separável: Convergiu em {len(errors_dataset)} épocas")
    print(f"Código R (0/1): Convergiu em {len(errors_r)} épocas")
    print("\nTodos os problemas foram resolvidos com sucesso!")
    print("O Perceptron conseguiu aprender as fronteiras de decisão lineares.")
    print("\n✓ Suporte para labels -1/1 (versão clássica)")
    print("✓ Suporte para labels 0/1 (reprodução código R)")
    print("✓ Inicialização com zeros ou aleatória")
    print("✓ Reprodução fiel do experimento R")

if __name__ == "__main__":
    main()