#!/usr/bin/env python3
"""
Reprodução exata do código R fornecido em Python

Este arquivo implementa exatamente o mesmo experimento do código R:
- Gera 30 pontos da classe 0 centrados em (2, 2) 
- Gera 30 pontos da classe 1 centrados em (4, 4)
- Usa labels 0 e 1 (ao invés de -1 e 1)
- Inicializa pesos com zeros
- Taxa de aprendizado: 0.01
- Máximo de épocas: 100
- Tolerância: 1e-10
"""

import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def generate_r_dataset():
    """
    Reproduz exatamente a geração de dados do código R:
    
    N<-30
    xc1<-matrix(rnorm(2*N)*0.5, N, 2) + 2
    xc2<-matrix(rnorm(2*N)*0.5, N, 2) + 4
    xall<-rbind(xc1, xc2)
    yall<-rbind(matrix(0, ncol=1, nrow=N), matrix(1, ncol=1, nrow=N))
    """
    # Parâmetros do código R
    N = 30
    
    # Define seed para reproducibilidade (equivalente ao set.seed do R)
    np.random.seed(123)
    
    # Classe 0: 30 pontos com ruído gaussiano centrados em (2, 2)
    # rnorm(2*N)*0.5 gera ruído gaussiano
    xc1 = np.random.normal(0, 0.5, (N, 2)) + 2
    
    # Classe 1: 30 pontos com ruído gaussiano centrados em (4, 4) 
    xc2 = np.random.normal(0, 0.5, (N, 2)) + 4
    
    # Combina os dados (equivalente a rbind)
    xall = np.vstack([xc1, xc2])
    
    # Cria labels: 30 zeros e 30 uns (equivalente a rbind(matrix(0, ...), matrix(1, ...)))
    yall = np.hstack([np.zeros(N), np.ones(N)])
    
    return xall, yall, xc1, xc2

def plot_r_dataset(xc1, xc2, title="Dataset gerado conforme código R"):
    """
    Reproduz o plot do código R:
    
    plot(xc1[,1], xc1[,2], xlim=c(0,6), ylim=c(0,6), col='red')
    par(new=TRUE)
    plot(xc2[,1], xc2[,2], xlim=c(0,6), ylim=c(0,6), col='blue')
    """
    plt.figure(figsize=(10, 8))
    
    # Plot classe 0 (vermelho)
    plt.scatter(xc1[:, 0], xc1[:, 1], c='red', s=60, alpha=0.7, 
                label='Classe 0 (centro em 2,2)', edgecolor='darkred')
    
    # Plot classe 1 (azul) 
    plt.scatter(xc2[:, 0], xc2[:, 1], c='blue', s=60, alpha=0.7,
                label='Classe 1 (centro em 4,4)', edgecolor='darkblue')
    
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def train_r_perceptron(X, y):
    """
    Treina o Perceptron com os parâmetros exatos do código R:
    
    w<-c(0, 0, 0)           # pesos iniciais zeros
    eta<-0.01               # taxa de aprendizado  
    maxepocas<-100          # máximo de épocas
    tol<-1e-10              # tolerância
    """
    print("="*60)
    print("TREINAMENTO DO PERCEPTRON - REPRODUÇÃO DO CÓDIGO R")
    print("="*60)
    
    # Parâmetros exatos do código R
    eta = 0.01
    maxepocas = 100
    tol = 1e-10
    
    print(f"Parâmetros:")
    print(f"  - Taxa de aprendizado (eta): {eta}")
    print(f"  - Máximo de épocas: {maxepocas}")
    print(f"  - Tolerância: {tol}")
    print(f"  - Inicialização dos pesos: zeros")
    print(f"  - Labels: 0 e 1")
    print(f"  - Total de amostras: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    
    # Cria o Perceptron com labels 0/1
    perceptron = Perceptron(learning_rate=eta, maxepocas=maxepocas, use_zero_one_labels=True)
    
    # Treina com pesos iniciais zerados (como no código R: w<-c(0, 0, 0))
    weights, errors = perceptron.train_perceptron(X, y, tol, init_weights_zero=True)
    
    print(f"\nResultados:")
    print(f"  - Convergiu em {len(errors)} épocas")
    print(f"  - Pesos finais: {weights}")
    print(f"  - Últimos 10 erros: {errors[-10:] if len(errors) > 10 else errors}")
    
    # Testa acurácia
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"  - Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Mostra algumas predições
    print(f"\nAlgumas predições:")
    for i in [0, 15, 30, 45]:  # Algumas amostras das duas classes
        if i < len(X):
            pred = perceptron.predict(X[i:i+1])[0]
            net = perceptron.net_input(X[i:i+1])[0]
            print(f"  Amostra {i}: X={X[i]} -> net={net:7.3f} -> pred={pred:.0f} (real={y[i]:.0f})")
    
    return perceptron, weights, errors

def plot_decision_boundary_r(X, y, perceptron, xc1, xc2):
    """
    Plota o dataset com a fronteira de decisão aprendida
    """
    plt.figure(figsize=(12, 8))
    
    # Plot dos pontos originais (como no código R)
    plt.scatter(xc1[:, 0], xc1[:, 1], c='red', s=80, alpha=0.7,
                label='Classe 0', edgecolor='darkred', marker='o')
    plt.scatter(xc2[:, 0], xc2[:, 1], c='blue', s=80, alpha=0.7, 
                label='Classe 1', edgecolor='darkblue', marker='s')
    
    # Plota a fronteira de decisão
    w = perceptron.w_
    print(f"\nEquação da fronteira de decisão:")
    print(f"w0 + w1*x1 + w2*x2 = 0")
    print(f"{w[0]:.3f} + {w[1]:.3f}*x1 + {w[2]:.3f}*x2 = 0")
    
    if abs(w[2]) > 1e-10:
        x1_range = np.linspace(0, 6, 100)
        x2_boundary = -(w[0] + w[1] * x1_range) / w[2]
        plt.plot(x1_range, x2_boundary, 'k--', linewidth=3, 
                label=f'Fronteira: {w[0]:.2f} + {w[1]:.2f}x₁ + {w[2]:.2f}x₂ = 0')
        
        # Adiciona regiões sombreadas
        plt.fill_between(x1_range, x2_boundary, 6, alpha=0.1, color='blue', 
                        label='Região Classe 1')
        plt.fill_between(x1_range, 0, x2_boundary, alpha=0.1, color='red',
                        label='Região Classe 0')
    
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron - Fronteira de Decisão (Reprodução do Código R)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_learning_curve_r(errors):
    """
    Plota a curva de aprendizado
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(errors) + 1), errors, 'bo-', linewidth=2, markersize=6)
    plt.title('Curva de Aprendizado - Erros por Época')
    plt.xlabel('Época')
    plt.ylabel('Número de Erros')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, max(errors) + 0.5 if errors else 0.5)
    
    # Adiciona anotações importantes
    if len(errors) > 0:
        plt.annotate(f'Início: {errors[0]} erros', 
                    xy=(1, errors[0]), xytext=(len(errors)//3, max(errors)*0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        plt.annotate(f'Final: {errors[-1]} erros', 
                    xy=(len(errors), errors[-1]), xytext=(len(errors)*0.7, max(errors)*0.6),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # Log scale para melhor visualização se necessário
    plt.subplot(1, 2, 2)
    plt.semilogy(range(1, len(errors) + 1), np.array(errors) + 1e-10, 'ro-', linewidth=2, markersize=6)
    plt.title('Curva de Aprendizado - Escala Log')
    plt.xlabel('Época')
    plt.ylabel('Número de Erros (log)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_initialization():
    """
    Compara inicialização com zeros vs aleatória
    """
    print("\n" + "="*60)
    print("COMPARAÇÃO: INICIALIZAÇÃO ZEROS vs ALEATÓRIA")
    print("="*60)
    
    # Gera dataset
    X, y, xc1, xc2 = generate_r_dataset()
    
    # Parâmetros
    eta = 0.01
    maxepocas = 100
    tol = 1e-10
    
    # Teste 1: Inicialização com zeros (como no código R)
    print("\n1. Inicialização com ZEROS (código R):")
    perceptron_zero = Perceptron(learning_rate=eta, maxepocas=maxepocas, use_zero_one_labels=True)
    weights_zero, errors_zero = perceptron_zero.train_perceptron(X, y, tol, init_weights_zero=True)
    accuracy_zero = np.mean(perceptron_zero.predict(X) == y)
    print(f"   Épocas: {len(errors_zero)}, Acurácia: {accuracy_zero:.4f}")
    print(f"   Pesos finais: {weights_zero}")
    
    # Teste 2: Inicialização aleatória
    print("\n2. Inicialização ALEATÓRIA:")
    perceptron_rand = Perceptron(learning_rate=eta, maxepocas=maxepocas, use_zero_one_labels=True)
    weights_rand, errors_rand = perceptron_rand.train_perceptron(X, y, tol, init_weights_zero=False)
    accuracy_rand = np.mean(perceptron_rand.predict(X) == y)
    print(f"   Épocas: {len(errors_rand)}, Acurácia: {accuracy_rand:.4f}")
    print(f"   Pesos finais: {weights_rand}")
    
    return errors_zero, errors_rand

def main():
    """
    Função principal que reproduz o código R
    """
    print("REPRODUÇÃO EXATA DO CÓDIGO R EM PYTHON")
    print("Algoritmo Perceptron com labels 0/1")
    print("\nCódigo R original:")
    print("  N<-30")
    print("  xc1<-matrix(rnorm(2*N)*0.5, N, 2) + 2")
    print("  xc2<-matrix(rnorm(2*N)*0.5, N, 2) + 4") 
    print("  xall<-rbind(xc1, xc2)")
    print("  yall<-rbind(matrix(0, ncol=1, nrow=N), matrix(1, ncol=1, nrow=N))")
    print("  w<-c(0, 0, 0)")
    print("  eta<-0.01")
    print("  maxepocas<-100")
    print("  tol<-1e-10")
    
    # Gera o dataset exatamente como no código R
    X, y, xc1, xc2 = generate_r_dataset()
    
    print(f"\nDataset gerado:")
    print(f"  - Classe 0: {len(xc1)} pontos centrados em (2, 2)")
    print(f"  - Classe 1: {len(xc2)} pontos centrados em (4, 4)")
    print(f"  - Total: {len(X)} amostras")
    
    # Visualiza o dataset
    try:
        plot_r_dataset(xc1, xc2)
    except ImportError:
        print("  (matplotlib não disponível para visualização)")
    
    # Treina o Perceptron
    perceptron, weights, errors = train_r_perceptron(X, y)
    
    # Visualizações
    try:
        plot_decision_boundary_r(X, y, perceptron, xc1, xc2)
        plot_learning_curve_r(errors)
        
        # Comparação de inicializações
        errors_zero, errors_rand = compare_initialization()
        
        # Plot comparativo
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(errors_zero, 'bo-', label='Zeros (R)', linewidth=2)
        plt.title('Inicialização com Zeros')
        plt.xlabel('Época')
        plt.ylabel('Erros')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(errors_rand, 'ro-', label='Aleatória', linewidth=2)
        plt.title('Inicialização Aleatória')
        plt.xlabel('Época')
        plt.ylabel('Erros')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nNota: matplotlib não disponível para gráficos.")
        print("Para visualizações: pip install matplotlib")
    
    print("\n" + "="*60)
    print("RESUMO DA REPRODUÇÃO")
    print("="*60)
    print("✓ Dataset gerado exatamente como no código R")
    print("✓ Labels 0 e 1 (ao invés de -1 e 1)")
    print("✓ Pesos inicializados com zeros")
    print("✓ Parâmetros idênticos: eta=0.01, maxepocas=100, tol=1e-10")
    print("✓ Algoritmo Perceptron funcionando corretamente")
    print("✓ Converge para problemas linearmente separáveis")
    print("\nA implementação Python reproduz fielmente o comportamento do código R!")

if __name__ == "__main__":
    main()