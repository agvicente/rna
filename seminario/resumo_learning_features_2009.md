# Learning Multiple Layers of Features from Tiny Images - Alex Krizhevsky (2009)

## Contexto Histórico

### O Estado do Deep Learning em 2009

No final da década de 2000, o campo do aprendizado profundo estava em um momento crucial de transição. Após o "AI Winter" dos anos 1980-1990, o interesse em redes neurais começou a ressurgir, principalmente devido aos trabalhos pioneiros de:

- **Geoffrey Hinton e colaboradores (2006)**: Introduziram as Deep Belief Networks (DBNs) e demonstraram que redes profundas podiam ser treinadas eficientemente usando pré-treinamento não supervisionado layer-by-layer
- **Yann LeCun**: Suas Convolutional Neural Networks (LeNet) já mostravam sucesso em reconhecimento de dígitos
- **Yoshua Bengio**: Contribuições fundamentais para entender os desafios de treinar redes profundas

### Desafios da Época

1. **Problema do Vanishing Gradient**: Redes profundas sofriam com gradientes que desapareciam nas camadas iniciais
2. **Falta de Dados Rotulados**: Datasets grandes e bem rotulados eram raros
3. **Limitações Computacionais**: GPUs ainda não eram amplamente usadas para deep learning
4. **Feature Engineering Manual**: A maioria dos sistemas dependia de características extraídas manualmente

### Importância das RBMs

As Restricted Boltzmann Machines (RBMs) emergiram como uma ferramenta fundamental para:
- Pré-treinamento não supervisionado de redes profundas
- Aprendizado de representações hierárquicas
- Inicialização de pesos de forma inteligente

## Resumo Executivo

Este trabalho de Alex Krizhevsky representa uma contribuição seminal para o campo do aprendizado profundo, focando em três aspectos principais:

1. **Modelagem Generativa de Imagens**: Demonstra como treinar RBMs e DBNs em imagens naturais pequenas
2. **Criação do Dataset CIFAR**: Introduz os datasets CIFAR-10 e CIFAR-100, que se tornaram benchmarks fundamentais
3. **Paralelização de Treinamento**: Desenvolve algoritmos para treinar RBMs em múltiplas máquinas

## Objetivos e Motivação

### Objetivo Principal
Aprender múltiplas camadas de características (features) a partir de imagens pequenas coloridas, superando as limitações de trabalhos anteriores que falharam em extrair features meaningfuls de imagens naturais.

### Motivações Específicas
1. **Superar Falhas Anteriores**: Trabalhos prévios do MIT e NYU falharam em aprender filtros interessantes do dataset "80 million tiny images"
2. **Criar Datasets Confiáveis**: A necessidade de labels confiáveis para experimentos de reconhecimento de objetos
3. **Escalabilidade**: Desenvolver métodos que funcionem com milhões de imagens

## Metodologia

### 1. Pré-processamento: ZCA Whitening

A transformação ZCA (Zero Component Analysis) é fundamental para o sucesso do método:

**Formulação Matemática:**
```
Matriz de Covariância: C = (1/(n-1)) * X * X^T
Decomposição: C = P * D * P^T
Matriz de Whitening: W = (1/√(n-1)) * P * D^(-1/2) * P^T
Dados Transformados: Y = W * X
```

**Propriedades da ZCA:**
- Remove correlações de segunda ordem entre pixels
- Força o modelo a focar em correlações de alta ordem
- Preserva informação de bordas enquanto remove regiões uniformes

### 2. Restricted Boltzmann Machines (RBMs)

#### RBM Binária-Binária
**Função de Energia:**
```
E(v,h) = -∑∑(v_i * h_j * w_ij) - ∑(v_i * b_v_i) - ∑(h_j * b_h_j)
```

**Probabilidades Condicionais:**
```
P(h_j = 1|v) = 1/(1 + exp(-∑(v_i * w_ij) - b_h_j))
P(v_i = 1|h) = 1/(1 + exp(-∑(h_j * w_ij) - b_v_i))
```

#### RBM Gaussiana-Bernoulli
Para dados reais como intensidades de pixels:

**Função de Energia:**
```
E(v,h) = ∑(v_i - b_v_i)²/(2σ_i²) - ∑(b_h_j * h_j) - ∑∑(v_i * h_j * w_ij / σ_i)
```

**Distribuição Condicional:**
```
P(v|h) ~ N(b_v_i + σ_i * ∑(h_j * w_ij), σ_i²)
P(h_j = 1|v) = 1/(1 + exp(-∑(v_i * w_ij / σ_i) - b_h_j))
```

### 3. Treinamento: Contrastive Divergence (CD-1)

**Atualização de Pesos:**
```
Δw_ij = ε * (E_data[v_i * h_j] - E_model[v_i * h_j])
```

Onde:
- E_data: Expectativa quando unidades visíveis são fixadas nos dados
- E_model: Expectativa sob distribuição do modelo (aproximada por CD)

### 4. Deep Belief Networks (DBNs)

**Treinamento Layer-by-Layer:**
1. Treinar primeira RBM nos dados
2. Usar ativações da primeira camada como dados para segunda RBM
3. Repetir para camadas subsequentes
4. Fine-tuning supervisionado opcional

### 5. Estratégias de Treinamento

#### Treinamento em Patches
- Dividir imagens 32×32 em patches 8×8
- Treinar RBMs independentes para cada patch
- Combinar RBMs em uma única rede grande
- Permite redução da complexidade computacional

#### Aprendizado de Variâncias Visíveis
**Regra de Atualização para σ_i:**
```
Δσ_i = ε_σ * (E_data[(v_i - b_v_i)²/σ_i³ - ∑(h_j * w_ij * v_i)/σ_i²] - E_model[...])
```

### 6. Algoritmo de Paralelização

**Divisão do Trabalho:**
- Cada máquina k computa 1/K das unidades ocultas
- Sincronização após cada passo de sampling
- Comunicação mínima para RBMs binárias (apenas bits)

**Custo de Comunicação:**
Para K máquinas: 48 * (K-1) MB por batch para RBMs binárias

## Resultados Principais

### 1. Qualidade dos Filtros Aprendidos

**RBMs treinadas em patches 8×8:**
- Aprenderam filtros de detecção de bordas
- Preferência por filtros coloridos de baixa frequência
- Filtros preto-e-branco de alta frequência

**RBMs treinadas em imagens 32×32 completas:**
- Filtros similares aos de patches
- Comportamento interessante: filtros "pontuais" evoluem para detectores de borda

### 2. Performance em Classificação (CIFAR-10)

| Método | Erro (%) |
|--------|----------|
| Logistic Regression (pixels) | ~40 |
| Logistic Regression (whitened pixels) | ~37 |
| Logistic Regression (RBM features) | ~22 |
| Neural Network (RBM init) | ~18.5 |

### 3. Dataset CIFAR

**CIFAR-10:**
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 6.000 imagens por classe
- 50.000 treino + 10.000 teste

**CIFAR-100:**
- 100 classes organizadas em 20 superclasses
- 600 imagens por classe

### 4. Análise de Escalabilidade

**Paralelização Binary-to-Binary RBM:**
- Speedup quase linear até 8 máquinas
- Comunicação mínima (apenas bits)

**Paralelização Gaussian-to-Binary RBM:**
- Speedup ligeiramente menor devido ao overhead de comunicação
- Ainda eficiente para problemas grandes

## Contribuições Principais

### 1. Técnicas de Pré-processamento
- Demonstra a importância crítica da transformação ZCA
- Remove direções de menor variância para melhor aprendizado

### 2. Metodologia de Treinamento
- Estratégia de patches para escalar para imagens maiores
- Técnicas para aprender variâncias das unidades visíveis
- Procedimento de merge de RBMs treinadas em patches

### 3. Datasets de Referência
- CIFAR-10 e CIFAR-100 se tornaram benchmarks fundamentais
- Metodologia rigorosa de rotulação
- Critérios claros para inclusão de imagens

### 4. Paralelização Eficiente
- Algoritmo para distribuir treinamento de RBMs
- Análise teórica dos custos de comunicação
- Implementação prática com TCP sockets

### 5. Análise de Segundas Camadas
- Demonstra que camadas superiores aprendem features de nível mais alto
- Visualização de como features similares se agrupam

## Limitações e Desafios

### 1. Limitações Técnicas
- RBMs ainda produzem alguns filtros "ruidosos"
- Dependência crítica de hiperparâmetros (learning rates)
- Treinamento computacionalmente intensivo

### 2. Limitações dos Dados
- Imagens 32×32 são muito pequenas para objetos complexos
- Classes do CIFAR ainda são relativamente simples
- Datasets pequenos para padrões atuais

### 3. Limitações dos Métodos
- CD-1 é apenas uma aproximação
- Dificuldade em avaliar qualidade do modelo generativo
- Synchronization overhead na paralelização

## Impacto e Legado

Este trabalho foi fundamental para:

1. **Estabelecer CIFAR como benchmark**: CIFAR-10 ainda é usado extensivamente
2. **Demonstrar viabilidade de deep learning**: Provou que redes profundas podiam aprender features úteis
3. **Influenciar AlexNet (2012)**: Krizhevsky posteriormente revolucionou computer vision
4. **Estabelecer práticas de pré-processamento**: ZCA whitening tornou-se técnica padrão
5. **Computação Distribuída**: Pioneiro em paralelização de deep learning

## Ferramentas e Bibliotecas Modernas

### Implementações de RBMs

#### Python
```python
# scikit-learn (limitado)
from sklearn.neural_network import BernoulliRBM

# Implementações dedicadas
import rbm  # biblioteca dedicada
```

#### TensorFlow/Keras
```python
# Implementação customizada de RBM
import tensorflow as tf
import tensorflow_probability as tfp
```

#### PyTorch
```python
# Implementações community
import torch
import torch.nn.functional as F
```

### Bibliotecas Especializadas Modernas

1. **scikit-learn**: Implementação básica de BernoulliRBM
2. **TensorFlow Probability**: Ferramentas para modelos probabilísticos
3. **PyTorch**: Flexibilidade para implementações customizadas
4. **JAX**: Para computação científica e automatização
5. **Theano** (descontinuado): Era popular para RBMs

### Frameworks de Deep Learning Distribuído

1. **Horovod**: Treinamento distribuído moderno
2. **Ray**: Computação distribuída em Python
3. **TensorFlow Distributed**: Paralelização nativa
4. **PyTorch Distributed**: DDP (DistributedDataParallel)

### Datasets e Benchmarks

1. **CIFAR-10/100**: Ainda amplamente usado
2. **ImageNet**: Sucessor natural para imagens maiores
3. **MNIST**: Para experimentos simples
4. **Fashion-MNIST**: Alternativa moderna ao MNIST

## Como Aprofundar no Tema

### 1. Fundamentos Teóricos

#### Livros Recomendados
- "Deep Learning" - Goodfellow, Bengio, Courville
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "Elements of Statistical Learning" - Hastie, Tibshirani, Friedman

#### Papers Clássicos
- Hinton, G. (2002) "Training Products of Experts by Minimizing Contrastive Divergence"
- Hinton, G. (2006) "A Fast Learning Algorithm for Deep Belief Nets"
- Bengio, Y. (2009) "Learning Deep Architectures for AI"

### 2. Implementação Prática

#### Código de Exemplo - RBM Simples
```python
import numpy as np
import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        
    def sample_from_p(self, p):
        return torch.bernoulli(p)
    
    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h
    
    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v
```

#### ZCA Whitening Implementation
```python
def zca_whitening(X, epsilon=1e-5):
    """
    Aplica ZCA whitening nos dados
    X: dados de entrada (samples × features)
    """
    # Calcula matriz de covariância
    cov = np.cov(X.T)
    
    # Decomposição eigenvalue
    U, S, V = np.linalg.svd(cov)
    
    # Constrói matriz de whitening
    W = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    
    # Aplica transformação
    X_white = np.dot(X, W)
    return X_white, W
```

### 3. Projetos Práticos

#### Projeto 1: RBM para MNIST
1. Implementar RBM básica
2. Treinar em MNIST
3. Visualizar filtros aprendidos
4. Comparar com PCA

#### Projeto 2: Reproduzir Experimentos CIFAR-10
1. Implementar pipeline completo
2. ZCA whitening
3. RBM treinamento
4. Classificação com features aprendidas

#### Projeto 3: Paralelização Moderna
1. Implementar versão distribuída com Ray/Horovod
2. Comparar com versão serial
3. Analisar speedup

### 4. Tópicos Avançados

#### Desenvolvimentos Posteriores
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Transformer architectures
- Self-supervised learning

#### Conexões Modernas
- Contrastive learning (SimCLR, MoCo)
- Energy-based models revival
- Neural ODE e continuous normalizing flows

## Exercícios Propostos

### Exercício 1: Implementação Básica
Implemente uma RBM binária-binária do zero usando apenas NumPy. Treine em um dataset simples e visualize os filtros aprendidos.

### Exercício 2: Análise de Hiperparâmetros
Investigue o impacto de diferentes learning rates, número de unidades ocultas e épocas de treinamento na qualidade dos filtros.

### Exercício 3: Comparação de Pré-processamento
Compare o impacto de diferentes técnicas de pré-processamento: raw pixels, PCA whitening, ZCA whitening.

### Exercício 4: Implementação Distribuída
Implemente uma versão simplificada do algoritmo de paralelização usando multiprocessing em Python.

## Referências

### Artigo Original
- Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images". Master's thesis, University of Toronto.

### Bibliografia Fundamental

#### Trabalhos Precursores
- Hinton, G. E. (2002). "Training Products of Experts by Minimizing Contrastive Divergence". Neural Computation, 14(8), 1771-1800.
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). "A Fast Learning Algorithm for Deep Belief Nets". Neural Computation, 18(7), 1527-1554.
- Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks". Science, 313(5786), 504-507.

#### Contexto Histórico
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition". Proceedings of the IEEE, 86(11), 2278-2324.
- Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). "Greedy layer-wise training of deep networks". Advances in neural information processing systems, 19, 153.

#### Trabalhos Subsequentes
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks". Advances in neural information processing systems, 25, 1097-1105.

### Recursos Online

#### Implementações e Tutoriais
- Hinton's Coursera Course: "Neural Networks for Machine Learning"
- Deep Learning Specialization (Coursera): https://www.coursera.org/specializations/deep-learning
- Fast.ai Deep Learning Course: https://www.fast.ai/

#### Datasets
- CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
- Tiny Images Dataset: https://groups.csail.mit.edu/vision/TinyImages/

#### Bibliotecas
- TensorFlow: https://tensorflow.org/
- PyTorch: https://pytorch.org/
- scikit-learn: https://scikit-learn.org/
- JAX: https://github.com/google/jax

#### Implementações de RBM
- sklearn.neural_network.BernoulliRBM
- TensorFlow Probability: https://www.tensorflow.org/probability
- PyTorch RBM implementations (GitHub community)

### Artigos de Acompanhamento

#### Surveys e Reviews
- Bengio, Y. (2009). "Learning Deep Architectures for AI". Foundations and trends in Machine Learning, 2(1), 1-127.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning". Nature, 521(7553), 436-444.

#### Desenvolvimentos Modernos
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep learning". MIT press.
- Kingma, D. P., & Welling, M. (2013). "Auto-encoding variational bayes". arXiv preprint arXiv:1312.6114.

## Desenvolvimentos Modernos Relacionados

### Energy-Based Models Revival (2020+)
O interesse em modelos baseados em energia ressurgiu recentemente:
- **EBGAN (Energy-Based GANs)**: Combinam conceitos de RBMs com GANs
- **JEM (Joint Energy-based Model)**: Modelos que fazem classificação e geração simultaneamente
- **Score-Based Generative Models**: Usar gradientes de energia para geração

### Contrastive Learning Moderno
Evolução direta dos conceitos de Contrastive Divergence:
- **SimCLR, MoCo**: Self-supervised learning para visão computacional
- **CLIP**: Aprendizado contrastivo multimodal (imagem-texto)
- **SwAV**: Contrastive learning com clustering

### Arquiteturas Modernas Influenciadas
- **Vision Transformers (ViTs)**: Hierarquias de patches similar às abordagens de Krizhevsky
- **ResNets**: Inspiração nas conexões skip das DBNs
- **Normalização em Batch**: Evolução das técnicas de whitening

### Ferramentas de Paralelização Atuais
- **Horovod**: Sucessor direto das técnicas de paralelização apresentadas
- **FairScale**: Ferramentas de Meta para treinamento distribuído
- **DeepSpeed**: Microsoft's framework para modelos massivos
- **Ray**: Computação distribuída moderna

---

**Nota**: Este resumo foi elaborado para uma apresentação de pós-graduação, mantendo rigor científico e matemático. As implementações fornecidas são exemplos didáticos e podem precisar de otimizações para uso em produção.

**Última Atualização**: Janeiro 2024

**Importância Histórica**: Este trabalho de 2009 foi fundamental para estabelecer as bases do que viria a ser a revolução do deep learning da década de 2010, culminando com o próprio AlexNet de Krizhevsky em 2012 que revolucionou computer vision.
