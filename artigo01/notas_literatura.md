# Notas de Literatura - Classificador de Margem Máxima Baseado em Mediatrizes

## Resumo do Contexto

Este documento compila as principais referências bibliográficas relevantes para o desenvolvimento de um classificador linear baseado em mediatrizes para problemas linearmente separáveis, com foco em maximização de margem e comparação com métodos clássicos (Perceptron, Regressão Logística, SVM).

---

## 1. Fundamentação Teórica - Classificadores Lineares

### 1.1 Support Vector Machines (Referência Fundamental)

**[129] Cortes & Vapnik (1995). "Support-Vector Networks"**
- **Relevância**: Trabalho seminal sobre SVMs e maximização de margem
- **Conceito-chave**: Formalização matemática da margem máxima como critério de otimização
- **Aplicação ao trabalho**: Benchmark teórico para comparação; SVMs resolvem otimização quadrática complexa, enquanto nossa abordagem propõe solução geométrica direta
- **Citação obrigatória**: Base teórica de margem máxima

### 1.2 Perceptron (Baseline Clássico)

**[b2] Rosenblatt (1958). "The perceptron: A probabilistic model..."**
- **Relevância**: Algoritmo iterativo clássico para classificação linear
- **Limitação**: Não maximiza margem; qualquer separador que classifica corretamente é aceito
- **Aplicação ao trabalho**: Baseline de comparação direto; contraste entre aprendizado iterativo vs. solução geométrica

### 1.3 Fundamentos de Classificação Linear

**[b1] Hastie, Tibshirani & Friedman (2009). "The Elements of Statistical Learning"**
- **Relevância**: Referência fundamental para teoria de classificação
- **Conceitos**: Separabilidade linear, bias-variance tradeoff, generalização
- **Aplicação ao trabalho**: Contextualização teórica geral

**[b4] Bishop (2006). "Pattern Recognition and Machine Learning"**
- **Relevância**: Fundamentos de reconhecimento de padrões
- **Conceitos**: Classificadores lineares, fronteiras de decisão, teoria bayesiana
- **Aplicação ao trabalho**: Fundamentação teórica complementar

---

## 2. Métodos Geométricos e Nearest-Point Approaches

### 2.1 Algoritmo Schlesinger-Kozinec (SK) - Mais Próximo Conceitualmente

**[3] Leng Qiangku (2013). "Construction of Multiconlitron Using SK Algorithm"**
- **Relevância**: ⭐ ALTA - Abordagem mais próxima da proposta
- **Conceito-chave**: Usa algoritmo SK para encontrar hiperplano separador entre convex polytopes via nearest opposite-class points
- **Diferenças com nossa proposta**:
  - SK opera em convex hulls (conjuntos), não em pares individuais
  - Gera classificadores piecewise linear (multiconlitron)
  - Não foca em margem máxima explícita
  - Não tem orientação para hardware/quantização
- **Contribuição**: Valida a viabilidade de métodos baseados em nearest-point geometry
- **Uso no artigo**: 
  - Citar como antecedente algorítmico
  - Contrastar: "enquanto SK opera em convex hulls, nossa abordagem usa pares individuais..."
  - Destacar que SK não maximiza margem explicitamente

**Citação do relatório Undermind:**
> "The closest algorithmic antecedent is the Schlesinger–Kozinec nearest-point separator for convex polytopes, used to construct conlitron/multiconlitron classifiers for convexly or linearly separable data"

### 2.2 Multiconlitron Framework

**[99] Li et al. (2011). "Multiconlitron: A General Piecewise Linear Classifier"**
- **Relevância**: Extensão do conceito de conlitron
- **Conceito**: Framework geral para classificadores piecewise linear
- **Aplicação ao trabalho**: Contextualização de métodos geométricos alternativos

**[83] Leng Qiang (2013). "A Soft Margin Method for Multiconlitron Design"**
- **Relevância**: Extensão com soft margin
- **Conceito**: Incorpora margem suave ao multiconlitron
- **Diferença**: Ainda usa convex hulls, não mediatrizes de pares individuais

---

## 3. Implementações Hardware-Efficient (Motivação Prática)

### 3.1 Chipclas - Classificador para Circuitos Integrados

**[b8] Torres et al. (2015). "Distance-based large margin classifier suitable for integrated circuit implementation"**
- **Relevância**: ⭐ ALTA - Mencionado explicitamente no exercício
- **Conceito-chave**: Classificador baseado em distâncias adequado para implementação em CI
- **Abordagem**: Usa grafos para determinar pares de pontos próximos; mediatrizes definidas apenas entre amostras próximas da região de separação
- **Motivação**: Eficiência em hardware (operações simples)
- **Aplicação ao trabalho**: 
  - Inspiração para estratégia de pruning (considerar apenas pontos próximos)
  - Justificativa de simplicidade computacional vs. SVM
  - Argumento de viabilidade em hardware

**Citação do exercício:**
> "Métodos como o Chipclas consideram uma abordagem semelhante, porém, os pares de pontos que definem os separadores são determinados por um grafo cujas arestas [...] existem somente entre amostras próximas no espaço"

### 3.2 In-Memory Computing para ML

**[8] Zhang et al. (2017). "In-Memory Computation of a Machine-Learning Classifier in a Standard 6T SRAM Array"**
- **Relevância**: Moderada - Fundação de implementação em hardware
- **Conceito**: Computação in-memory para classificadores lineares
- **Aplicação ao trabalho**: Discussão sobre implementação futura; vantagens de operações simples

**[5] Gonugondla et al. (2018). "A Variation-Tolerant In-Memory Machine Learning Classifier via On-Chip Training"**
- **Relevância**: Moderada
- **Conceito**: SVM in-memory com treinamento on-chip para robustez
- **Contraste**: SVM requer otimização complexa; mediatrizes são diretas
- **Métricas**: 42 pJ/decision, 3.12 TOPS/W
- **Aplicação ao trabalho**: Discussão sobre trade-offs hardware vs. acurácia

---

## 4. Teoria de Margem e Precisão

### 4.1 Análise de Precisão para Classificadores de Margem

**[6] Sakr et al. (2019). "Minimum Precision Requirements of General Margin Hyperplane Classifiers"**
- **Relevância**: Moderada a Alta
- **Conceito-chave**: Bounds analíticos sobre precisão mínima para margin hyperplanes
- **Contribuição**: Esquema para trade-off entre precisão de entrada e pesos
- **Aplicação ao trabalho**: 
  - Discussão sobre quantização e robustez
  - Argumentar que mediatrizes têm equações simples (menos sensíveis a precisão?)
  - Análise de margem quantizada

**[36] Minimum precision requirements for the SVM-SGD learning algorithm**
- **Relevância**: Complementar
- **Conceito**: Precisão para SVM-SGD
- **Aplicação**: Comparação de requisitos de precisão

---

## 5. Classificadores Lineares e Comparações

### 5.1 Regressão Logística

**Referência geral (livros-texto)**
- **Hastie et al. (2009)** - Seção sobre logistic regression
- **Bishop (2006)** - Capítulo sobre probabilistic classification
- **Conceito**: Abordagem probabilística com função sigmoide
- **Treinamento**: Gradient descent ou Newton-Raphson
- **Aplicação ao trabalho**: Baseline de comparação

### 5.2 Comparações Entre Métodos

**[64] Huang & Lin (2016). "Linear and Kernel Classification: When to Use Which?"**
- **Relevância**: Moderada
- **Conceito**: Estudo sobre quando usar classificadores lineares vs. kernel
- **Aplicação ao trabalho**: Discussão sobre quando métodos simples são suficientes

---

## 6. Conceitos Matemáticos Fundamentais

### 6.1 Mediatriz (Perpendicular Bisector)

**Definição matemática:**
- Para dois pontos **xᵢ** e **xⱼ**:
  - Ponto médio: **m** = (xᵢ + xⱼ)/2
  - Vetor normal: **w** = xᵢ - xⱼ
  - Bias: b = -wᵀm
- **Propriedade**: Todos os pontos na mediatriz estão equidistantes de xᵢ e xⱼ

**Referência conceitual:**
- Geometria analítica básica (não requer citação específica)
- Pode-se mencionar como "geometria euclidiana elementar"

### 6.2 Margem de Separação

**Definição (do exercício):**
> "a margem de separação de um separador linear é definida como a média das distâncias dos pontos mais próximos de cada classe ao separador"

**Fórmula:**
```
margem(w, b) = min_{i=1,...,N} |wᵀxᵢ + b| / ||w||
```

**Referência fundamental:**
- Vapnik & Cortes (1995) - SVM paper
- Hastie et al. (2009) - capítulo sobre margin classifiers

---

## 7. Algoritmos de Otimização (Para Contraste)

### 7.1 Sequential Minimal Optimization (SMO)

**Conceito**: Algoritmo de otimização para SVMs
- **Complexidade**: O(n²) a O(n³)
- **Contraste com mediatrizes**: Mediatrizes evitam otimização iterativa

### 7.2 Stochastic Gradient Descent (SGD)

**Aplicação em**:
- Perceptron
- Regressão Logística
- SVM online

**Contraste**: Métodos iterativos vs. solução direta geométrica

---

## 8. Datasets e Benchmarks

### 8.1 Datasets Padrão para Classificação

**MNIST**
- **Uso na literatura**: [6], [9], [121]
- **Características**: 28×28 imagens, 10 classes
- **Relevância**: Não diretamente aplicável (não é binário linearmente separável)

**Iris**
- **Uso**: Dataset clássico de classificação
- **Características**: 4 features, 3 classes
- **Aplicação ao trabalho**: 
  - Iris (Setosa vs. resto) - linearmente separável
  - Perfeito para validação do método

**Breast Cancer Wisconsin**
- **Características**: ~30 features, 2 classes
- **Aplicação**: Dataset médico real para validação

**Wine**
- **Características**: 13 features, 3 classes
- **Aplicação**: Configurações binárias linearmente separáveis

---

## 9. Lacunas Identificadas (Gap Analysis)

### Do Relatório Undermind:

**Lacuna Principal:**
> "No paper in this set implements a nearest-opposite-pair perpendicular-bisector linear classifier [...] with quantization-robust training and full baseline comparisons"

**Elementos Faltantes:**
1. ❌ Nenhum trabalho define w,b a partir da mediatriz do par mais próximo de classes opostas
2. ❌ Nenhum usa seleção quantizada de margem consciente de par
3. ❌ Nenhum fornece comparações unificadas SVM/perceptron/logistic sob quantização

**Oportunidade Identificada:**
> "Combining SK-style nearest-element geometry as motivation with [...] precision theory can yield a hardware-efficient, quantization-robust bisector classifier with minimal training overhead"

---

## 10. Estrutura de Citações por Seção do Artigo

### Introdução
- **Classificação binária fundamental**: [b1], [b4]
- **Perceptron e métodos clássicos**: [b2]
- **SVM e margem máxima**: [129]
- **Motivação hardware**: [b8]

### Revisão de Literatura

#### Classificadores Lineares Clássicos
- Perceptron: [b2]
- Regressão Logística: [b1]
- SVM: [129]

#### Métodos Geométricos
- SK Algorithm: [3]
- Multiconlitron: [99], [83]
- Chipclas: [b8] ⭐ OBRIGATÓRIO (citado no exercício)

#### Margem Máxima
- SVM: [129]
- Teoria de margem: [48], [130]
- Precisão e margem: [6], [36]

### Metodologia
- Definição de mediatriz: geometria analítica básica
- Cálculo de margem: baseado em [129], [b1]

### Discussão
- Implementação em hardware: [b8], [8], [5]
- Precisão e quantização: [6]
- Comparações: [64]

---

## 11. Argumentos-Chave para o Artigo

### Por que Mediatrizes?

1. **Solução Analítica Direta**
   - Não requer otimização iterativa (vs. SVM)
   - Não requer convergência (vs. Perceptron)
   - Equações explícitas: w = xᵢ - xⱼ, b = -wᵀm

2. **Simplicidade Computacional**
   - Operações: soma, subtração, produto escalar
   - vs. SVM: otimização quadrática
   - vs. Perceptron: múltiplas iterações

3. **Viabilidade em Hardware**
   - Inspirado em Chipclas [b8]
   - Operações aritméticas básicas
   - Adequado para in-memory computing [8]

4. **Interpretabilidade**
   - Pontos de suporte explícitos (par definidor)
   - Geometria intuitiva
   - Fácil visualização (2D)

### Limitações Reconhecidas

1. **Escalabilidade**: O(N₁ × N₂) candidatas
   - Mitigação: pruning inspirado em Chipclas [b8]
   
2. **Apenas Linear**: Restrito a problemas linearmente separáveis
   - Reconhecimento honesto da limitação
   
3. **Sensibilidade a Outliers**: Qualquer par pode definir mediatriz
   - Discussão sobre robustez

---

## 12. Referências Bibliográficas Formatadas (IEEE)

### Referências Obrigatórias

```
[b1] T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of 
     Statistical Learning: Data Mining, Inference, and Prediction," 
     2nd ed., Springer, 2009.

[b2] F. Rosenblatt, "The perceptron: A probabilistic model for 
     information storage and organization in the brain," 
     Psychological Review, vol. 65, no. 6, pp. 386–408, 1958.

[b3] C. Cortes and V. Vapnik, "Support-vector networks," 
     Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.

[b4] C. M. Bishop, "Pattern Recognition and Machine Learning," 
     Springer, 2006.

[b8] L. C. B. Torres, et al., "Distance-based large margin classifier 
     suitable for integrated circuit implementation," Electronics 
     Letters, vol. 51, no. 24, pp. 1967–1969, 2015.
```

### Referências Adicionais Recomendadas

```
[3]  L. Qiangku, "Construction of Multiconlitron Using SK Algorithm," 
     Journal of Frontiers of Computer Science and Technology, 2013.

[6]  C. Sakr, Y. Kim, and N. R. Shanbhag, "Minimum Precision 
     Requirements of General Margin Hyperplane Classifiers," IEEE 
     Journal on Emerging and Selected Topics in Circuits and 
     Systems, 2019.

[8]  J. Zhang, et al., "In-Memory Computation of a Machine-Learning 
     Classifier in a Standard 6T SRAM Array," IEEE Journal of 
     Solid-State Circuits, 2017.

[99] Y. Li, et al., "Multiconlitron: A General Piecewise Linear 
     Classifier," IEEE Transactions on Neural Networks, 2011.
```

---

## 13. Frases-Chave para Uso no Artigo

### Para Introdução

> "Enquanto SVMs maximizam a margem através de otimização quadrática complexa [b3], métodos geométricos baseados em mediatrizes oferecem uma alternativa analiticamente direta."

> "Inspirado por abordagens hardware-efficient como o Chipclas [b8], propomos um classificador que calcula separadores diretamente da geometria dos dados."

### Para Revisão de Literatura

> "O algoritmo Schlesinger-Kozinec [3] representa o antecedente algorítmico mais próximo, operando com nearest-points entre convex polytopes para construir classificadores piecewise linear."

> "Diferentemente de [3], que opera em convex hulls, nossa abordagem utiliza pares individuais de pontos para definir mediatrizes, resultando em um único hiperplano com margem maximizada."

### Para Discussão

> "A simplicidade das operações envolvidas (soma, subtração, produto escalar) torna o método adequado para implementação em circuitos integrados, seguindo a filosofia de [b8]."

> "Enquanto SVMs requerem O(n²) a O(n³) operações para otimização [b3], nossa abordagem avalia O(N₁×N₂) candidatas com operações elementares."

---

## 14. Checklist de Citações por Seção

### ✅ Introdução
- [ ] Classificação binária fundamental
- [ ] Perceptron e Regressão Logística
- [ ] SVM e margem máxima
- [ ] Motivação hardware (Chipclas)

### ✅ Revisão de Literatura
- [ ] Perceptron [b2]
- [ ] Regressão Logística [b1]
- [ ] SVM [b3]
- [ ] SK Algorithm [3]
- [ ] Chipclas [b8] ⭐
- [ ] Teoria de precisão [6]

### ✅ Metodologia
- [ ] Definição de margem (baseada em SVM)
- [ ] Geometria de mediatrizes

### ✅ Discussão
- [ ] Comparação de complexidade
- [ ] Hardware implementation [b8], [8]
- [ ] Limitações e trabalhos futuros

---

## 15. Notas Finais

### Prioridades de Citação

**ALTA (Obrigatórias)**
1. Torres et al. [b8] - Chipclas (mencionado no exercício)
2. Vapnik & Cortes [b3] - SVM (referência de margem máxima)
3. Rosenblatt [b2] - Perceptron (baseline)
4. Hastie et al. [b1] - Fundamentos

**MÉDIA (Recomendadas)**
5. Leng Qiangku [3] - SK Algorithm (nearest-point method)
6. Sakr et al. [6] - Precisão em margin classifiers
7. Zhang et al. [8] - In-memory computing

**BAIXA (Opcionais)**
8. Multiconlitron papers [99], [83]
9. Outras implementações hardware

### Estratégia de Escrita

1. **Introdução**: Contextualizar com [b1], [b2], [b3], motivar com [b8]
2. **Revisão**: Cobrir clássicos ([b2], [b3]), métodos geométricos ([3]), hardware ([b8])
3. **Metodologia**: Foco no método proposto, referências mínimas
4. **Discussão**: Comparações detalhadas, citar precisão [6], hardware [8]

### Tempo Estimado de Escrita

- Introdução com citações: 15 min
- Revisão de Literatura: 20 min
- Ajustes e formatação: 10 min
- **Total**: ~45 min

---

## 16. Template de Parágrafo com Citações

```latex
% Exemplo de parágrafo bem citado para Revisão de Literatura

O Perceptron, proposto por Rosenblatt \cite{b2}, é um dos primeiros 
algoritmos de aprendizado de máquina capaz de encontrar um separador 
linear para dados linearmente separáveis. Apesar de sua simplicidade, 
o algoritmo não garante a maximização da margem de separação. Esta 
limitação foi abordada pelas Support Vector Machines \cite{b3}, que 
formulam o problema de classificação como otimização da margem através 
de programação quadrática. Contudo, a complexidade computacional de 
SVMs motivou o desenvolvimento de alternativas geometricamente mais 
simples. Torres et al. \cite{b8} propuseram o Chipclas, um classificador 
baseado em distâncias adequado para implementação em circuitos integrados, 
que utiliza grafos para determinar pares de pontos próximos à região de 
separação. De forma relacionada, o algoritmo Schlesinger-Kozinec \cite{3} 
constrói separadores através de nearest-points entre convex polytopes, 
representando um antecedente algorítmico próximo à abordagem de mediatrizes.
```

---

**Documento preparado para suportar a escrita da seção de Revisão de Literatura**
**Data**: Outubro 2025
**Versão**: 1.0

