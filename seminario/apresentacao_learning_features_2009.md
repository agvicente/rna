# Learning Multiple Layers of Features from Tiny Images
## Alex Krizhevsky (2009) - Apresentação para Banca

---

## Slide 1: Título
**Learning Multiple Layers of Features from Tiny Images**
- **Autor**: Alex Krizhevsky
- **Ano**: 2009
- **Instituição**: University of Toronto
- **Orientador**: Geoffrey Hinton

---

## Slide 2: Contexto Histórico - Estado do Deep Learning (2009)

### O Momento Crítico
- **"AI Winter"** dos anos 1980-1990 estava terminando
- **Ressurgimento das Redes Neurais** após trabalhos de Hinton (2006)
- **Breakthrough das DBNs**: Pré-treinamento layer-by-layer não supervisionado

### Desafios da Época
- ❌ **Vanishing Gradient Problem**
- ❌ **Falta de datasets rotulados grandes**
- ❌ **Limitações computacionais** (GPUs não eram padrão)
- ❌ **Dependência de feature engineering manual**

---

## Slide 3: Trabalhos Precursores

### Fundações Estabelecidas
- **Geoffrey Hinton (2006)**: Deep Belief Networks
- **Yann LeCun**: Convolutional Neural Networks (LeNet)
- **Yoshua Bengio**: Fundamentos teóricos do deep learning

### O Problema Central
> Como treinar redes neurais profundas de forma eficiente para extrair features hierárquicas de imagens naturais?

---

## Slide 4: Motivação do Trabalho

### Falhas Anteriores
- **MIT/NYU**: Tentativas fracassaram em extrair features do dataset "80 million tiny images"
- **Filtros sem sentido**: Modelos anteriores aprendiam apenas ruído

### Objetivo Principal
**Aprender múltiplas camadas de características (features) a partir de imagens pequenas coloridas**

### Contribuições Esperadas
1. Modelagem generativa eficaz de imagens
2. Datasets confiáveis para benchmarking
3. Algoritmos de paralelização para escalabilidade

---

## Slide 5: Propriedades dos Dados - Matriz de Covariância

![Covariance Matrix](covariance_matrix_image)

### Estrutura das Imagens Naturais
**Imagem**: Matriz de covariância do dataset tiny images mostrando correlações entre pixels

**Características Observadas**:
- Pixels próximos são **fortemente correlacionados**
- Pixels distantes são **fracamente correlacionados**
- **Simetria** nas imagens (horizontal/vertical)
- **Estrutura por canais** de cor (RGB separados)

---

## Slide 6: ZCA Whitening - Motivação Teórica

### Por que Whitening?
- Remove **correlações de segunda ordem** entre pixels
- Força o modelo a focar em **correlações de alta ordem**
- Melhora significativamente o aprendizado de features

### Formulação Matemática
```
Matriz de Covariância: C = (1/(n-1)) * X * X^T
Decomposição: C = P * D * P^T
Matriz de Whitening: W = (1/√(n-1)) * P * D^(-1/2) * P^T
Dados Transformados: Y = W * X
```

---

## Slide 7: Filtros de Whitening e Dewhitening

![Whitening Filters](whitening_filters_image)

**Filtros de Whitening**: Transformam dados para remover correlações
- **Localização**: Filtros são altamente locais
- **Canais**: Separação por componentes RGB
- **Simetria**: Reflete propriedades das imagens naturais

![Dewhitening Filters](dewhitening_filters_image)

**Filtros de Dewhitening**: Permitem reconstrução dos dados originais

---

## Slide 8: Resultado do ZCA Whitening

![Original vs Whitened](original_whitened_image)

### Efeito da Transformação
- **Preserva informação de bordas**
- **Remove regiões uniformes**
- **Destaca estruturas importantes**
- **Separação clara por canais de cor**

---

## Slide 9: Restricted Boltzmann Machines (RBMs)

![RBM Architecture](rbm_architecture_image)

### Arquitetura
- **Unidades Visíveis**: Dados de entrada
- **Unidades Ocultas**: Features aprendidas
- **Conexões bidirecionais**: Sem conexões intra-camada

### Função de Energia (RBM Binária)

```
E(v,h) = -∑∑(v_i * h_j * w_ij) - ∑(v_i * b_v_i) - ∑(h_j * b_h_j)
```

**Onde**:
- `v`: estado das unidades visíveis
- `h`: estado das unidades ocultas  
- `w_ij`: peso entre unidade visível i e oculta j
- `b_v_i`, `b_h_j`: biases das unidades

---

## Slide 10: RBM Gaussiana-Bernoulli

### Para Dados Reais (Intensidades de Pixels)

```
E(v,h) = ∑(v_i - b_v_i)²/(2σ_i²) - ∑(b_h_j * h_j) - ∑∑(v_i * h_j * w_ij / σ_i)
```

### Distribuições Condicionais

**Unidades Visíveis** (Gaussianas):
```
P(v|h) ~ N(b_v_i + σ_i * ∑(h_j * w_ij), σ_i²)
```

**Unidades Ocultas** (Bernoulli):
```
P(h_j = 1|v) = 1/(1 + exp(-∑(v_i * w_ij / σ_i) - b_h_j))
```

---

## Slide 11: Contrastive Divergence (CD-1)

![CD-N Procedure](cd_n_procedure_image)

### Algoritmo de Treinamento
**Procedimento CD-N**: Aproxima a expectativa do modelo através de sampling

### Atualização de Pesos
```
Δw_ij = ε * (E_data[v_i * h_j] - E_model[v_i * h_j])
```

**Onde**:
- `E_data`: Expectativa quando visíveis fixadas nos dados
- `E_model`: Expectativa sob distribuição do modelo (aproximada)
- `ε`: Taxa de aprendizado

---

## Slide 12: Deep Belief Networks (DBNs)

![DBN Architecture](dbn_architecture_image)

### Treinamento Layer-by-Layer
1. **Primeira RBM**: Treina nos dados originais
2. **Segunda RBM**: Treina nas ativações da primeira
3. **Camadas Subsequentes**: Processo iterativo
4. **Fine-tuning**: Ajuste supervisionado opcional

### Vantagem Teórica
- **Inicialização inteligente** de pesos
- **Extração hierárquica** de features
- **Melhor convergência** comparado a treinamento aleatório

---

## Slide 13: Primeiras Tentativas - Filtros Sem Sentido

![Meaningless Filters](meaningless_filters_image)

### Problema Inicial
**Filtros Aprendidos**: RBM treinada em dados whitened produz filtros sem significado

### Causa do Problema
- **Alto ruído** de alta frequência
- **Correlações complexas** não capturadas
- **Necessidade de estratégias** mais sofisticadas

---

## Slide 14: Análise Espectral - Remoção de Componentes

![Log Eigenspectrum](log_eigenspectrum_image)

### Estratégia de Preprocessamento
**Espectro de Eigenvalues**: Variância nas 1000 componentes menos significativas é várias ordens de magnitude menor

### Solução Implementada
- **Remoção das 1000 componentes** menos significativas
- **Preservação da informação** importante
- **Redução do ruído** de alta frequência

---

## Slide 15: Estratégia de Patches

![Segmenting Image](segmenting_patches_image)

### Divisão em Patches 8×8
**Motivação**: Reduzir complexidade computacional e dimensional

### Abordagem
- **25 patches não-overlapping** de 8×8 pixels
- **1 patch global** subsampled da imagem completa
- **26 RBMs independentes** treinadas
- **Merge posterior** em uma única rede

---

## Slide 16: Resultados - Filtros de Qualidade

![Filters on 8x8 Patch](filters_8x8_patch_image)

### Filtros Aprendidos (Patch 8×8)
**Detectores de Borda**: RBM conseguiu aprender filtros significativos
- **Preferência**: Filtros coloridos de baixa frequência
- **Contraste**: Filtros preto-e-branco de alta frequência
- **Interpretação**: Informação de posição precisa + informação de cor aproximada

---

## Slide 17: Filtros Subsampled

![Filters Subsampled](filters_subsampled_image)

### RBM Treinada em Versões Subsampled
**Características**: Filtros mais suaves e globais
- **Escala diferente** de detecção
- **Informação complementar** aos patches locais
- **Base para merge** com patches locais

---

## Slide 18: Merge de RBMs

![Converting Hidden Units](converting_hidden_units_image)

### Procedimento de Combinação
**Conversão**: Como converter unidades treinadas em patch para imagem completa

### Estratégia
- **Duplicação de pesos** (16 vezes para patch global)
- **Divisão por fator** correspondente
- **Inicialização a zero** para conexões não existentes
- **Untied weights**: Liberdade para diferenciação

---

## Slide 19: Resultados de Classificação

### Performance no CIFAR-10

| Método | Erro (%) |
|--------|----------|
| Logistic Regression (raw pixels) | ~40 |
| Logistic Regression (whitened pixels) | ~37 |
| Logistic Regression (RBM features) | ~22 |
| **Neural Network (RBM init)** | **~18.5** |

### Análise
- **Features RBM** significativamente melhores que pixels
- **Inicialização RBM** melhora performance da rede neural
- **Dados não-whitened** performers melhor que whitened para RBMs

---

## Slide 20: Matriz de Confusão

![Confusion Matrix](confusion_matrix_image)

### Padrões de Classificação
**Observações**:
- **Clustering animal vs não-animal**: Separação clara entre categorias biológicas
- **Confusão cat-dog**: Alta confusão entre classes similares
- **Bird-plane**: Confusão ocasional interessante
- **Estrutura hierárquica**: Modelo captura relações semânticas

---

## Slide 21: Dataset CIFAR - Contribuição Duradoura

### CIFAR-10
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **60.000 imagens**: 50.000 treino + 10.000 teste
- **Metodologia rigorosa** de rotulação

### CIFAR-100
- **100 classes** organizadas em 20 superclasses
- **600 imagens** por classe
- **Estrutura hierárquica** para estudos avançados

### Impacto
- **Benchmark fundamental** até hoje
- **Padrão de avaliação** para computer vision
- **Base para competições** e pesquisas

---

## Slide 22: Paralelização - Inovação Computacional

### Problema de Escalabilidade
- **Datasets milhões** de imagens
- **RBMs complexas**: 8000 visíveis × 20000 ocultas
- **Necessidade**: Distribuir computação eficientemente

### Algoritmo Desenvolvido
- **Divisão por máquinas**: Cada máquina processa subset de unidades
- **Sincronização**: Após cada etapa de sampling
- **Comunicação mínima**: Apenas estados binários (bits)

---

## Slide 23: Resultados de Paralelização - Binary RBM

![Speedup Binary RBM](speedup_binary_rbm_image)

### Speedup Quase Linear
**Observações**:
- **Escalabilidade excelente** até 8 máquinas
- **Minibatch maior**: Melhor eficiência
- **Double precision**: Melhor escalabilidade que single
- **Comunicação negligível** para dados binários

---

## Slide 24: Paralelização - Gaussian RBM

![Speedup Gaussian RBM](speedup_gaussian_rbm_image)

### Performance Ligeiramente Inferior
**Diferenças**:
- **Overhead de comunicação** maior (dados reais vs binários)
- **Speedup ainda significativo**
- **Minibatch grandes**: Compensam overhead

---

## Slide 25: Tempos Absolutos de Treinamento

![Training Time Binary](training_time_binary_image)

### Análise de Tempo Real
**Binary-to-Binary RBM**:
- **Redução dramática** nos tempos
- **Single precision**: ~2-3× mais rápido inicialmente
- **Escalabilidade mantida** em diferentes configurações

---

## Slide 26: Análise de Custos de Comunicação

### Custo Teórico
Para K máquinas, dados binários:
```
Total Communication = 48 × (K-1) MB por batch
```

### Características
- **Crescimento linear** com número de máquinas
- **Custo por máquina**: Constante (bound superior)
- **Rede não-bloqueante**: Comunicação paralela eficiente

---

## Slide 27: Impacto e Legado

### Contribuições Imediatas
1. **Prova de Conceito**: Deep learning funciona para imagens naturais
2. **Benchmarks Duradouros**: CIFAR-10/100 ainda usado extensivamente
3. **Paralelização Pioneira**: Base para computação distribuída moderna
4. **Metodologia Sólida**: ZCA whitening tornou-se técnica padrão

### Influência no Futuro
- **AlexNet (2012)**: Krizhevsky revoluciona computer vision
- **Frameworks Modernos**: Horovod, Ray seguem princípios similares
- **Preprocessing**: ZCA ainda usado em modelos atuais

---

## Slide 28: Conexões com Desenvolvimentos Modernos

### Energy-Based Models (2020+)
- **EBGAN**: Combinam conceitos RBM com GANs
- **JEM**: Modelos unificados classificação/geração
- **Score-Based**: Gradientes de energia para geração

### Contrastive Learning
- **SimCLR, MoCo**: Evolução direta de Contrastive Divergence
- **CLIP**: Aprendizado contrastivo multimodal
- **Self-Supervised**: Princípios de CD em nova roupagem

### Arquiteturas Modernas
- **Vision Transformers**: Patches similar à abordagem de Krizhevsky
- **ResNets**: Inspiração nas conexões skip das DBNs

---

## Slide 29: Lições Aprendidas

### Insights Técnicos
- **Preprocessing é crucial**: ZCA whitening foi fundamental
- **Estratégia de patches**: Eficaz para escalar complexidade
- **Paralelização inteligente**: Comunicação mínima é chave
- **Inicialização importa**: RBM features >> random initialization

### Insights Metodológicos
- **Benchmarks confiáveis**: Essenciais para progresso científico
- **Visualização de filtros**: Crucial para validar aprendizado
- **Análise sistemática**: Matrizes de confusão revelam padrões

---

## Slide 30: Conclusões

### Trabalho Revolucionário
Este trabalho de 2009 **estabeleceu as bases** do que viria a ser a revolução do deep learning

### Contribuições Duradouras
1. **Demonstração prática**: Deep learning funciona em dados reais
2. **Infraestrutura**: Datasets e algoritmos para a comunidade
3. **Metodologia**: Padrões de avaliação e visualização
4. **Escalabilidade**: Soluções para computação distribuída

### Importância Histórica
**Ponte crucial** entre os fundamentos teóricos de Hinton (2006) e a revolução prática do AlexNet (2012)

---

## Slide 31: Perguntas?

### Contato
- **Resumo completo**: Disponível em formato markdown
- **Implementações**: Código de exemplo para RBMs e ZCA
- **Bibliografia**: Referências completas para aprofundamento

### Tópicos para Discussão
- Comparação com métodos modernos
- Aplicações atuais de RBMs
- Evolution para Transformers
- Princípios de paralelização em deep learning atual

---

## Slide 32: Referências Principais

### Artigos Fundamentais
- Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images"
- Hinton, G. E. (2006). "A Fast Learning Algorithm for Deep Belief Nets"
- Hinton, G. E. (2002). "Training Products of Experts by Minimizing Contrastive Divergence"

### Trabalhos Relacionados
- Krizhevsky, A. et al. (2012). "ImageNet classification with deep convolutional neural networks"
- Bengio, Y. (2009). "Learning Deep Architectures for AI"
- LeCun, Y. et al. (1998). "Gradient-based learning applied to document recognition"

---

**Fim da Apresentação**
