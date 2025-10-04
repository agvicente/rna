# GUIA DE INSERÇÃO DE IMAGENS E EQUAÇÕES

## MAPEAMENTO DETALHADO PARA POWERPOINT

---

### SLIDE 4: PROPRIEDADES DOS DADOS

**IMAGEM 1 - Covariance Matrix (Figure 1.1)**
```
Descrição: Matriz de covariância do dataset tiny images. Branco = valores altos, preto = valores baixos. 
Pixels indexados em row-major order. Matriz dividida em 9 quadrados pelos 3 canais de cor.
Posição: Centro do slide, tamanho grande
```

**IMAGEM 2 - Red Channel Covariance (Figure 1.2)**  
```
Descrição: Matriz de covariância apenas do canal vermelho. Ampliação do quadrante superior-esquerdo da Figure 1.1.
Posição: Slide adicional ou canto inferior direito
```

---

### SLIDE 5: ZCA WHITENING - TEORIA

**EQUAÇÕES MATEMÁTICAS:**
```latex
\text{Matriz de Covariância: } C = \frac{1}{n-1} XX^T
```
```latex
\text{Decomposição: } C = PDP^T  
```
```latex
\text{Matriz de Whitening: } W = \frac{1}{\sqrt{n-1}} P D^{-1/2} P^T
```
```latex
\text{Dados Transformados: } Y = WX
```

---

### SLIDE 6: FILTROS DE WHITENING

**IMAGEM 3 - Whitening Filters (Figure 1.3)**
```
Descrição: Filtros de whitening para pixel (2,0) parte (a) e pixel (15,15) parte (b).
Mostra componentes RGB separadamente. Filtros altamente locais confirmam correlação espacial.
Posição: Metade superior do slide
```

**IMAGEM 4 - Dewhitening Filters (Figure 1.4)**
```
Descrição: Filtros de dewhitening correspondentes aos da Figure 1.3. 
Permitem reconstrução da imagem original a partir dos dados whitened.
Posição: Metade inferior do slide
```

---

### SLIDE 7: EFEITO DO WHITENING

**IMAGEM 5 - Original vs Whitened (Figure 1.5)**
```
Descrição: Resultado do whitening. Imagem original vs whitened, separado por canais RGB.
Preserva informação de bordas enquanto remove regiões uniformes.
Posição: Centro do slide, ocupando maior parte do espaço
```

---

### SLIDE 8: RESTRICTED BOLTZMANN MACHINES

**IMAGEM 6 - RBM Architecture (Figure 1.6)**
```
Descrição: Arquitetura da Restricted Boltzmann Machine.
Unidades visíveis (círculos brancos) conectadas a unidades ocultas (círculos cinzas).
Posição: Lado esquerdo do slide
```

**EQUAÇÃO 1 - Energy Function Binary RBM:**
```latex
E(v,h) = -\sum_{i=1}^V \sum_{j=1}^H v_i h_j w_{ij} - \sum_{i=1}^V v_i b_i^v - \sum_{j=1}^H h_j b_j^h
```
```
Onde:
• v: vetor de estado binário das unidades visíveis
• h: vetor de estado binário das unidades ocultas  
• wᵢⱼ: peso real entre unidade visível i e oculta j
• bᵢᵛ, bⱼʰ: biases das unidades
```

---

### SLIDE 9: RBM GAUSSIANA-BERNOULLI

**EQUAÇÃO 2 - Energy Function Gaussian RBM:**
```latex
E(v,h) = \sum_{i=1}^V \frac{(v_i - b_i^v)^2}{2\sigma_i^2} - \sum_{j=1}^H b_j^h h_j - \sum_{i=1}^V \sum_{j=1}^H \frac{v_i}{\sigma_i} h_j w_{ij}
```
```
Para dados de valor real (intensidades de pixels).
• vᵢ: atividade real da unidade visível i
• σᵢ: controla largura da parábola da unidade i
```

---

### SLIDE 10: CONTRASTIVE DIVERGENCE

**IMAGEM 7 - CD-N Procedure (Figure 1.7)**
```
Descrição: Procedimento de aprendizado CD-N. Para estimar E_model[vᵢhⱼ], 
inicializa unidades visíveis nos dados e alterna sampling das ocultas e visíveis.
Posição: Centro do slide
```

**EQUAÇÃO - Weight Update:**
```latex
\Delta w_{ij} = \epsilon \left( \mathbb{E}_{data}[v_i h_j] - \mathbb{E}_{model}[v_i h_j] \right)
```

---

### SLIDE 11: DEEP BELIEF NETWORKS

**IMAGEM 8 - DBN Architecture (Figure 1.8)**
```
Descrição: Arquitetura DBN. RBM da segunda camada treinada nas atividades das unidades
ocultas da primeira camada, mantendo W₁ fixo.
Posição: Centro do slide
```

---

### SLIDE 12: PROBLEMA INICIAL

**IMAGEM 9 - Meaningless Filters (Figure 2.1)**
```
Descrição: Filtros sem sentido aprendidos por RBM em dados whitened.
Filtros estão no domínio whitened, aplicados a imagens whitened.
Posição: Centro do slide, grid de filtros
```

---

### SLIDE 13: ANÁLISE ESPECTRAL

**IMAGEM 10 - Log Eigenspectrum (Figure 2.2)**
```
Descrição: Log-eigenspectrum do dataset tiny images. 
Variância nas 1000 componentes menos significativas é ordens de magnitude menor.
Posição: Centro do slide
```

---

### SLIDE 14: ESTRATÉGIA DE PATCHES

**IMAGEM 11 - Segmenting Images (Figure 2.4)**
```
Descrição: Segmentação de imagem 32×32 em 25 patches de 8×8.
Mostra numeração e organização espacial dos patches.
Posição: Centro do slide
```

---

### SLIDE 15: SUCESSO COM PATCHES

**IMAGEM 12 - Filters 8x8 Patch (Figure 2.5)**
```
Descrição: Filtros aprendidos por RBM no patch #1 (8×8) do dataset tiny images whitened.
RBMs nos outros patches aprenderam filtros similares. Detectores de borda de qualidade.
Posição: Centro do slide, grid grande de filtros
```

---

### SLIDE 16: FILTROS SUBSAMPLED

**IMAGEM 13 - Subsampled Filters (Figure 2.6)**
```
Descrição: Filtros aprendidos por RBM em versões 8×8 subsampled das imagens 32×32 whitened.
Filtros mais suaves e globais comparado aos patches.
Posição: Centro do slide
```

---

### SLIDE 17: MERGE DE RBMS

**IMAGEM 14 - Converting Units (Figure 2.7)**
```
Descrição: Como converter unidades ocultas treinadas em patch 8×8 subsampled globalmente 
para unidades da imagem 32×32 completa. Peso w duplicado 16 vezes e dividido por 16.
Posição: Centro do slide
```

---

### SLIDE 19: ANÁLISE DE ERROS

**IMAGEM 15 - Confusion Matrix (Figure 3.2)**
```
Descrição: Matriz de confusão para regressão logística com 10000 features RBM em dados não-whitened.
Área do quadrado (i,j) indica frequência de classificação errônea. Valores somam 1 por linha.
Posição: Centro do slide
```

---

### SLIDE 22: SPEEDUP BINÁRIO

**IMAGEM 16 - Binary RBM Speedup (Figure 4.2)**
```
Descrição: Speedup por paralelização para RBM binária-binária (vs 1 thread em 1 máquina).
(a) Minibatch 100, double precision (b) Minibatch 1000, double precision
(c) Minibatch 100, single precision (d) Minibatch 1000, single precision
Posição: Ocupar todo slide com 4 subgráficos
```

---

### SLIDE 23: SPEEDUP GAUSSIANO

**IMAGEM 17 - Gaussian RBM Speedup (Figure 4.3)**
```
Descrição: Speedup por paralelização para RBM Gaussiana-binária (vs 1 thread em 1 máquina).
Mesma estrutura da Figure 4.2 mas com overhead maior devido a dados reais.
Posição: Ocupar todo slide com 4 subgráficos
```

---

### SLIDE 24: TEMPOS ABSOLUTOS

**IMAGEM 18A - Binary Training Time (Figure 4.4)**
```
Descrição: Tempo para treinar em 8000 exemplos de dados binários aleatórios (RBM binária-binária).
Mostra tempos reais de treinamento em diferentes configurações.
Posição: Metade superior do slide
```

**IMAGEM 18B - Gaussian Training Time (Figure 4.5)**
```
Descrição: Tempo para treinar em 8000 exemplos de dados reais aleatórios (RBM Gaussiana-binária).
Complementa análise de performance com dados reais.
Posição: Metade inferior do slide
```

---

### SLIDE 25: FATORES DE SPEEDUP

**IMAGEM 19A - Binary Speedup Factor (Figure 4.6)**
```
Descrição: Fator de speedup para RBM binária-binária: 4 vs 2 máquinas (quadrados azuis) 
e 8 vs 4 máquinas (diamantes vermelhos). Dobrar máquinas quase dobra performance.
Posição: Metade superior do slide
```

**IMAGEM 19B - Gaussian Speedup Factor (Figure 4.7)**
```
Descrição: Fator de speedup para RBM Gaussiana-binária com mesma estrutura.
Performance ligeiramente inferior devido ao overhead de comunicação.
Posição: Metade inferior do slide
```

---

### SLIDE 25: ANÁLISE DE CUSTOS

**EQUAÇÃO - Communication Cost:**
```latex
\text{Custo Total} = 48 \times (K-1) \text{ MB por batch}
```
```latex
\text{Para dados binários com K máquinas}
```

---

## RESUMO DE DISTRIBUIÇÃO:

- **Total de Slides**: 30
- **Imagens Únicas**: 19 (algumas figuras usadas múltiplas vezes)
- **Equações LaTeX**: 8 
- **Slides com Imagens**: 15
- **Slides com Equações**: 5
- **Slides Apenas Texto**: 10

## SUGESTÕES DE DESIGN:

1. **Esquema de Cores**: Azul escuro (#1f4e79) para títulos, cinza (#5f5f5f) para texto
2. **Fontes**: Calibri ou Arial para legibilidade
3. **Imagens**: Manter proporção original, bordas sutis
4. **Equações**: Fundo levemente sombreado para destaque
5. **Transições**: Simples, sem distrair do conteúdo científico
