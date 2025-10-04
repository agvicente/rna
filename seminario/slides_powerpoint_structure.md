# ESTRUTURA PARA POWERPOINT - Learning Multiple Layers of Features from Tiny Images

## INSTRUÇÕES PARA CRIAÇÃO DOS SLIDES

### Como usar este documento:
1. Cada "---" marca um novo slide
2. Imagens estão identificadas com [IMAGEM X] 
3. Equações estão em blocos LaTeX
4. Bullet points mantêm hierarquia visual
5. Cores sugeridas: Azul escuro para títulos, cinza para texto

---

## SLIDE 1: TÍTULO
**Learning Multiple Layers of Features from Tiny Images**

Alex Krizhevsky (2009)
University of Toronto
Orientador: Geoffrey Hinton

*Apresentação para Banca de Pós-Graduação*

---

## SLIDE 2: CONTEXTO HISTÓRICO (2009)

### O Renascimento do Deep Learning
• **2006**: Breakthrough de Hinton com DBNs
• **Problema**: Como treinar redes profundas?
• **Solução**: Pré-treinamento não supervisionado

### Desafios da Época
• Vanishing gradient problem
• Falta de grandes datasets rotulados  
• Limitações computacionais
• Dependência de feature engineering manual

---

## SLIDE 3: MOTIVAÇÃO

### Falhas Anteriores
• MIT/NYU falharam com "80 million tiny images"
• Modelos aprendiam apenas filtros ruidosos

### Objetivos do Trabalho
1. **Modelagem generativa** eficaz de imagens
2. **Datasets confiáveis** para benchmarking
3. **Paralelização** para escalabilidade

---

## SLIDE 4: PROPRIEDADES DOS DADOS

[IMAGEM 1: Covariance matrix of all pixels - Figure 1.1]
*Matriz de covariância mostrando correlações entre pixels RGB*

### Características das Imagens Naturais
• Pixels próximos: **fortemente correlacionados**
• Pixels distantes: **fracamente correlacionados**
• **Simetria** horizontal/vertical
• **Separação por canais** de cor

---

## SLIDE 5: ZCA WHITENING - TEORIA

### Por que Whitening?
• Remove correlações de **segunda ordem**
• Força foco em correlações de **alta ordem**
• **Fundamental** para sucesso do método

### Formulação Matemática
```latex
C = \frac{1}{n-1} XX^T
```
```latex
W = \frac{1}{\sqrt{n-1}} P D^{-1/2} P^T
```
```latex
Y = WX
```

---

## SLIDE 6: FILTROS DE WHITENING

[IMAGEM 3: Whitening filters - Figure 1.3]
*Filtros para componentes RGB de pixels específicos*

[IMAGEM 4: Dewhitening filters - Figure 1.4] 
*Filtros correspondentes para reconstrução*

### Características
• **Altamente locais** devido à correlação espacial
• **Separação por canais** RGB
• **Simetria** reflete propriedades das imagens

---

## SLIDE 7: EFEITO DO WHITENING

[IMAGEM 5: Original vs Whitened image - Figure 1.5]
*Comparação entre imagem original e whitened*

### Resultados
• **Preserva** informação de bordas
• **Remove** regiões uniformes  
• **Destaca** estruturas importantes
• **Separação clara** por canais

---

## SLIDE 8: RESTRICTED BOLTZMANN MACHINES

[IMAGEM 6: RBM architecture - Figure 1.6]
*Arquitetura básica da RBM com unidades visíveis e ocultas*

### Função de Energia (RBM Binária)
```latex
E(v,h) = -\sum_{i,j} v_i h_j w_{ij} - \sum_i v_i b_i^v - \sum_j h_j b_j^h
```

**Onde:**
• v: estado das unidades visíveis
• h: estado das unidades ocultas
• w_ij: pesos entre unidades

---

## SLIDE 9: RBM GAUSSIANA-BERNOULLI

### Para Dados Reais (Intensidades)

[EQUAÇÃO DA IMAGEM: Energy function (1.8)]
```latex
E(v,h) = \sum_{i=1}^V \frac{(v_i - b_i^v)^2}{2\sigma_i^2} - \sum_{j=1}^H b_j^h h_j - \sum_{i=1}^V \sum_{j=1}^H \frac{v_i}{\sigma_i} h_j w_{ij}
```

### Distribuições Condicionais
• **Visíveis**: Gaussianas com média dependente de h
• **Ocultas**: Bernoulli com probabilidade sigmoid

---

## SLIDE 10: CONTRASTIVE DIVERGENCE

[IMAGEM 7: CD-N procedure - Figure 1.7]
*Procedimento de aprendizado CD-N com sampling alternado*

### Algoritmo CD-1
```latex
\Delta w_{ij} = \epsilon \langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model}
```

• **E_data**: Expectativa com visíveis fixadas
• **E_model**: Expectativa aproximada por CD
• **Eficiente**: Apenas 1 passo de Gibbs sampling

---

## SLIDE 11: DEEP BELIEF NETWORKS

[IMAGEM 8: DBN architecture - Figure 1.8]
*Arquitetura de DBN com múltiplas camadas RBM*

### Treinamento Layer-by-Layer
1. **RBM 1**: Treina nos dados originais
2. **RBM 2**: Treina nas ativações da RBM 1
3. **Repetir**: Para camadas adicionais
4. **Fine-tuning**: Ajuste supervisionado opcional

---

## SLIDE 12: PROBLEMA INICIAL

[IMAGEM 9: Meaningless filters - Figure 2.1]
*Filtros sem sentido aprendidos por RBM em dados whitened*

### Causa do Problema
• **Ruído de alta frequência** dominante
• **Correlações complexas** não capturadas
• **Necessidade**: Estratégias mais sofisticadas

---

## SLIDE 13: ANÁLISE ESPECTRAL

[IMAGEM 10: Log eigenspectrum - Figure 2.2]
*Espectro logarítmico mostrando variância das componentes*

### Solução: Remoção de Componentes
• **1000 componentes** menos significativas removidas
• **Variância**: Várias ordens de magnitude menor
• **Resultado**: Redução significativa do ruído

---

## SLIDE 14: ESTRATÉGIA DE PATCHES

[IMAGEM 11: Segmenting 32x32 into patches - Figure 2.4]
*Divisão da imagem 32×32 em 25 patches de 8×8*

### Abordagem
• **25 patches** de 8×8 pixels
• **1 patch global** subsampled
• **26 RBMs independentes**
• **Redução**: Complexidade dimensional

---

## SLIDE 15: SUCESSO COM PATCHES

[IMAGEM 12: Filters learned on 8x8 patch - Figure 2.5]
*Filtros de qualidade aprendidos em patch 8×8*

### Breakthrough: Detectores de Borda
• **Filtros coloridos**: Baixa frequência
• **Filtros P&B**: Alta frequência  
• **Interpretação**: Posição precisa + cor aproximada

---

## SLIDE 16: FILTROS SUBSAMPLED

[IMAGEM 13: Filters on subsampled versions - Figure 2.6]
*Filtros aprendidos em versões subsampled da imagem*

### Características
• **Mais suaves** e globais
• **Escala diferente** de detecção
• **Complementar** aos patches locais

---

## SLIDE 17: MERGE DE RBMS

[IMAGEM 14: Converting hidden units - Figure 2.7]
*Como converter unidades de patch para imagem completa*

### Procedimento
• **Duplicação**: Pesos × 16 para patch global
• **Divisão**: Por fator correspondente  
• **Inicialização zero**: Conexões inexistentes
• **Untied weights**: Liberdade para diferenciação

---

## SLIDE 18: RESULTADOS DE CLASSIFICAÇÃO

### CIFAR-10 Performance

| **Método** | **Erro (%)** |
|------------|--------------|
| Logistic (raw pixels) | ~40 |
| Logistic (whitened) | ~37 |
| Logistic (RBM features) | **~22** |
| Neural Net (RBM init) | **~18.5** |

### Insights
• **Features RBM** >>> pixels crus
• **Dados não-whitened** melhores para RBMs

---

## SLIDE 19: ANÁLISE DE ERROS

[IMAGEM 15: Confusion matrix - Figure 3.2]
*Matriz de confusão mostrando padrões de classificação*

### Padrões Descobertos
• **Clustering animal vs não-animal**
• **Alta confusão**: cat ↔ dog
• **Ocasional**: bird ↔ plane
• **Estrutura semântica** capturada

---

## SLIDE 20: DATASET CIFAR

### Contribuição Duradoura
**CIFAR-10**: 10 classes, 60.000 imagens
**CIFAR-100**: 100 classes, 60.000 imagens

### Metodologia Rigorosa
• **Critérios claros** para rotulação
• **Remoção de duplicatas**
• **Divisão treino/teste** padronizada

### Impacto
• **Benchmark fundamental** até hoje
• **Base para pesquisas** em computer vision

---

## SLIDE 21: PARALELIZAÇÃO - INOVAÇÃO

### Desafio Computacional
• **8000 visíveis × 20000 ocultas**
• **Milhões de imagens**
• **Necessidade**: Distribuição eficiente

### Algoritmo Desenvolvido
• **Divisão por máquinas**: Subset de unidades
• **Sincronização**: Após cada sampling
• **Comunicação mínima**: Apenas bits

---

## SLIDE 22: RESULTADOS DE SPEEDUP - BINÁRIO

[IMAGEM 16: Speedup binary RBM - Figure 4.2]
*Gráficos de speedup para RBM binária-binária*

### Escalabilidade Excelente
• **Speedup quase linear** até 8 máquinas
• **Minibatch maior**: Melhor eficiência
• **Double precision**: Melhor que single
• **Comunicação**: Praticamente negligível

---

## SLIDE 23: SPEEDUP GAUSSIANO

[IMAGEM 17: Speedup Gaussian RBM - Figure 4.3]
*Gráficos de speedup para RBM Gaussiana-Bernoulli*

### Performance Ainda Excelente
• **Overhead maior**: Dados reais vs binários
• **Speedup significativo** mantido
• **Minibatches grandes**: Compensam overhead

---

## SLIDE 24: TEMPOS ABSOLUTOS

[IMAGEM 18: Training time graphs - Figure 4.4 & 4.5]
*Tempos reais de treinamento em diferentes configurações*

### Análise Prática
• **Redução dramática** nos tempos
• **Single precision**: 2-3× mais rápido
• **Escalabilidade**: Mantida em todas configurações

---

## SLIDE 25: ANÁLISE DE CUSTOS

[IMAGEM 19: Speedup factor comparisons - Figure 4.6 & 4.7]
*Fatores de speedup comparando diferentes números de máquinas*

### Custo de Comunicação
```
Total = 48 × (K-1) MB por batch (dados binários)
```

• **Crescimento linear** com K máquinas
• **Custo por máquina**: Constante
• **Eficiência**: Mantida mesmo com 8 máquinas

---

## SLIDE 26: IMPACTO CIENTÍFICO

### Contribuições Imediatas
1. **Prova de conceito**: Deep learning para imagens reais
2. **Benchmarks duradouros**: CIFAR ainda usado
3. **Paralelização pioneira**: Base para frameworks atuais
4. **Metodologia sólida**: ZCA tornou-se padrão

### Influência Futura
• **AlexNet (2012)**: Revolução de Krizhevsky
• **Frameworks modernos**: Horovod, Ray
• **Preprocessing**: Técnicas ainda relevantes

---

## SLIDE 27: CONEXÕES MODERNAS

### Energy-Based Models (2020+)
• **EBGAN**: RBM + GANs
• **JEM**: Classificação + geração unificada
• **Score-based**: Gradientes de energia

### Contrastive Learning
• **SimCLR, MoCo**: Evolução de CD
• **CLIP**: Contrastivo multimodal
• **Self-supervised**: Princípios de CD

### Arquiteturas
• **Vision Transformers**: Patches similares
• **ResNets**: Skip connections inspiradas em DBNs

---

## SLIDE 28: LIÇÕES APRENDIDAS

### Insights Técnicos
• **Preprocessing crucial**: ZCA fundamental
• **Patches eficazes**: Para escalar complexidade
• **Paralelização inteligente**: Comunicação mínima
• **Inicialização importa**: RBM >>> random

### Insights Metodológicos  
• **Benchmarks confiáveis**: Essenciais para ciência
• **Visualização**: Validação por filtros aprendidos
• **Análise sistemática**: Matrizes de confusão

---

## SLIDE 29: CONCLUSÕES

### Trabalho Revolucionário
Este trabalho estabeleceu as **bases da revolução** do deep learning

### Legado Duradouro
• **Ponte histórica**: Hinton (2006) → AlexNet (2012)
• **Infraestrutura**: Datasets e algoritmos para comunidade
• **Metodologia**: Padrões de avaliação e visualização
• **Escalabilidade**: Computação distribuída

### Importância
**Demonstrou viabilidade prática** do deep learning em dados reais

---

## SLIDE 30: PERGUNTAS?

### Disponível
• **Resumo completo**: Markdown detalhado
• **Código exemplo**: RBMs e ZCA whitening
• **Bibliografia**: Referências para aprofundamento

### Discussão
• Comparação com métodos atuais
• Aplicações modernas de RBMs
• Evolução para Transformers
• Paralelização em deep learning atual

---

## NOTAS TÉCNICAS PARA MONTAGEM:

### Imagens por Slide:
- Slide 4: Covariance matrix (Figures 1.1 & 1.2)
- Slide 6: Whitening/Dewhitening filters (Figures 1.3 & 1.4)  
- Slide 7: Original vs Whitened (Figure 1.5)
- Slide 8: RBM architecture (Figure 1.6)
- Slide 10: CD-N procedure (Figure 1.7)
- Slide 11: DBN architecture (Figure 1.8)
- Slide 12: Meaningless filters (Figure 2.1)
- Slide 13: Log eigenspectrum (Figure 2.2)
- Slide 14: Patches segmentation (Figure 2.4)
- Slide 15: Good filters 8x8 (Figure 2.5)
- Slide 16: Subsampled filters (Figure 2.6)
- Slide 17: Converting units (Figure 2.7)
- Slide 19: Confusion matrix (Figure 3.2)
- Slide 22: Binary speedup (Figure 4.2)
- Slide 23: Gaussian speedup (Figure 4.3)
- Slide 24: Training times (Figures 4.4 & 4.5)
- Slide 25: Speedup factors (Figures 4.6 & 4.7)

### Equações por Slide:
- Slide 5: ZCA whitening math
- Slide 8: Binary RBM energy (Equation 1.1)
- Slide 9: Gaussian RBM energy (Equation 1.8)
- Slide 10: CD weight update
- Slide 25: Communication cost formula
