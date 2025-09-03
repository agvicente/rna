---
Título: Online Federated Learning via Non-Stationary Detection and Adaptation Amidst Concept Drift
Autor(es): Bhargav Ganguly, Vaneet Aggarwal
Data: February 2024
Fonte: IEEE/ACM Transactions on Networking, Vol. 32, No. 1
Palavras-chave: Federated Learning, Non-Stationary, Concept Drift, Dynamic Regret, Online Convex Optimization
---
**Resumo/Abstract**

Este trabalho propõe um framework algorítmico multi-escala que combina garantias teóricas dos algoritmos FedAvg e FedOMD em configurações quase estacionárias com técnicas de detecção e adaptação não-estacionárias para melhorar o desempenho de generalização do FL na presença de concept drifts.

# 📌 Problema e Contexto
**Qual é o problema abordado? Por que ele é importante?**  
- A literatura existente em Federated Learning (FL) assume processos de geração de dados estacionários, o que é irrealista em condições do mundo real onde ocorrem concept drifts
- Concept drifts ocorrem devido a observações sazonais, falhas em sensores, mudanças abruptas no ambiente (ex: pandemia afetando dados de reservas de voos)
- Metodologias convencionais como FedAvg são agnósticas a essas mudanças temporais nos dados, resultando em piores resultados de generalização
- É crítico aumentar esses frameworks de aprendizado com procedimentos de detecção e adaptação de não-estacionaridade

# 🎯 Objetivo do Estudo
**O que o estudo pretende resolver ou demonstrar?**  
- Desenvolver um framework multi-escala que pode equipar qualquer metodologia FL baseline que funcione bem em ambientes quase estacionários
- Proporcionar a primeira análise de regret dinâmico para otimização convexa online para funções convexas gerais no contexto de FL
- Demonstrar bounds de regret dinâmico em termos do número de mudanças de drift (L) e magnitude cumulativa de drift (∆)
- Criar testes de detecção de drift não baseados apenas em heurísticas, mas fundamentados matematicamente

# 🛠️ Metodologia
**Como o estudo foi conduzido?**  
- Tipo de pesquisa: Teórico-experimental com análise matemática rigorosa e validação experimental
- Ferramentas e técnicas utilizadas: 
  - Framework algorítmico multi-escala (Master-FL)
  - Procedimento de agendamento randomizado para instâncias de algoritmos base
  - Dois testes de detecção de drift (Test 1 e Test 2) baseados em decomposição matemática do regret dinâmico
- Algoritmos/métodos: 
  - Algoritmos base: FedAvg e FedOMD
  - Multi-Scale FL Runner (MSFR)
  - Testes de não-estacionaridade com fundamentação teórica
- Dataset e experimentação: 
  - Datasets LIBSVM: covtype e mnist
  - 20 clientes DPUs, 500 rounds de treinamento
  - Dois tipos de concept drift: Class Introduction (CI) e Class Swap (CS)
  - Comparação com FedNova e FedProx

# 📊 Resultados Obtidos
**Quais foram as descobertas principais?**  
- Bound de regret dinâmico de Õ(min{√LT, ∆^(1/3)T^(2/3) + √T}) para T rounds
- Framework preserva propriedades dos algoritmos base FL em horizontes quase estacionários
- Complexidade de armazenamento limitada por Õ(C(T)) instâncias sobre horizonte T
- Experimentalmente, Master-FL-FedAvg e Master-FL-FedOMD superam métodos competidores
- Capacidade de re-treinamento rápido de modelos impactados por concept drift
- Primeira análise rigorosa de regret dinâmico para FL não-estacionário

# 🔍 Análise e Discussão
**Quais são as implicações dos resultados? Como eles se relacionam com o problema inicial?**  
- O framework resolve o problema fundamental de FL assumir dados estacionários
- Os testes de detecção são matematicamente fundamentados (não apenas heurísticos) através da decomposição do regret dinâmico
- O bound obtido é mais apertado que trabalhos anteriores em otimização distribuída
- A abordagem multi-escala permite detecção eficiente de drifts "súbitos" (Test 1) e "graduais" (Test 2)
- Framework é geral e pode ser aplicado a outros algoritmos FL que funcionem bem em configurações quase estacionárias
- Resultados experimentais confirmam a eficácia teórica do método

# ✅ Conclusão e Impacto
**Qual é a principal contribuição do artigo? Há recomendações futuras?**  
- Primeira análise de regret dinâmico para otimização federated não-estacionária com funções convexas gerais
- Framework unificado que combina algoritmos FL baseline com detecção/adaptação de mudanças
- Bounds teóricos rigorosos sem necessidade de conhecimento prévio de L, ∆ ou T
- **Trabalhos futuros recomendados:**
  - Estender para outros algoritmos FL baseline além de FedAvg e FedOMD
  - Considerar abordagens de seleção de clientes para dados não-estacionários heterogêneos
  - Investigação detalhada de técnicas de preservação de privacidade para computação de loss

# 📚 Referências Adicionais
- Sugiyama & Kawanabe (2012) - Machine Learning in Non-Stationary Environments
- McMahan et al. (2017) - Communication-Efficient Learning (FedAvg original)
- Wei & Luo (2021) - Non-stationary reinforcement learning (base para framework multi-escala)
- Mallick et al. (2022) - Matchmaker: Data drift mitigation in ML
- Wang et al. (2020) - FedNova: Tackling objective inconsistency in heterogeneous FL