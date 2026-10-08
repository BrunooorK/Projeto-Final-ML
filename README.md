# Projeto Final Machine Learning
Área 3: *Análise de sentimentos e classificação de texto*.

> **Estado atual:** as etapas 3 a 9 (EDA até explicabilidade) estão implementadas em `app.py` e foram executadas localmente (VS Code, Windows). O notebook do Google Colab com as etapas 5 a 9, a análise de viés e a simulação de impacto de negócio ainda estão em andamento: veja [Próximos passos](#9-próximos-passos).

## Sumário

1. [Problema e objetivo](#1-problema-e-objetivo)
2. [Dataset](#2-dataset)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Metodologia](#4-metodologia)
5. [Resultados da execução de referência](#5-resultados-da-execução-de-referência)
6. [Como reproduzir no Windows (VS Code + PowerShell)](#6-como-reproduzir-no-windows-vs-code--powershell)
7. [Configuração do `app.py`](#7-configuração-do-apppy)
8. [Como continuar o projeto](#8-como-continuar-o-projeto)
9. [Próximos passos](#9-próximos-passos)
10. [Limitações](#10-limitações)
11. [Uso de ferramentas de IA e referências](#11-uso-de-ferramentas-de-ia-e-referências)

---

## 1. Problema e objetivo

Grandes plataformas de varejo online (como as do ecossistema B2W e Olist) recebem milhares de avaliações de clientes por dia. A nota em estrelas indica a satisfação geral, mas não isola a percepção financeira do consumidor: um cliente pode dar 4 estrelas pela qualidade técnica e, no texto, dizer que o valor pago foi desproporcional. Ler esses textos manualmente é lento, caro e inviável em grande escala.

**Objetivo deste repositório:** construir, validar e interpretar modelos que classificam avaliações em português como **positivas** ou **negativas**, comparando-os com um modelo de referência (*baseline*).

**Escopo atual:** o modelo estima a **polaridade geral** do texto. A análise específica da percepção de preço ainda não foi modelada (veja [Próximos passos](#9-próximos-passos)).

## 2. Dataset

- **Fonte:** [Brazilian Portuguese Sentiment Analysis Datasets (Kaggle)](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets)
- **Bases reunidas:** B2W, Buscapé, Olist, UTLC Apps e UTLC Movies.
- **Arquivo usado:** somente `archive/concatenated.csv`, que reúne as bases. Os arquivos individuais **não** são carregados junto, para evitar duplicação.
- **Colunas usadas:** `review_text` (entrada) e `polarity` (alvo: 0 = negativo, 1 = positivo).
- **Outras colunas do arquivo:** `original_index`, `review_text_processed`, `review_text_tokenized`, `rating`, `kfold_polarity`, `kfold_rating` e `dataset` (origem da linha).
- **Tamanho bruto:** 2.786.092 registros, dos quais 399.928 não têm `polarity`.
- **`kfold_polarity`:** não é usada no treino. Na execução de referência, o valor `-1` aparece em 399.928 linhas (o mesmo número de rótulos ausentes) e os valores de 1 a 10 aparecem em cerca de 238,6 mil linhas cada. Isso sugere uma divisão em 10 partes apenas para as linhas rotuladas, mas **o significado não foi confirmado pela documentação** e a divisão foi feita sobre os dados originais, com duplicados. Por isso o projeto usa a sua própria divisão estratificada.
- **Licença e citação:** consulte a licença na página do Kaggle. Os dados **não** estão neste repositório. Veja [`archive/LEIA-ME.md`](archive/LEIA-ME.md) para baixá-los.

## 3. Estrutura do repositório

```
analise-sentimentos-ptbr/
├── app.py                      # Código principal (etapas 3 a 9), roda do início ao fim
├── requirements.txt            # Dependências
├── README.md
├── .gitignore
├── .gitattributes
├── archive/
│   └── LEIA-ME.md              # Como baixar o dataset (os CSV não são versionados)
├── resultados_referencia/      # Resultados da execução de referência (versionados)
│   ├── relatorio.md
│   ├── eda_graficos.png
│   ├── comparacao_modelos.png
│   ├── matriz_confusao.png
│   ├── termos_mais_importantes.png
│   ├── comparacao_modelos.csv
│   ├── grade_hiperparametros.csv
│   └── exemplos_de_erros.csv
└── docs/
    └── GUIA_GIT_GITHUB.md      # Passo a passo de Git/GitHub para o grupo
```

Quando você roda o `app.py`, ele cria a pasta `resultados/` (ignorada pelo Git). Assim, rodar de novo não altera os resultados de referência.

O `app.py` é um único script organizado em seções numeradas (carregar dados, EDA, tratamento, baseline, modelagem e validação, métricas, explicabilidade, salvar arquivos). Não foi dividido em módulos porque está funcionando e o tamanho ainda é razoável.

## 4. Metodologia

| Etapa | O que o `app.py` faz | Justificativa |
|---|---|---|
| **3. EDA** | Conta ausentes, duplicados, distribuição das classes e tamanho dos textos; gera `eda_graficos.png` | Entender os dados antes de decidir o tratamento. O histograma de tamanhos é cortado no percentil 99 **apenas para visualização** |
| **4. Tratamento** | Remove ausentes e duplicados; normaliza o texto (minúsculas, sem links e símbolos, acentos mantidos); remove textos com até 2 letras; remove textos iguais após a normalização e textos iguais com rótulos conflitantes; amostra estratificada de 60.000 | Sem texto ou rótulo não há o que aprender. Duplicados e conflitos podem cair no treino e no teste ao mesmo tempo (vazamento) |
| **Codificação / escalonamento** | Rótulo já é 0/1; texto vira números com TF-IDF (1 e 2 palavras, até 60.000 termos). Sem `StandardScaler` | O TF-IDF já normaliza cada vetor. Outliers de tamanho **não** foram removidos |
| **5. Baseline** | `DummyClassifier(strategy="most_frequent")`: sempre prevê a classe mais comum | Referência mínima que qualquer modelo útil precisa superar |
| **6. Modelagem** | Regressão Logística, Naive Bayes (Multinomial) e SVM Linear, todos com TF-IDF dentro de um `Pipeline` | Três famílias de algoritmos lineares/probabilísticos, rápidas e adequadas a texto esparso |
| **7. Validação** | Divisão estratificada 80/20 (semente 42); validação cruzada estratificada de 3 folds **só no treino**; `GridSearchCV` no parâmetro C da Regressão Logística (0,1; 1; 5; 10) | O melhor modelo é escolhido pela validação cruzada, e o teste é usado uma única vez, no final |
| **8. Métricas** | Acurácia, precisão, recall e F1 por classe, F1-macro, acurácia balanceada e matriz de confusão | Há poucas avaliações negativas, então a acurácia sozinha engana (o baseline já acerta 80,7%). O foco é o **F1-macro** e o **recall da classe negativa** |
| **9. Explicabilidade** | Pesos (coeficientes) de cada termo no modelo escolhido, gráfico dos 15 termos mais positivos e negativos, contribuição por termo em 4 frases de teste | Mostrar por que o modelo decide como decide |

**Cuidados contra vazamento de dados:** duplicados e conflitos são tratados antes da divisão; o TF-IDF é ajustado dentro de cada fold e nunca vê o teste; a escolha do modelo e do hiperparâmetro usa só o treino; o baseline é avaliado no mesmo teste; semente fixa (42).

## 5. Resultados da execução de referência

Execução feita no VS Code (Windows) em 08/10/2026, com Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.6 e numpy 2.5.3. O arquivo [`resultados_referencia/relatorio.md`](resultados_referencia/relatorio.md) é a saída automática dessa execução.

> Resultados de versões anteriores do código (por exemplo, F1-macro de 0,836 para a Regressão Logística) vieram de um tratamento que não removia duplicados e conflitos após a normalização. **Não misture** esses números com os de agora.

### Dados (etapas 3 e 4)

| Passo | Registros |
|---|---:|
| Brutos | 2.786.092 |
| Sem ausentes (texto e rótulo) | 2.386.163 |
| Sem textos duplicados exatos | 2.066.231 |
| Após limpeza (textos com até 2 letras removidos) | 2.062.861 |
| Sem duplicados e conflitos após normalizar | 1.963.776 |
| Amostra estratificada | 60.000 |

- Classes entre os rótulos presentes na base bruta: 82,8% positivas e 17,2% negativas.
- Na amostra: 80,7% positivas e 19,3% negativas.
- Treino: 48.000 | Teste: 12.000 (2.312 negativas e 9.688 positivas).
- Textos: em média 26,7 palavras (mediana 13; máximo 10.605).
- Textos iguais após a normalização: 96.753; textos iguais com rótulos conflitantes: 2.332.

![EDA](resultados_referencia/eda_graficos.png)

### Ajuste de hiperparâmetros (etapa 7)

| C | F1-macro médio (validação cruzada) |
|---:|---:|
| 0,1 | 0,7859 |
| 1 | 0,8211 |
| **5** | **0,8305** |
| 10 | 0,8297 |

### Comparação dos modelos (etapas 5, 6 e 8)

| Modelo | Acurácia | Recall negativo | F1 negativo | F1-macro (teste) | F1-macro (validação cruzada) |
|---|---:|---:|---:|---:|---:|
| **Regressão Logística (C = 5)** | 0,8904 | 0,7898 | 0,7353 | **0,8331** | 0,8305 ± 0,0034 |
| SVM Linear | 0,8878 | 0,7612 | 0,7232 | 0,8264 | 0,8278 ± 0,0033 |
| Naive Bayes | 0,8758 | 0,4113 | 0,5607 | 0,7442 | 0,6892 ± 0,0063 |
| Baseline (classe majoritária) | 0,8073 | 0,0000 | 0,0000 | 0,4467 | 0,4467 ± 0,0000 |

A tabela completa (precisão, recall e F1 por classe, acurácia balanceada) está em `resultados_referencia/comparacao_modelos.csv`.

![Comparação dos modelos](resultados_referencia/comparacao_modelos.png)

**Como interpretar:**
- O baseline acerta 80,7% só prevendo "positivo", mas **não encontra nenhuma avaliação negativa**. Por isso o F1-macro é a métrica principal.
- A Regressão Logística foi escolhida pela validação cruzada. A vantagem sobre o SVM Linear é pequena (0,8305 contra 0,8278, diferenças da ordem do desvio padrão), então os dois são tecnicamente próximos.
- O Naive Bayes tem alta precisão para a classe negativa (0,88), mas só encontra 41% das avaliações negativas: é conservador demais para este problema.

### Matriz de confusão (Regressão Logística, teste)

![Matriz de confusão](resultados_referencia/matriz_confusao.png)

Das 2.312 avaliações negativas, o modelo encontrou 1.826 (79%) e deixou 486 passarem como positivas. Das 9.688 positivas, 829 (9%) foram marcadas como negativas. Quando o modelo diz "negativo", acerta em 68,8% das vezes. Total: 10.685 acertos e 1.315 erros.

### Explicabilidade (etapa 9)

![Termos mais importantes](resultados_referencia/termos_mais_importantes.png)

- **Puxam para positivo:** ótimo, excelente, muito bom, perfeito, sensacional, amei, maravilhoso, incrível.
- **Puxam para negativo:** péssimo, não gostei, horrível, não, pior, não recomendo, fraco, ruim, lixo.
- Os pesos são **associações aprendidas nesta base**, não relações de causa e efeito. Palavras como "não" mudam de sentido conforme o contexto.
- As 4 frases de teste (duas positivas e duas negativas, incluindo "Muito caro para o que entrega, não vale o preço que paguei") foram classificadas conforme o esperado. Nessa frase, os termos que mais pesaram foram "não vale" (−1,69), "paguei" (−0,84) e "não" (−0,81).

## 6. Como reproduzir no Windows (VS Code + PowerShell)

### Pré-requisitos

- [Python 3](https://www.python.org/downloads/) (o projeto foi desenvolvido com 3.14.7; se a instalação das dependências falhar em uma versão muito nova, use o Python 3.12 ou 3.13).
- [Git para Windows](https://git-scm.com/download/win).
- [VS Code](https://code.visualstudio.com/) com a extensão **Python**.
- Conta no Kaggle (para baixar o dataset).

### Passo a passo

1. **Clonar o repositório** (troque pelo endereço real):

   ```powershell
   git clone https://github.com/USUARIO/REPOSITORIO.git
   cd REPOSITORIO
   ```

2. **Abrir no VS Code:** `code .` (ou *File > Open Folder*).

3. **Criar e ativar o ambiente virtual** (no terminal do VS Code, em PowerShell):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Se aparecer um erro de política de execução, rode antes `Set-ExecutionPolicy -Scope Process Bypass` e tente de novo. No VS Code, escolha o interpretador do `.venv` em *Ctrl+Shift+P > Python: Select Interpreter*.

4. **Instalar as dependências:**

   ```powershell
   py -m pip install -r requirements.txt
   ```

5. **Baixar o dataset** e colocar em `archive/concatenated.csv`, seguindo [`archive/LEIA-ME.md`](archive/LEIA-ME.md).

6. **Executar:**

   ```powershell
   py app.py
   ```

7. **Ver os resultados** na pasta `resultados/` (criada automaticamente): gráficos `.png`, tabelas `.csv`, `relatorio.md` e o modelo `modelo_sentimentos.joblib`.

### Recursos e tempo

O script lê cerca de 2,8 milhões de linhas e consome bastante memória RAM. Se o computador travar ou faltar memória, reduza o paralelismo em `N_JOBS = 1` (veja a [configuração](#7-configuração-do-apppy)). Com a mesma semente e as mesmas versões das bibliotecas, os resultados devem ser iguais aos de referência; versões diferentes podem causar pequenas variações.

### Usar o modelo salvo

O `app.py` é um script e executa tudo ao ser importado, então para usar o modelo em outro código é preciso repetir a limpeza do texto:

```python
import re
import joblib

def limpar_texto(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-záàâãéêíóôõúüç\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

modelo = joblib.load("resultados/modelo_sentimentos.joblib")   # 0 = negativo, 1 = positivo
frases = ["Chegou quebrado e ninguém resolveu", "Adorei, recomendo muito"]
print(modelo.predict([limpar_texto(f) for f in frases]))
```

> **Segurança:** só carregue arquivos `.joblib` gerados por você ou pelo grupo. Arquivos de terceiros podem executar código.

## 7. Configuração do `app.py`

Os parâmetros ficam no início do arquivo, na seção `CONFIGURAÇÃO`:

| Parâmetro | Valor | Função |
|---|---|---|
| `MAX_LINHAS` | 60000 | Tamanho da amostra estratificada (`None` usa tudo e exige muita memória) |
| `SEMENTE` | 42 | Reprodutibilidade |
| `TAMANHO_TESTE` | 0.20 | Proporção do teste |
| `CV_FOLDS` | 3 | Folds da validação cruzada |
| `GRADE_C` | [0.1, 1, 5, 10] | Valores de C testados na Regressão Logística |
| `N_JOBS` | 2 | Paralelismo (use 1 se faltar memória) |
| `DEDUP_APOS_NORMALIZAR` | True | Remove textos iguais depois da limpeza |
| `REMOVER_CONFLITOS` | True | Remove textos iguais com rótulos diferentes |
| `INSPECIONAR_KFOLD` | True | Só mostra os valores de `kfold_polarity`; não é usada no treino |

Alterar esses valores muda os resultados. Se isso acontecer, trate como um **novo experimento** e registre os novos números.

## 8. Como continuar o projeto

Ninguém precisa refazer o desenvolvimento:

1. Leia [`resultados_referencia/relatorio.md`](resultados_referencia/relatorio.md) e este README para entender o que já foi feito e quais números são os de referência.
2. Clone o repositório e siga a seção 6 **apenas se** precisar rodar o código (por exemplo, para gerar o modelo ou testar uma mudança).
3. Trabalhe em uma **branch** por tarefa e abra um *Pull Request* para o grupo revisar. O passo a passo está em [`docs/GUIA_GIT_GITHUB.md`](docs/GUIA_GIT_GITHUB.md).
4. Evite que duas pessoas editem o mesmo arquivo ao mesmo tempo. Novas análises podem ir em arquivos ou notebooks separados.
5. O modelo `.joblib` e os dados não vão para o Git. Para compartilhar o modelo, use o Google Drive do grupo ou uma *Release* do GitHub.
6. Se rodar o código e quiser atualizar os resultados de referência, copie os arquivos de `resultados/` para `resultados_referencia/` **somente se** a execução for a nova versão oficial do grupo, e avise a todos.

## 9. Próximos passos

O enunciado da UC9 exige, além do que já foi feito, os itens abaixo. Marque quem assume cada tarefa.

- [ ] Notebook do Google Colab com as etapas 5 a 9 (responsável: ____)
- [ ] **Análise de viés:** a base não tem atributos sensíveis de pessoas, mas tem a coluna `dataset` (origem). Comparar o desempenho por origem (varejo, apps, filmes) e discutir vieses de amostragem, coleta e rotulagem (responsável: ____)
- [ ] **Simulação de impacto de negócio** com premissas explícitas (custo de uma reclamação não detectada, esforço de leitura manual evitado), comparando o cenário com modelo e com o baseline (responsável: ____)
- [ ] **Percepção de preço:** analisar as avaliações que mencionam preço e valor, para ligar o modelo ao problema do projeto (responsável: ____)
- [ ] Demonstração de uso real e, se houver tempo, mini deploy (Streamlit ou Gradio) como bônus (responsável: ____)
- [ ] Relatório técnico, com divisão de tarefas, citação das ferramentas de IA usadas, licença e fonte dos dados (responsável: ____)

## 10. Limitações

- **Domínio misto:** o `concatenated.csv` mistura avaliações de varejo, aplicativos e filmes, enquanto o problema do projeto é o varejo online. Termos como "filmaço" aparecem entre os mais positivos, o que mostra a influência dos filmes.
- **Rótulos ausentes:** 399.928 registros (14%) não têm `polarity` e foram excluídos. O motivo da ausência não foi verificado.
- **Amostra:** o experimento usa 60.000 dos 1.963.776 registros tratados.
- **Teste possivelmente otimista:** remover duplicados e textos conflitantes elimina exemplos ambíguos também do teste, o que pode deixar a avaliação mais fácil que o uso real.
- **Desbalanceamento:** com 19,3% de negativas, ainda há 486 avaliações negativas no teste classificadas como positivas.
- **Modelos comparados:** só a Regressão Logística teve hiperparâmetros ajustados; Naive Bayes e SVM Linear usam os valores padrão. A diferença entre Regressão Logística e SVM é pequena.
- **Naive Bayes:** o F1-macro na validação cruzada (0,6892) ficou bem abaixo do teste (0,7442). A causa não foi investigada.
- **Um único teste:** os resultados vêm de uma única divisão treino/teste, sem intervalo de confiança.
- **Representação de texto:** TF-IDF não entende ironia nem contexto. O termo "só não", por exemplo, aparece com peso positivo, o que mostra associações ambíguas.
- **Viés e impacto de negócio:** ainda não analisados (veja a seção 9).

## 11. Uso de ferramentas de IA e referências

Conforme as regras da UC (seção 11 do enunciado), o uso de ferramentas de IA como apoio deve ser citado, e a equipe precisa dominar e saber explicar tudo o que entrega. Este projeto usou o assistente **Claude (Anthropic)** como apoio na estrutura do código, na organização do repositório e na documentação.

- Dataset: [fredericods/ptbr-sentiment-analysis-datasets (Kaggle)](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets)
- Bibliotecas: [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [joblib](https://joblib.readthedocs.io/)
