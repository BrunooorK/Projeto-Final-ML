# -*- coding: utf-8 -*-
"""
Projeto Final UC9 - Análise de sentimentos em PT-BR (versão local: VS Code / Windows)

Etapas cobertas
  3  EDA documentada            -> estatísticas + gráficos (dados brutos)
  4  Tratamento de dados        -> ausentes, duplicados, limpeza, outliers, codificação
  5  Baseline                   -> DummyClassifier (classe majoritária)
  6  Modelagem (3 algoritmos)   -> Regressão Logística, Naive Bayes, SVM Linear (TF-IDF)
  7  Validação e ajuste         -> treino/teste estratificado + validação cruzada + GridSearchCV
  8  Métricas justificadas      -> acurácia, precisão/recall/F1 por classe, F1-macro, matriz
  9  Explicabilidade            -> coeficientes dos termos + explicação de frases

Como rodar (PowerShell, dentro da pasta do projeto):
    py -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    py -m pip install -r requirements.txt
    py app.py

Entrada : archive/concatenated.csv  (ÚNICA fonte; não some com b2w.csv, olist.csv etc.)
Saída   : pasta resultados/ (gráficos .png, métricas .csv, modelo .joblib, relatorio.md)
"""

from __future__ import annotations

import platform
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
try:  # evita erro de acentos se a saída do terminal for redirecionada
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# =============================================================================
# CONFIGURAÇÃO  (altere aqui; o resto do código não precisa mudar)
# =============================================================================
PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA_PROJETO / "archive" / "concatenated.csv"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

COLUNA_TEXTO = "review_text"   # entrada
COLUNA_ROTULO = "polarity"     # alvo binário: 0 = negativo, 1 = positivo

MAX_LINHAS = 60000             # amostra estratificada (None = usar tudo; exige muita RAM)
SEMENTE = 42                   # reprodutibilidade
TAMANHO_TESTE = 0.20           # 80% treino / 20% teste
CV_FOLDS = 3                   # validação cruzada (sempre só no treino)
GRADE_C = [0.1, 1, 5, 10]      # hiperparâmetro C da Regressão Logística
N_JOBS = 2                     # paralelismo; use 1 se o PC ficar sem memória

DEDUP_APOS_NORMALIZAR = True   # remove também textos iguais depois da limpeza
REMOVER_CONFLITOS = True       # remove textos iguais com rótulos diferentes
INSPECIONAR_KFOLD = True       # só imprime/documenta kfold_polarity; NÃO é usada no treino

NOMES = {0: "negativo", 1: "positivo"}
FRASES_TESTE = [  # (frase, rótulo esperado)
    ("Produto excelente, chegou antes do prazo e superou minhas expectativas", 1),
    ("Péssimo, chegou quebrado e ninguém resolveu meu problema", 0),
    ("Muito caro para o que entrega, não vale o preço que paguei", 0),
    ("Ótima qualidade e preço justo, recomendo", 1),
]

# Cores (paleta categórica validada: slots 1-3) e texto
COR_1, COR_2, COR_3 = "#2a78d6", "#eb6834", "#1baf7a"
COR_TEXTO, COR_GRADE = "#0b0b0b", "#d9d8d2"


# =============================================================================
# Funções auxiliares
# =============================================================================
def titulo(txt: str) -> None:
    print("\n" + "=" * 70 + f"\n{txt}\n" + "=" * 70)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def estilo_eixos(ax) -> None:
    """Gráfico limpo: sem moldura superior/direita, grade discreta."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)


RE_URL = re.compile(r"http\S+|www\.\S+")
RE_NAO_LETRA = re.compile(r"[^a-záàâãéêíóôõúüç\s]")  # mantém letras acentuadas
RE_ESPACOS = re.compile(r"\s+")


def limpar_texto(t: str) -> str:
    """Limpeza de UMA frase (usada nas frases de teste). Igual a limpar_serie."""
    t = str(t).lower()
    t = RE_URL.sub(" ", t)
    t = RE_NAO_LETRA.sub(" ", t)
    return RE_ESPACOS.sub(" ", t).strip()


def limpar_serie(s: pd.Series) -> pd.Series:
    """Mesma limpeza, vetorizada (rápida para milhões de linhas)."""
    return (s.astype(str).str.lower()
            .str.replace(RE_URL, " ", regex=True)
            .str.replace(RE_NAO_LETRA, " ", regex=True)
            .str.replace(RE_ESPACOS, " ", regex=True)
            .str.strip())


def novo_tfidf() -> TfidfVectorizer:
    # O TF-IDF fica DENTRO do Pipeline: o vocabulário e os pesos são aprendidos
    # só com o treino de cada divisão, nunca com o teste (evita vazamento).
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                           sublinear_tf=True)


def calcular_metricas(nome: str, y_real, y_prev) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_real, y_prev, labels=[0, 1], zero_division=0)
    return {
        "modelo": nome,
        "acuracia": accuracy_score(y_real, y_prev),
        "acuracia_balanceada": balanced_accuracy_score(y_real, y_prev),
        "precisao_neg": p[0], "recall_neg": r[0], "f1_neg": f[0],
        "precisao_pos": p[1], "recall_pos": r[1], "f1_pos": f[1],
        "f1_macro": f1_score(y_real, y_prev, average="macro", zero_division=0),
    }


def coeficientes_do_modelo(modelo: Pipeline):
    """Peso de cada termo a favor do POSITIVO (>0) ou do NEGATIVO (<0)."""
    termos = np.array(modelo.named_steps["tfidf"].get_feature_names_out())
    clf = modelo.named_steps["clf"]
    if hasattr(clf, "coef_"):                      # Regressão Logística e SVM linear
        coef = clf.coef_[0]
    else:                                          # Naive Bayes
        coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    return termos, np.asarray(coef)


def explicar_frase(modelo: Pipeline, frase: str, termos_coef, top: int = 5):
    """Contribuição de cada termo da frase (valor TF-IDF x coeficiente)."""
    termos, coef = termos_coef
    vet = modelo.named_steps["tfidf"].transform([limpar_texto(frase)]).tocoo()
    contrib = vet.data * coef[vet.col]
    ordem = np.argsort(-np.abs(contrib))[:top]
    return [(termos[vet.col[i]], float(contrib[i])) for i in ordem]


def pontuar(rotulo: str, t0: float) -> None:
    print(f"   [{rotulo}] concluído em {time.time() - t0:.1f}s")


# =============================================================================
# 1. CARREGAR OS DADOS (somente concatenated.csv)
# =============================================================================
titulo("1. CARREGANDO O DATASET")
inicio_total = time.time()
PASTA_RESULTADOS.mkdir(exist_ok=True)

if not ARQUIVO_DADOS.exists():
    raise SystemExit(
        f"Arquivo não encontrado: {ARQUIVO_DADOS}\n"
        "Confira se existe a pasta 'archive' ao lado do app.py com o concatenated.csv.")

cabecalho = pd.read_csv(ARQUIVO_DADOS, nrows=0).columns.tolist()
faltando = [c for c in (COLUNA_TEXTO, COLUNA_ROTULO) if c not in cabecalho]
if faltando:
    raise SystemExit(f"Colunas ausentes no CSV: {faltando}\nColunas disponíveis: {cabecalho}")

t0 = time.time()
df = pd.read_csv(ARQUIVO_DADOS, usecols=[COLUNA_TEXTO, COLUNA_ROTULO])
df = df.rename(columns={COLUNA_TEXTO: "texto", COLUNA_ROTULO: "rotulo"})
print("Arquivo:", ARQUIVO_DADOS)
print("Registros brutos:", f"{len(df):,}".replace(",", "."))
pontuar("leitura", t0)

# kfold_polarity: só inspeção. O significado NÃO foi confirmado, então não é usada.
info_kfold = "coluna kfold_polarity não encontrada no arquivo."
if INSPECIONAR_KFOLD and "kfold_polarity" in cabecalho:
    kf = pd.read_csv(ARQUIVO_DADOS, usecols=["kfold_polarity"])["kfold_polarity"]
    contagem = kf.value_counts(dropna=False).sort_index().head(12)
    info_kfold = ("valores (até 12 primeiros):\n" + contagem.to_string()
                  + "\nNão usada: o significado não foi confirmado; usamos divisão própria.")
    print("\nkfold_polarity:\n" + info_kfold)
    del kf

# Validação do alvo: só pode existir 0, 1 (e ausentes)
df["rotulo"] = pd.to_numeric(df["rotulo"], errors="coerce")
valores = set(df["rotulo"].dropna().unique().tolist())
if not valores or not valores.issubset({0.0, 1.0}):
    raise SystemExit(f"'{COLUNA_ROTULO}' deveria ter só 0/1, mas tem: {sorted(valores)[:10]}")

# =============================================================================
# 2. EDA (etapa 3) - dados brutos
# =============================================================================
titulo("2. EDA (ETAPA 3) - DADOS BRUTOS")
eda = {}
eda["registros"] = len(df)
eda["ausentes_texto"] = int(df["texto"].isna().sum())
eda["ausentes_rotulo"] = int(df["rotulo"].isna().sum())
eda["duplicados_texto"] = int(df["texto"].duplicated().sum())
dist_bruta = df["rotulo"].value_counts(normalize=True).sort_index()
eda["pos_bruto"] = float(dist_bruta.get(1.0, 0))
eda["neg_bruto"] = float(dist_bruta.get(0.0, 0))
df["tamanho"] = df["texto"].fillna("").astype(str).str.split().str.len()
desc = df["tamanho"].describe()

print("Ausentes -> texto:", eda["ausentes_texto"], "| rótulo:", eda["ausentes_rotulo"])
print("Textos duplicados:", eda["duplicados_texto"])
print("Classes (entre rótulos presentes): positivo", pct(eda["pos_bruto"]),
      "| negativo", pct(eda["neg_bruto"]))
print("Tamanho dos textos (palavras):\n", desc.round(1).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
contagem = df["rotulo"].value_counts().sort_index()
barras = ax[0].bar([NOMES[int(i)] for i in contagem.index], contagem.values,
                   color=[COR_2, COR_1], width=0.55)
for b, v in zip(barras, contagem.values):
    ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,}".replace(",", "."),
               ha="center", va="bottom", fontsize=9, color=COR_TEXTO)
ax[0].set_title("Avaliações por classe (rótulos presentes)", fontsize=11, loc="left")
ax[0].yaxis.set_visible(False)
estilo_eixos(ax[0])
ax[0].grid(False)
corte = df["tamanho"].quantile(0.99)
ax[1].hist(df["tamanho"].clip(upper=corte), bins=40, color=COR_1)
ax[1].set_title(f"Tamanho dos textos (palavras; cortado no percentil 99 = {corte:.0f})",
                fontsize=11, loc="left")
ax[1].set_xlabel("palavras")
estilo_eixos(ax[1])
plt.tight_layout()
plt.savefig(PASTA_RESULTADOS / "eda_graficos.png", dpi=130)
plt.close()

# =============================================================================
# 3. TRATAMENTO (etapa 4)
# =============================================================================
titulo("3. TRATAMENTO DOS DADOS (ETAPA 4)")
t0 = time.time()
trat = {"brutos": len(df)}

df = df.dropna(subset=["texto", "rotulo"]).copy()               # ausentes
trat["apos_ausentes"] = len(df)
df = df.drop_duplicates(subset=["texto"])                       # duplicados exatos
trat["apos_duplicados"] = len(df)
df["rotulo"] = df["rotulo"].astype(int)

df["texto_limpo"] = limpar_serie(df["texto"])                   # normalização
df = df[df["texto_limpo"].str.len() > 2]                        # textos vazios/curtíssimos
trat["apos_limpeza"] = len(df)

# Duplicados que só aparecem DEPOIS de normalizar (ex.: "Ótimo!" e "ótimo")
trat["dup_normalizados"] = int(df.duplicated(subset="texto_limpo").sum())
mask_dup = df.duplicated(subset="texto_limpo", keep=False)
rotulos_por_texto = df[mask_dup].groupby("texto_limpo")["rotulo"].nunique()
textos_conflito = rotulos_por_texto[rotulos_por_texto > 1].index
trat["textos_conflito"] = len(textos_conflito)
print("Textos iguais após normalizar:", trat["dup_normalizados"])
print("Textos iguais com rótulos CONFLITANTES:", trat["textos_conflito"])
if REMOVER_CONFLITOS and len(textos_conflito):
    df = df[~df["texto_limpo"].isin(textos_conflito)]
if DEDUP_APOS_NORMALIZAR:
    df = df.drop_duplicates(subset="texto_limpo")
trat["apos_conflitos_dedup"] = len(df)

print(f"Ausentes removidos: {trat['brutos'] - trat['apos_ausentes']:,}".replace(",", "."))
print(f"Duplicados exatos removidos: {trat['apos_ausentes'] - trat['apos_duplicados']:,}".replace(",", "."))
print(f"Textos curtos/vazios removidos: {trat['apos_duplicados'] - trat['apos_limpeza']:,}".replace(",", "."))
print(f"Base tratada: {trat['brutos']:,} -> {trat['apos_conflitos_dedup']:,} registros".replace(",", "."))
print("Outliers de tamanho: NÃO removidos (o gráfico só foi cortado no percentil 99).")
print("Escalonamento: não usado (o TF-IDF já normaliza cada vetor).")

if MAX_LINHAS and len(df) > MAX_LINHAS:
    df, _ = train_test_split(df, train_size=MAX_LINHAS, random_state=SEMENTE,
                             stratify=df["rotulo"])
    print(f"Amostra estratificada de {MAX_LINHAS:,} linhas.".replace(",", "."))
trat["amostra"] = len(df)
dist_amostra = df["rotulo"].value_counts(normalize=True).sort_index()
print("Classes na amostra:", {NOMES[int(k)]: pct(v) for k, v in dist_amostra.items()})

X_treino, X_teste, y_treino, y_teste = train_test_split(
    df["texto_limpo"], df["rotulo"], test_size=TAMANHO_TESTE,
    random_state=SEMENTE, stratify=df["rotulo"])
print(f"Treino: {len(X_treino):,} | Teste: {len(X_teste):,}".replace(",", "."))
print("Teste -> negativos:", int((y_teste == 0).sum()), "| positivos:", int((y_teste == 1).sum()))
pontuar("tratamento", t0)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEMENTE)

# =============================================================================
# 4. BASELINE (etapa 5)
# =============================================================================
titulo("4. BASELINE (ETAPA 5)")
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_treino, y_treino)
cv_baseline = cross_val_score(baseline, X_treino, y_treino, cv=cv, scoring="f1_macro")
print("O baseline sempre responde:", NOMES[int(baseline.classes_[np.argmax(baseline.class_prior_)])])
print("Qualquer modelo útil precisa ganhar dele, principalmente no F1-macro.")

# =============================================================================
# 5. MODELAGEM + VALIDAÇÃO (etapas 6 e 7)
# =============================================================================
titulo("5. MODELAGEM E VALIDAÇÃO (ETAPAS 6 E 7)")
modelos = {
    "Regressão Logística": Pipeline([
        ("tfidf", novo_tfidf()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Naive Bayes": Pipeline([
        ("tfidf", novo_tfidf()),
        ("clf", MultinomialNB())]),
    "SVM Linear": Pipeline([
        ("tfidf", novo_tfidf()),
        ("clf", LinearSVC(class_weight="balanced"))]),
}

cv_resumo = {"Baseline (classe majoritária)": (cv_baseline.mean(), cv_baseline.std())}

# 5a. Ajuste de hiperparâmetros (Regressão Logística) - SÓ com o treino
t0 = time.time()
print(f"Ajustando C da Regressão Logística {GRADE_C} com {CV_FOLDS}-fold (só no treino)...")
busca = GridSearchCV(modelos["Regressão Logística"], {"clf__C": GRADE_C},
                     cv=cv, scoring="f1_macro", n_jobs=N_JOBS, refit=True)
busca.fit(X_treino, y_treino)
modelos["Regressão Logística"] = busca.best_estimator_
melhor_c = busca.best_params_["clf__C"]
i = busca.best_index_
cv_resumo["Regressão Logística"] = (busca.cv_results_["mean_test_score"][i],
                                    busca.cv_results_["std_test_score"][i])
grade = pd.DataFrame({"C": [p["clf__C"] for p in busca.cv_results_["params"]],
                      "f1_macro_cv_media": busca.cv_results_["mean_test_score"],
                      "f1_macro_cv_desvio": busca.cv_results_["std_test_score"]})
grade.to_csv(PASTA_RESULTADOS / "grade_hiperparametros.csv", index=False, encoding="utf-8-sig")
print(grade.round(4).to_string(index=False))
print(f"Melhor C = {melhor_c} | F1-macro (validação cruzada) = {cv_resumo['Regressão Logística'][0]:.4f}")
pontuar("GridSearchCV", t0)

# 5b. Validação cruzada dos outros dois (mesmos folds) e treino final
for nome in ("Naive Bayes", "SVM Linear"):
    t0 = time.time()
    notas = cross_val_score(modelos[nome], X_treino, y_treino, cv=cv,
                            scoring="f1_macro", n_jobs=N_JOBS)
    cv_resumo[nome] = (notas.mean(), notas.std())
    modelos[nome].fit(X_treino, y_treino)
    print(f"{nome}: F1-macro CV = {notas.mean():.4f} (+/- {notas.std():.4f})")
    pontuar(nome, t0)

# Escolha do melhor modelo pela VALIDAÇÃO CRUZADA (o teste não participa da escolha)
melhor_nome = max(modelos, key=lambda n: cv_resumo[n][0])
print(f"\nModelo escolhido pela validação cruzada: {melhor_nome}")

# =============================================================================
# 6. AVALIAÇÃO NO TESTE (etapa 8) - o teste é usado uma única vez, no fim
# =============================================================================
titulo("6. MÉTRICAS NO TESTE (ETAPA 8)")
previsoes = {"Baseline (classe majoritária)": baseline.predict(X_teste)}
for nome, m in modelos.items():
    previsoes[nome] = m.predict(X_teste)

linhas = []
for nome, prev in previsoes.items():
    linha = calcular_metricas(nome, y_teste, prev)
    linha["f1_macro_cv_media"], linha["f1_macro_cv_desvio"] = cv_resumo[nome]
    linhas.append(linha)
tabela = pd.DataFrame(linhas)
ordem = ["modelo", "acuracia", "acuracia_balanceada", "precisao_neg", "recall_neg", "f1_neg",
         "precisao_pos", "recall_pos", "f1_pos", "f1_macro", "f1_macro_cv_media",
         "f1_macro_cv_desvio"]
tabela = tabela[ordem].sort_values("f1_macro", ascending=False).round(4)
tabela.to_csv(PASTA_RESULTADOS / "comparacao_modelos.csv", index=False, encoding="utf-8-sig")
print(tabela[["modelo", "acuracia", "recall_neg", "f1_neg", "f1_pos", "f1_macro"]]
      .to_string(index=False))

linha_base = tabela[tabela["modelo"].str.startswith("Baseline")].iloc[0]
linha_melhor = tabela[tabela["modelo"] == melhor_nome].iloc[0]
print(f"\nGanho do {melhor_nome} sobre o baseline: "
      f"F1-macro {linha_base['f1_macro']:.4f} -> {linha_melhor['f1_macro']:.4f} | "
      f"acurácia {linha_base['acuracia']:.4f} -> {linha_melhor['acuracia']:.4f}")

# Gráfico de comparação (acurácia x F1-macro x F1 negativo)
ordem_g = tabela["modelo"].tolist()
x = np.arange(len(ordem_g))
larg = 0.26
fig, ax = plt.subplots(figsize=(10, 4.8))
for k, (col, rot, cor) in enumerate([("acuracia", "Acurácia", COR_1),
                                     ("f1_macro", "F1-macro", COR_2),
                                     ("f1_neg", "F1 da classe negativa", COR_3)]):
    vals = tabela.set_index("modelo").loc[ordem_g, col].values
    bars = ax.bar(x + (k - 1) * (larg + 0.02), vals, larg, label=rot, color=cor)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                va="bottom", fontsize=8, color=COR_TEXTO)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" (classe majoritária)", "\n(classe majoritária)") for n in ordem_g],
                   fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_title("Comparação no teste: a acurácia sozinha esconde o desempenho na classe negativa",
             fontsize=11, loc="left")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
estilo_eixos(ax)
plt.tight_layout()
plt.savefig(PASTA_RESULTADOS / "comparacao_modelos.png", dpi=130)
plt.close()

# Matriz de confusão do melhor modelo (contagens e proporção por classe real)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ConfusionMatrixDisplay.from_predictions(y_teste, previsoes[melhor_nome], ax=ax[0],
                                        display_labels=list(NOMES.values()), cmap="Blues",
                                        colorbar=False)
ConfusionMatrixDisplay.from_predictions(y_teste, previsoes[melhor_nome], ax=ax[1],
                                        display_labels=list(NOMES.values()), cmap="Blues",
                                        normalize="true", values_format=".2f", colorbar=False)
ax[0].set_title("Contagens", fontsize=11)
ax[1].set_title("Proporção por classe real (diagonal = recall)", fontsize=11)
fig.suptitle(f"Matriz de confusão - {melhor_nome}", fontsize=12)
plt.tight_layout()
plt.savefig(PASTA_RESULTADOS / "matriz_confusao.png", dpi=130)
plt.close()

# Erros do melhor modelo (para a análise crítica do relatório)
modelo_final = modelos[melhor_nome]
erros_idx = X_teste.index[previsoes[melhor_nome] != y_teste.values]
erros = df.loc[erros_idx, ["texto", "rotulo"]].rename(columns={"rotulo": "rotulo_real"})
if len(erros):
    erros["previsto"] = modelo_final.predict(df.loc[erros_idx, "texto_limpo"])
else:
    erros["previsto"] = []
erros.sample(min(100, len(erros)), random_state=SEMENTE).to_csv(
    PASTA_RESULTADOS / "exemplos_de_erros.csv", index=False, encoding="utf-8-sig")
print(f"Erros no teste: {len(erros)} (100 exemplos em exemplos_de_erros.csv)")

# =============================================================================
# 7. EXPLICABILIDADE (etapa 9)
# =============================================================================
titulo("7. EXPLICABILIDADE (ETAPA 9)")
termos_coef = coeficientes_do_modelo(modelo_final)
termos, coef = termos_coef
ordem_coef = np.argsort(coef)
top_neg = [(termos[j], float(coef[j])) for j in ordem_coef[:15]]
top_pos = [(termos[j], float(coef[j])) for j in ordem_coef[::-1][:15]]
pd.DataFrame({"termo": termos[ordem_coef], "peso": coef[ordem_coef]}).to_csv(
    PASTA_RESULTADOS / "coeficientes_termos.csv", index=False, encoding="utf-8-sig")
print(f"Explicando: {melhor_nome} (peso > 0 puxa para POSITIVO, < 0 para NEGATIVO)")
print("Termos mais negativos:", [t for t, _ in top_neg[:12]])
print("Termos mais positivos:", [t for t, _ in top_pos[:12]])

fig, ax = plt.subplots(figsize=(10, 6.2))
nomes_t = [t for t, _ in top_neg[::-1]] + [t for t, _ in top_pos[::-1]]
pesos_t = [w for _, w in top_neg[::-1]] + [w for _, w in top_pos[::-1]]
ax.barh(range(len(nomes_t)), pesos_t,
        color=[COR_2 if w < 0 else COR_1 for w in pesos_t], height=0.65)
ax.set_yticks(range(len(nomes_t)))
ax.set_yticklabels(nomes_t, fontsize=9)
ax.axvline(0, color=COR_TEXTO, linewidth=0.8)
ax.set_xlabel("peso do termo no modelo   (← negativo | positivo →)")
ax.set_title(f"Termos que mais pesam - {melhor_nome} (laranja = negativo, azul = positivo)",
             fontsize=11, loc="left")
estilo_eixos(ax)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=COR_GRADE, linewidth=0.6)
plt.tight_layout()
plt.savefig(PASTA_RESULTADOS / "termos_mais_importantes.png", dpi=130)
plt.close()

print("\nFrases de teste (conferência de bom senso):")
explicacoes = []
acertos_frases = 0
intercepto = float(getattr(modelo_final.named_steps["clf"], "intercept_", [0.0])[0]) \
    if hasattr(modelo_final.named_steps["clf"], "intercept_") else None
for frase, esperado in FRASES_TESTE:
    prev = int(modelo_final.predict([limpar_texto(frase)])[0])
    ok = prev == esperado
    acertos_frases += ok
    contrib = explicar_frase(modelo_final, frase, termos_coef)
    explicacoes.append((frase, esperado, prev, contrib))
    print(f"[{'OK  ' if ok else 'ERRO'}] previsto={NOMES[prev]:8} esperado={NOMES[esperado]:8} | {frase}")
    print("       termos que mais pesaram:", ", ".join(f"{t} ({w:+.2f})" for t, w in contrib))
print(f"Frases coerentes: {acertos_frases}/{len(FRASES_TESTE)}")

# =============================================================================
# 8. SALVAR MODELO E RELATÓRIO
# =============================================================================
titulo("8. SALVANDO ARQUIVOS")
joblib.dump(modelo_final, PASTA_RESULTADOS / "modelo_sentimentos.joblib")

L = []
L.append("# Relatório automático - Análise de sentimentos PT-BR\n")
L.append(f"Gerado por `app.py` em {time.strftime('%Y-%m-%d %H:%M')} "
         f"(Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
         f"pandas {pd.__version__}, numpy {np.__version__}).\n")
L.append("Todos os números abaixo vêm da execução deste script; nada foi digitado à mão.\n")
L.append("## Configuração\n")
L.append(f"- Fonte única: `archive/{ARQUIVO_DADOS.name}` (colunas `{COLUNA_TEXTO}` e `{COLUNA_ROTULO}`)")
L.append(f"- Amostra: {MAX_LINHAS} linhas estratificadas | semente {SEMENTE} | teste {pct(TAMANHO_TESTE)} | CV {CV_FOLDS} folds")
L.append(f"- Remover conflitos: {REMOVER_CONFLITOS} | dedup após normalizar: {DEDUP_APOS_NORMALIZAR}\n")
L.append("## Etapa 3 - EDA (dados brutos)\n")
L.append(f"- Registros: {eda['registros']:,} | ausentes: texto {eda['ausentes_texto']:,}, rótulo {eda['ausentes_rotulo']:,} | textos duplicados: {eda['duplicados_texto']:,}".replace(",", "."))
L.append(f"- Classes entre rótulos presentes: positivo {pct(eda['pos_bruto'])}, negativo {pct(eda['neg_bruto'])}")
L.append(f"- Tamanho (palavras): média {desc['mean']:.1f}, mediana {desc['50%']:.0f}, máximo {desc['max']:.0f}")
L.append("- Interpretação: as classes são desbalanceadas, então a acurácia sozinha engana; "
         "a cauda longa de tamanhos indica poucos textos muito grandes.\n")
L.append("![EDA](eda_graficos.png)\n")
L.append("## Etapa 4 - Tratamento\n")
L.append("| Passo | Registros depois |\n|---|---:|")
for rot, chave in [("Brutos", "brutos"), ("Sem ausentes (texto e rótulo)", "apos_ausentes"),
                   ("Sem textos duplicados exatos", "apos_duplicados"),
                   ("Após limpeza (remove textos com até 2 letras)", "apos_limpeza"),
                   ("Sem conflitos/duplicados após normalizar", "apos_conflitos_dedup"),
                   ("Amostra usada no experimento", "amostra")]:
    L.append(f"| {rot} | {trat[chave]:,} |".replace(",", "."))
L.append(f"\n- Textos iguais com rótulos conflitantes (após normalizar): {trat['textos_conflito']}")
L.append("- Ausentes: removidos (sem texto ou rótulo não há o que aprender; não se inventa sentimento).")
L.append("- Duplicados/conflitos: removidos para o mesmo texto não cair no treino e no teste (vazamento).")
L.append("- Limpeza: minúsculas, remoção de links e símbolos, acentos mantidos.")
L.append("- Outliers: textos longos NÃO foram removidos; o gráfico foi só cortado no percentil 99.")
L.append("- Codificação: rótulo já é 0/1; texto vira números com TF-IDF (1 e 2 palavras) dentro do Pipeline.")
L.append("- Escalonamento: não aplicado; o TF-IDF já normaliza cada vetor.")
L.append(f"- `kfold_polarity`: {info_kfold}\n")
L.append("## Etapa 5 - Baseline\n")
L.append("`DummyClassifier(strategy='most_frequent')`: sempre prevê a classe mais comum. "
         f"No teste: acurácia {linha_base['acuracia']:.4f} e F1-macro {linha_base['f1_macro']:.4f}.\n")
L.append("## Etapas 6 e 7 - Modelos e validação\n")
L.append("Todos usam TF-IDF + classificador no mesmo Pipeline. A escolha do melhor modelo usou "
         "a validação cruzada no treino; o teste só entrou na avaliação final.\n")
L.append(f"GridSearchCV na Regressão Logística (C em {GRADE_C}): melhor C = {melhor_c}, "
         f"F1-macro de validação = {cv_resumo['Regressão Logística'][0]:.4f}.\n")
L.append("Naive Bayes e SVM Linear usaram os parâmetros padrão do script (sem busca), "
         "o que favorece levemente a Regressão Logística na comparação.\n")
L.append(f"**Modelo escolhido: {melhor_nome}.**\n")
L.append("## Etapa 8 - Métricas (teste)\n")
L.append("```\n" + tabela.to_string(index=False) + "\n```")
L.append("\n**Por que estas métricas:** com poucas avaliações negativas, um modelo que só diz "
         "\"positivo\" já teria acurácia alta (veja o baseline). Por isso o foco é o **F1-macro** "
         "(média igual das duas classes) e o **recall da classe negativa**, que mostra quantas "
         "reclamações reais o modelo encontra. A precisão mostra quantos alertas de negativo "
         "são verdadeiros. A matriz de confusão detalha os tipos de erro.\n")
L.append("![Comparação](comparacao_modelos.png)\n\n![Matriz](matriz_confusao.png)\n")
L.append("## Etapa 9 - Explicabilidade\n")
L.append(f"Modelo explicado: {melhor_nome}. Pesos positivos puxam para *positivo*; negativos, para *negativo*.\n")
L.append("- Termos mais negativos: " + ", ".join(t for t, _ in top_neg[:12]))
L.append("- Termos mais positivos: " + ", ".join(t for t, _ in top_pos[:12]) + "\n")
L.append("![Termos](termos_mais_importantes.png)\n")
L.append("Limitações: os pesos mostram associações aprendidas nesta base, não causas; "
         "palavras como \"não\" mudam de sentido conforme o contexto, e o modelo não entende "
         "ironia nem comparações de preço sutis.\n")
L.append("### Frases de teste\n")
for frase, esperado, prev, contrib in explicacoes:
    marca = "OK" if prev == esperado else "ERRO"
    L.append(f"- **{marca}** previsto *{NOMES[prev]}* (esperado {NOMES[esperado]}): \"{frase}\" "
             "- termos: " + ", ".join(f"{t} ({w:+.2f})" for t, w in contrib))
L.append("\n## Arquivos gerados\n")
L.append("`eda_graficos.png`, `comparacao_modelos.png`, `matriz_confusao.png`, "
         "`termos_mais_importantes.png`, `comparacao_modelos.csv`, `grade_hiperparametros.csv`, "
         "`coeficientes_termos.csv`, `exemplos_de_erros.csv`, `modelo_sentimentos.joblib`.\n")
L.append("Aviso: só carregue arquivos `.joblib` gerados por você; arquivos de terceiros podem executar código.")
(PASTA_RESULTADOS / "relatorio.md").write_text("\n".join(L), encoding="utf-8")

print("Arquivos em:", PASTA_RESULTADOS)
for arq in sorted(PASTA_RESULTADOS.iterdir()):
    print("  -", arq.name)
print(f"\nTempo total: {(time.time() - inicio_total) / 60:.1f} min")