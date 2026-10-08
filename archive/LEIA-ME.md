# Pasta `archive/` — dados (não versionados)

Esta pasta deve conter o dataset do Kaggle. Os arquivos CSV **não estão no GitHub**: são muito grandes (milhões de linhas) e devem ser baixados da fonte original por cada pessoa.

**Fonte:** https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

O `app.py` procura exatamente este arquivo:

```
archive/concatenated.csv
```

Só o `concatenated.csv` é usado. Ele já reúne as demais bases (`b2w.csv`, `buscape.csv`, `olist.csv`, `utlc_apps.csv`, `utlc_movies.csv`). **Não use o consolidado junto com os arquivos individuais**, pois isso duplicaria avaliações e distorceria os resultados. Os arquivos individuais podem ficar na pasta (o Git os ignora) ou ser apagados.

## Opção 1 — Download manual (mais simples)

1. Entre no Kaggle com sua conta e abra o link acima.
2. Clique em **Download**. Será baixado um arquivo `archive.zip`.
3. Extraia o conteúdo dentro desta pasta, de modo que o caminho final seja `archive/concatenated.csv` (e não `archive/archive/concatenated.csv`).

## Opção 2 — Kaggle CLI (opcional)

Exige credenciais do Kaggle configuradas conforme a documentação oficial: https://www.kaggle.com/docs/api

```powershell
py -m pip install kaggle
kaggle datasets download -d fredericods/ptbr-sentiment-analysis-datasets -p archive --unzip
```

## Conferir se deu certo

Na raiz do projeto, com o ambiente virtual ativado:

```powershell
Get-ChildItem archive
py -c "import pandas as pd; print(pd.read_csv('archive/concatenated.csv', nrows=3).columns.tolist())"
```

A lista de colunas deve incluir `review_text` e `polarity`.

## Licença e citação

Consulte a licença na página do dataset no Kaggle e cite a fonte no relatório técnico. Não redistribua os arquivos neste repositório.
