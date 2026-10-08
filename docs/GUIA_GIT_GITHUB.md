# Guia de Git e GitHub para o grupo

Passo a passo para publicar o projeto e trabalhar em equipe, no Windows com PowerShell.

## 1. Preparação (cada pessoa, uma vez)

1. Instale o [Git para Windows](https://git-scm.com/download/win) e crie uma conta no [GitHub](https://github.com/).
2. Configure seu nome e e-mail. Se não quiser expor seu e-mail pessoal, use o endereço *noreply* que o GitHub oferece em *Settings > Emails*:

   ```powershell
   git config --global user.name "Seu Nome"
   git config --global user.email "seu-email-ou-noreply@exemplo.com"
   ```

## 2. Publicar o projeto (quem cria o repositório)

1. No GitHub, clique em **New repository**. Dê um nome (por exemplo, `analise-sentimentos-ptbr`) e **não** marque "Add a README", ".gitignore" nem "license", porque o projeto já tem esses arquivos.
2. Escolha **Private** e convide os colegas em *Settings > Collaborators > Add people*. Se a professora pedir um link aberto, escolha **Public**.
3. Antes de publicar, confirme que a pasta `resultados_referencia/` tem os 8 arquivos listados no README.
4. No PowerShell, dentro da pasta do projeto:

   ```powershell
   git init
   git branch -M main
   git add .
   git status
   ```

5. **Confira o `git status` antes de continuar.** Estes itens **não** devem aparecer: `.venv`, `resultados/`, `archive/*.csv`, `*.joblib`, `app_anterior.py`. Para ver tudo o que será enviado:

   ```powershell
   git ls-files
   ```

6. Se estiver tudo certo:

   ```powershell
   git commit -m "Primeira versão: etapas 3 a 9 (EDA a explicabilidade)"
   git remote add origin https://github.com/USUARIO/REPOSITORIO.git
   git push -u origin main
   ```

   No primeiro `push`, o Git abre uma janela do navegador para entrar no GitHub.

**Se um arquivo grande entrar por engano** (o GitHub recusa arquivos acima de 100 MB): rode `git rm --cached caminho\do\arquivo`, confirme que ele está no `.gitignore` e refaça o commit. Se o commit já tiver sido feito, peça ajuda antes de continuar.

## 3. Entrar no projeto (colegas)

```powershell
git clone https://github.com/USUARIO/REPOSITORIO.git
cd REPOSITORIO
```

Depois siga a seção 6 do `README.md` (ambiente virtual, dependências e dataset).

## 4. Rotina de trabalho em grupo

Use uma **branch por tarefa**, para ninguém sobrescrever o trabalho dos outros:

```powershell
git switch main
git pull                                   # traz as novidades do grupo
git switch -c analise-vies                 # cria sua branch (use um nome descritivo)

# ... trabalhe, teste ...

git add .
git status                                 # confira o que vai no commit
git commit -m "Descreve em uma frase o que mudou"
git push -u origin analise-vies
```

No GitHub, abra um **Pull Request** da sua branch para a `main`, peça para alguém revisar e faça o *merge*. Depois, volte para a `main` e atualize:

```powershell
git switch main
git pull
```

Boas práticas:
- Rode `git pull` antes de começar a trabalhar.
- Combine no grupo quem mexe em qual arquivo. Evite editar o `app.py` em duas pessoas ao mesmo tempo; use arquivos ou notebooks separados para novas análises.
- Nunca versione dados, `.venv`, modelos `.joblib`, senhas ou o `kaggle.json`.
- Mensagens de commit curtas e claras.

## 5. Problemas comuns

| Mensagem | O que fazer |
|---|---|
| `remote origin already exists` | Rode `git remote set-url origin https://github.com/USUARIO/REPOSITORIO.git` |
| `failed to push some refs` | Alguém enviou mudanças antes. Rode `git pull` e tente `git push` de novo |
| `Permission denied` / erro 403 | Você ainda não foi adicionado como colaborador, ou entrou com outra conta |
| `.\.venv\Scripts\Activate.ps1 cannot be loaded` | Rode `Set-ExecutionPolicy -Scope Process Bypass` e tente de novo |
