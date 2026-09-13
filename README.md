# ATP Serve Analysis

Projeto de modelagem estatística (análise descritiva, testes de hipótese e regressão linear) aplicado a dados de partidas do ATP Tour, com foco em entender o que explica o **desempenho de saque** dos jogadores (aces, % de pontos de saque ganhos).

Resultado final: notebook de análise + dashboard interativo em Streamlit.

## Estrutura do projeto

```
.
├── data/
│   ├── raw/          # CSVs baixados via src/download_data.py (não editar à mão)
│   └── processed/    # dados tratados, gerados pelos notebooks
├── notebooks/        # notebooks de análise (EDA, testes de hipótese, regressão)
├── dashboard/         # app do dashboard interativo (Streamlit)
├── src/               # scripts auxiliares (ex: download dos dados)
├── requirements.txt
└── README.md
```

## Sobre os dados

Os dados vêm originalmente do repositório `tennis_atp` do Jeff Sackmann, mas esse repositório **não está mais público no GitHub**. Por isso estamos usando um mirror ativo e mantido diariamente, com o mesmo formato de colunas:

- Fonte: https://github.com/Tennismylife/TML-Database
- Cobertura usada no projeto: partidas de **2010 a 2024** (estatísticas de saque antes disso têm muitos dados ausentes)

Os dados **não estão no `.gitignore`** — os CSVs brutos ficam versionados em `data/raw/` para o projeto rodar sem depender de uma URL externa ficar no ar (o que já aconteceu uma vez com a fonte original).

## Como rodar o projeto (colaborador)

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd atp-serve-analysis
```

### 2. Criar e ativar o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar os dados (só necessário se `data/raw/` estiver vazia)

```bash
python src/download_data.py
```

### 5. Rodar os notebooks

Abra a pasta `notebooks/` no Jupyter Lab ou VS Code e rode as células na ordem numérica dos arquivos.

### 6. Rodar o dashboard

```bash
streamlit run dashboard/app.py
```

## Fluxo de colaboração

- Cada etapa do projeto (EDA/testes de hipótese, regressão, dashboard) tem seu próprio notebook em `notebooks/`, para evitar conflitos de merge.
- Trabalhe em uma branch por etapa (`git checkout -b nome-da-etapa`) e abra um Pull Request para revisão mútua antes do merge na `main`.
- O código final entregue não deve conter comentários (`#`) — a documentação do raciocínio fica em células de markdown no notebook.
