import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Configuração de diretório de cache do matplotlib para evitar warnings de permissão
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

# Configuração inicial da página Streamlit
st.set_page_config(
    page_title="ATP Serve Analysis",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo visual personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stat-badge-grass { background-color: #DEF7EC; color: #03543F; }
    .stat-badge-hard { background-color: #E1EFFE; color: #1E429F; }
    .stat-badge-clay { background-color: #FDE8E8; color: #9B1C1C; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CARREGAMENTO E CACHE DOS DADOS
# -------------------------------------------------------------
@st.cache_data
def carregar_dados():
    caminho_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "jogadores_saque.csv")
    if not os.path.exists(caminho_csv):
        caminho_csv = "data/processed/jogadores_saque.csv"
    
    df = pd.read_csv(caminho_csv)
    
    # Extrair ano a partir de tourney_date (formato AAAAMMDD)
    df['ano'] = df['tourney_date'].astype(str).str[:4].astype(int)
    
    # Rótulo legível para mão dominante
    df['mao_rotulo'] = df['hand'].map({'R': 'Destro (R)', 'L': 'Canhoto (L)', 'U': 'Indefinido (U)'}).fillna('Indefinido')
    
    # Filtro apenas superfícies principais para análises
    df['superficie_valida'] = df['surface'].isin(['Hard', 'Clay', 'Grass'])
    
    return df

@st.cache_resource
def treinar_modelo_regressao(df):
    df_modelo = df[df['superficie_valida']].dropna(subset=['pct_pontos_saque_ganhos', 'ht', 'age', 'surface', 'rank']).copy()
    
    treino, teste = train_test_split(df_modelo, test_size=0.2, random_state=42)
    
    formula = 'pct_pontos_saque_ganhos ~ ht + age + surface + np.log(rank)'
    modelo = smf.ols(formula=formula, data=treino).fit()
    
    previsoes_teste = modelo.predict(teste)
    residuos_teste = teste['pct_pontos_saque_ganhos'] - previsoes_teste
    mae = mean_absolute_error(teste['pct_pontos_saque_ganhos'], previsoes_teste)
    r2 = r2_score(teste['pct_pontos_saque_ganhos'], previsoes_teste)
    rmse = root_mean_squared_error(teste['pct_pontos_saque_ganhos'], previsoes_teste)
    
    return {
        'modelo': modelo,
        'treino': treino,
        'teste': teste,
        'previsoes_teste': previsoes_teste,
        'residuos_teste': residuos_teste,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }

# Carregar dados e modelo
df_completo = carregar_dados()
resultados_regressao = treinar_modelo_regressao(df_completo)

# -------------------------------------------------------------
# BARRA LATERAL (FILTROS)
# -------------------------------------------------------------
st.sidebar.title("🎾 Filtros Globais")

superficies_disponiveis = ['Todas', 'Hard', 'Clay', 'Grass']
superficie_selecionada = st.sidebar.selectbox("Superfície", superficies_disponiveis, index=0)

maos_disponiveis = ['Todas', 'Destro (R)', 'Canhoto (L)']
mao_selecionada = st.sidebar.selectbox("Mão Dominante", maos_disponiveis, index=0)

ano_min, ano_max = int(df_completo['ano'].min()), int(df_completo['ano'].max())
anos_selecionados = st.sidebar.slider("Período (Anos)", min_value=ano_min, max_value=ano_max, value=(ano_min, ano_max), step=1)

altura_min, altura_max = int(df_completo['ht'].dropna().min()), int(df_completo['ht'].dropna().max())
altura_selecionada = st.sidebar.slider("Altura do Jogador (cm)", min_value=altura_min, max_value=altura_max, value=(altura_min, altura_max))

rank_max_slider = st.sidebar.slider("Ranking Máximo (ATP)", min_value=10, max_value=500, value=300, step=10)

# Aplicar filtros
df_filtrado = df_completo.copy()

if superficie_selecionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['surface'] == superficie_selecionada]

if mao_selecionada != 'Todas':
    cod_mao = 'R' if 'Destro' in mao_selecionada else 'L'
    df_filtrado = df_filtrado[df_filtrado['hand'] == cod_mao]

df_filtrado = df_filtrado[
    (df_filtrado['ano'] >= anos_selecionados[0]) &
    (df_filtrado['ano'] <= anos_selecionados[1]) &
    (df_filtrado['ht'].isna() | ((df_filtrado['ht'] >= altura_selecionada[0]) & (df_filtrado['ht'] <= altura_selecionada[1]))) &
    (df_filtrado['rank'].isna() | (df_filtrado['rank'] <= rank_max_slider))
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Amostras filtradas:** `{len(df_filtrado):,}` de `{len(df_completo):,}` ({len(df_filtrado)/len(df_completo)*100:.1f}%)")

# -------------------------------------------------------------
# CABEÇALHO PRINCIPAL
# -------------------------------------------------------------
st.markdown('<div class="main-header">🎾 ATP Serve Analysis — Dashboard Estatístico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modelagem estatística, testes de hipótese e regressão linear sobre o desempenho de saque no circuito ATP (2010–2024).</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# ABAS DE NAVEGAÇÃO
# -------------------------------------------------------------
tab_eda, tab_hipoteses, tab_regressao, tab_simulador, tab_sobre = st.tabs([
    "📊 1. Análise Exploratória (EDA)",
    "🧪 2. Testes de Hipótese",
    "📈 3. Modelo de Regressão",
    "🎯 4. Simulador Interativo",
    "📖 5. Metodologia & Dados"
])

# -------------------------------------------------------------
# ABA 1: ANÁLISE EXPLORATÓRIA (EDA)
# -------------------------------------------------------------
with tab_eda:
    st.subheader("Visão Geral do Desempenho de Saque")
    
    # KPIs rápidos
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Pontos de Saque Ganhos",
            value=f"{df_filtrado['pct_pontos_saque_ganhos'].mean():.2f}%",
            help="(1stWon + 2ndWon) / svpt * 100"
        )
    with col2:
        st.metric(
            label="1º Saque Dentro",
            value=f"{df_filtrado['pct_primeiro_saque_dentro'].mean():.2f}%",
            help="1stIn / svpt * 100"
        )
    with col3:
        st.metric(
            label="Aces / 100 Saques",
            value=f"{df_filtrado['aces_por_100_saques'].mean():.2f}",
            help="ace / svpt * 100"
        )
    with col4:
        st.metric(
            label="Volume de Partidas",
            value=f"{len(df_filtrado):,}"
        )

    st.markdown("---")
    
    # Gráficos de Distribuição
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Distribuição de % Pontos de Saque Ganhos")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.histplot(df_filtrado['pct_pontos_saque_ganhos'], bins=40, kde=True, color="#2563EB", ax=ax)
        media_val = df_filtrado['pct_pontos_saque_ganhos'].mean()
        mediana_val = df_filtrado['pct_pontos_saque_ganhos'].median()
        ax.axvline(media_val, color='red', linestyle='--', label=f'Média: {media_val:.2f}%')
        ax.axvline(mediana_val, color='green', linestyle=':', label=f'Mediana: {mediana_val:.2f}%')
        ax.set_xlabel("% de Pontos de Saque Ganhos")
        ax.set_ylabel("Frequência")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with c2:
        st.markdown("#### % Pontos Ganhos por Superfície")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        df_sup = df_filtrado[df_filtrado['surface'].isin(['Grass', 'Hard', 'Clay'])]
        palette = {'Grass': '#10B981', 'Hard': '#3B82F6', 'Clay': '#EF4444'}
        sns.boxplot(data=df_sup, x='surface', y='pct_pontos_saque_ganhos', order=['Grass', 'Hard', 'Clay'], palette=palette, ax=ax)
        ax.set_xlabel("Superfície")
        ax.set_ylabel("% Pontos de Saque Ganhos")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("#### Relações Bivariadas com a Variável Alvo")
    
    col_g1, col_g2, col_g3 = st.columns(3)
    
    with col_g1:
        st.markdown("**Altura vs Saque**")
        df_sample = df_filtrado.dropna(subset=['ht', 'pct_pontos_saque_ganhos'])
        if len(df_sample) > 4000:
            df_sample = df_sample.sample(4000, random_state=42)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.regplot(data=df_sample, x='ht', y='pct_pontos_saque_ganhos', scatter_kws={'alpha':0.15, 's':12, 'color': '#2563EB'}, line_kws={'color':'red'}, ax=ax)
        corr_ht = df_filtrado['ht'].corr(df_filtrado['pct_pontos_saque_ganhos'])
        ax.set_title(f"Correlação r = {corr_ht:+.2f}", fontsize=10)
        ax.set_xlabel("Altura (cm)")
        ax.set_ylabel("% Saque Ganho")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_g2:
        st.markdown("**Idade vs Saque**")
        df_sample = df_filtrado.dropna(subset=['age', 'pct_pontos_saque_ganhos'])
        if len(df_sample) > 4000:
            df_sample = df_sample.sample(4000, random_state=42)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.regplot(data=df_sample, x='age', y='pct_pontos_saque_ganhos', scatter_kws={'alpha':0.15, 's':12, 'color': '#8B5CF6'}, line_kws={'color':'red'}, ax=ax)
        corr_age = df_filtrado['age'].corr(df_filtrado['pct_pontos_saque_ganhos'])
        ax.set_title(f"Correlação r = {corr_age:+.2f}", fontsize=10)
        ax.set_xlabel("Idade (anos)")
        ax.set_ylabel("% Saque Ganho")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_g3:
        st.markdown("**Ranking (Log) vs Saque**")
        df_sample = df_filtrado.dropna(subset=['rank', 'pct_pontos_saque_ganhos'])
        if len(df_sample) > 4000:
            df_sample = df_sample.sample(4000, random_state=42)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.regplot(data=df_sample, x='rank', y='pct_pontos_saque_ganhos', scatter_kws={'alpha':0.15, 's':12, 'color': '#EC4899'}, line_kws={'color':'red'}, ax=ax)
        ax.set_xscale('log')
        corr_rank = np.log(df_filtrado['rank'].dropna()).corr(df_filtrado.loc[df_filtrado['rank'].dropna().index, 'pct_pontos_saque_ganhos'])
        ax.set_title(f"Correlação log(rank) r = {corr_rank:+.2f}", fontsize=10)
        ax.set_xlabel("Ranking ATP (escala log)")
        ax.set_ylabel("% Saque Ganho")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# -------------------------------------------------------------
# ABA 2: TESTES DE HIPÓTESE
# -------------------------------------------------------------
with tab_hipoteses:
    st.subheader("Testes Estatísticos Formais e Inferência")
    st.markdown("Verificação de hipóteses centrais levantadas na análise exploratória com testes paramétricos e cálculo do tamanho de efeito (*Cohen's d*).")

    # TESTE 1: CANHOTOS VS DESTROS
    st.markdown("### 1. Mão Dominante: Canhotos vs. Destros")
    
    canhotos = df_completo[df_completo['hand'] == 'L']['pct_pontos_saque_ganhos'].dropna()
    destros = df_completo[df_completo['hand'] == 'R']['pct_pontos_saque_ganhos'].dropna()
    
    t_stat, p_val_t = stats.ttest_ind(canhotos, destros, equal_var=False)
    
    n1, n2 = len(canhotos), len(destros)
    pooled_sd = np.sqrt(((n1-1)*canhotos.std()**2 + (n2-1)*destros.std()**2) / (n1+n2-2))
    cohen_d_mao = (canhotos.mean() - destros.mean()) / pooled_sd

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        resumo_mao = pd.DataFrame({
            'Grupo': ['Canhotos (L)', 'Destros (R)'],
            'Nº Observações': [f"{n1:,}", f"{n2:,}"],
            'Média (% Pontos Ganhos)': [f"{canhotos.mean():.2f}%", f"{destros.mean():.2f}%"],
            'Desvio Padrão': [f"{canhotos.std():.2f}", f"{destros.std():.2f}"]
        })
        st.dataframe(resumo_mao, use_container_width=True, hide_index=True)
        
        st.info(f"""
        **Resultado do Teste t de Welch:**
        - **Estatística t**: `{t_stat:.3f}`
        - **p-valor**: `{p_val_t:.4f}`
        - **Cohen's d**: `{cohen_d_mao:.4f}` (tamanho de efeito irrelevante/próximo de zero)
        
        **Conclusão**: Embora haja significância estatística pelo tamanho gigante da amostra ($N > 70.000$), a diferença prática na média de aproveitamento de saque entre canhotos e destros é desprezível (< 0,5 p.p.).
        """)

    with col_t2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=df_completo[df_completo['hand'].isin(['L', 'R'])], x='mao_rotulo', y='pct_pontos_saque_ganhos', palette=['#3B82F6', '#10B981'], ax=ax)
        ax.set_title("Aproveitamento de Saque por Mão Dominante")
        ax.set_xlabel("")
        ax.set_ylabel("% Pontos Ganhos")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # TESTE 2: EFEITO DA SUPERFÍCIE (ANOVA + TUKEY)
    st.markdown("### 2. Efeito da Superfície (Grass vs. Hard vs. Clay)")
    
    hard = df_completo[df_completo['surface'] == 'Hard']['pct_pontos_saque_ganhos'].dropna()
    clay = df_completo[df_completo['surface'] == 'Clay']['pct_pontos_saque_ganhos'].dropna()
    grass = df_completo[df_completo['surface'] == 'Grass']['pct_pontos_saque_ganhos'].dropna()
    
    f_stat, p_val_f = stats.f_oneway(hard, clay, grass)
    
    col_a1, col_a2 = st.columns([1, 1])
    
    with col_a1:
        st.markdown("#### Intervalos de Confiança (IC 95%)")
        ic_dados = []
        for nome, grupo in [('Grass (Grama)', grass), ('Hard (Rápida)', hard), ('Clay (Saibro)', clay)]:
            ic = stats.t.interval(0.95, len(grupo)-1, loc=grupo.mean(), scale=stats.sem(grupo))
            ic_dados.append({
                'Superfície': nome,
                'N': f"{len(grupo):,}",
                'Média (%)': f"{grupo.mean():.2f}%",
                'IC 95% Inferior': f"{ic[0]:.2f}%",
                'IC 95% Superior': f"{ic[1]:.2f}%",
                'Desvio Padrão': f"{grupo.std():.2f}"
            })
        st.dataframe(pd.DataFrame(ic_dados), use_container_width=True, hide_index=True)
        
        st.success(f"""
        **ANOVA One-Way:**
        - **Estatística F**: `{f_stat:.2f}` | **p-valor**: `{p_val_f:.4e}` (fortemente significativo)
        - **Hierarquia clara**: $\\text{{Grama}} (66.1\\%) > \\text{{Rápida}} (64.1\\%) > \\text{{Saibro}} (61.9\\%)$
        """)

    with col_a2:
        st.markdown("#### Comparações Múltiplas de Tukey (HSD)")
        dados_sup = df_completo[df_completo['surface'].isin(['Hard', 'Clay', 'Grass'])]
        tukey = pairwise_tukeyhsd(endog=dados_sup['pct_pontos_saque_ganhos'], groups=dados_sup['surface'], alpha=0.05)
        
        # Formatar tabela do Tukey
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_df['meandiff'] = tukey_df['meandiff'].map(lambda x: f"{x:+.2f} p.p.")
        tukey_df['p-adj'] = tukey_df['p-adj'].map(lambda x: f"{x:.4f}")
        tukey_df['lower'] = tukey_df['lower'].map(lambda x: f"{x:+.2f}")
        tukey_df['upper'] = tukey_df['upper'].map(lambda x: f"{x:+.2f}")
        
        st.dataframe(tukey_df, use_container_width=True, hide_index=True)
        
        # Tamanho de efeito Cohen's d entre pares
        def calc_cohen_d(g1, g2):
            n_1, n_2 = len(g1), len(g2)
            s_pool = np.sqrt(((n_1-1)*g1.std()**2 + (n_2-1)*g2.std()**2) / (n_1+n_2-2))
            return (g1.mean() - g2.mean()) / s_pool
        
        st.markdown(f"""
        **Tamanho de Efeito (Cohen's d):**
        - **Clay vs Grass**: `{calc_cohen_d(clay, grass):.3f}` *(médio/forte)*
        - **Clay vs Hard**: `{calc_cohen_d(clay, hard):.3f}` *(pequeno a moderado)*
        - **Grass vs Hard**: `{calc_cohen_d(grass, hard):.3f}` *(pequeno)*
        """)


# -------------------------------------------------------------
# ABA 3: MODELO DE REGRESSÃO
# -------------------------------------------------------------
with tab_regressao:
    st.subheader("Modelagem de Regressão Linear Múltipla (OLS)")
    
    st.markdown(r"""
    A especificação do modelo ajustado prediz o aproveitamento de pontos de saque com base nas características do jogador e nas condições da partida:
    $$\text{pct\_pontos\_saque\_ganhos} = \beta_0 + \beta_1 \cdot \text{ht} + \beta_2 \cdot \text{age} + \beta_3 \cdot \text{surface} + \beta_4 \cdot \ln(\text{rank})$$
    """)

    # Métricas de Validação no Teste
    r_treino = resultados_regressao['r2']
    mae_val = resultados_regressao['mae']
    rmse_val = resultados_regressao['rmse']
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("R² (Teste)", f"{r_treino:.3f}")
    col_m2.metric("MAE (Erro Médio Absoluto)", f"{mae_val:.2f} p.p.")
    col_m3.metric("RMSE", f"{rmse_val:.2f} p.p.")
    col_m4.metric("Amostras de Teste", f"{len(resultados_regressao['teste']):,}")

    st.markdown("---")
    
    col_c1, col_c2 = st.columns([1.2, 0.8])
    
    with col_c1:
        st.markdown("#### Coeficientes Estimados do Modelo")
        modelo_fit = resultados_regressao['modelo']
        
        coef_df = pd.DataFrame({
            'Variável': modelo_fit.params.index,
            'Coeficiente (β)': modelo_fit.params.values,
            'Erro Padrão': modelo_fit.bse.values,
            'Estatística t': modelo_fit.tvalues.values,
            'p-valor': modelo_fit.pvalues.values,
            'IC 95% Inferior': modelo_fit.conf_int()[0].values,
            'IC 95% Superior': modelo_fit.conf_int()[1].values,
        })
        
        # Nomes amigáveis
        nomes_map = {
            'Intercept': 'Intercepto (Base: Clay)',
            'surface[T.Grass]': 'Superfície: Grama (Grass)',
            'surface[T.Hard]': 'Superfície: Rápida (Hard)',
            'ht': 'Altura (cm)',
            'age': 'Idade (anos)',
            'np.log(rank)': 'log(Ranking ATP)'
        }
        coef_df['Variável'] = coef_df['Variável'].map(nomes_map).fillna(coef_df['Variável'])
        
        format_dict = {
            'Coeficiente (β)': '{:+.3f}'.format,
            'Erro Padrão': '{:.4f}'.format,
            'Estatística t': '{:+.2f}'.format,
            'p-valor': '{:.4e}'.format,
            'IC 95% Inferior': '{:+.3f}'.format,
            'IC 95% Superior': '{:+.3f}'.format,
        }
        st.dataframe(coef_df.style.format(format_dict), use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Interpretação Prática:**
        - **Altura (`ht`)**: Cada **+10 cm de altura** acrescenta aproximadamente **+1,5 a +1,6 p.p.** no aproveitamento de saque.
        - **Superfície**: Em relação ao Saibro (*baseline*), jogar na **Grama** acrescenta **+4,15 p.p.** e na **Rápida** acrescenta **+2,17 p.p.**
        - **Ranking (`log(rank)`)**: Piora no ranking (maior valor numérico) penaliza o saque de forma logarítmica.
        """)

    with col_c2:
        st.markdown("#### Diagnóstico dos Resíduos")
        residuos = resultados_regressao['residuos_teste']
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.histplot(residuos, kde=True, color='#2563EB', bins=35, ax=ax)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title("Distribuição dos Resíduos (Erros)")
        ax.set_xlabel("Erro (Real - Previsto)")
        ax.set_ylabel("Frequência")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("#### Gráfico de Dispersão: Real vs. Previsto")
    
    teste_df = resultados_regressao['teste']
    prevs = resultados_regressao['previsoes_teste']
    
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sample_idx = np.random.choice(len(prevs), size=min(3000, len(prevs)), replace=False)
    ax.scatter(prevs.iloc[sample_idx], teste_df['pct_pontos_saque_ganhos'].iloc[sample_idx], alpha=0.15, color='#7C3AED', s=14)
    ax.plot([40, 90], [40, 90], color='red', linestyle='--', linewidth=2, label="Previsão Ideal (y = x)")
    ax.set_xlabel("Previsão do Modelo (%)")
    ax.set_ylabel("Valor Real Observado (%)")
    ax.set_title("Desempenho no Conjunto de Teste (Valores Reais vs Previstos)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------------------------------------
# ABA 4: SIMULADOR INTERATIVO DE SAQUE
# -------------------------------------------------------------
with tab_simulador:
    st.subheader("🎯 Simulador Preditivo de Desempenho de Saque")
    st.markdown("Ajuste as características do atleta e o cenário da partida para prever o aproveitamento esperado de saque com base no modelo de regressão.")

    # Inicializar estado da sessão se necessário
    if 'sim_altura' not in st.session_state:
        st.session_state.sim_altura = 188
    if 'sim_idade' not in st.session_state:
        st.session_state.sim_idade = 26
    if 'sim_superficie' not in st.session_state:
        st.session_state.sim_superficie = 'Hard'
    if 'sim_rank' not in st.session_state:
        st.session_state.sim_rank = 20

    def definir_preset(altura, idade, superficie, rank):
        st.session_state.sim_altura = altura
        st.session_state.sim_idade = idade
        st.session_state.sim_superficie = superficie
        st.session_state.sim_rank = rank

    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("#### Configuração do Atleta e Cenário")
        
        sim_altura = st.slider("Altura do Jogador (cm)", min_value=165, max_value=215, key='sim_altura', help="Ex: Schwartzman (170cm), Federer/Nadal (185cm), Medvedev (198cm), Isner/Opelka (208cm)")
        sim_idade = st.slider("Idade do Jogador (anos)", min_value=17, max_value=42, key='sim_idade')
        sim_superficie = st.selectbox("Superfície da Partida", ['Grass', 'Hard', 'Clay'], key='sim_superficie')
        sim_rank = st.number_input("Ranking ATP", min_value=1, max_value=500, key='sim_rank')
        
        # Perfis pré-definidos rápidos
        st.markdown("**Perfis Típicos do Circuito:**")
        cp1, cp2, cp3 = st.columns(3)
        cp1.button("Big Server (Isner)", on_click=definir_preset, args=(208, 28, 'Grass', 15), use_container_width=True)
        cp2.button("All-Court (Federer)", on_click=definir_preset, args=(185, 27, 'Hard', 3), use_container_width=True)
        cp3.button("Counter-Puncher", on_click=definir_preset, args=(170, 26, 'Clay', 25), use_container_width=True)

    with col_output:
        st.markdown("#### Aproveitamento Estimado")
        
        # Gerar DataFrame de inferência
        df_sim = pd.DataFrame({
            'ht': [sim_altura],
            'age': [sim_idade],
            'surface': [sim_superficie],
            'rank': [sim_rank]
        })
        
        predicao = resultados_regressao['modelo'].predict(df_sim)[0]
        
        # Médias de referência
        media_sup = df_completo[df_completo['surface'] == sim_superficie]['pct_pontos_saque_ganhos'].mean()
        media_geral = df_completo['pct_pontos_saque_ganhos'].mean()
        diff_sup = predicao - media_sup
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#4B5563;">Previsão de Pontos de Saque Ganhos</h4>
            <div style="font-size: 3rem; font-weight: 800; color: #1E3A8A; margin: 10px 0;">{predicao:.2f}%</div>
            <p style="margin:0; font-size:1.05rem; color: {'#059669' if diff_sup >= 0 else '#DC2626'}; font-weight:600;">
                {diff_sup:+.2f} p.p. em relação à média da superfície ({media_sup:.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparativo entre todas as superfícies para este mesmo jogador
        df_comparativo = pd.DataFrame({
            'ht': [sim_altura, sim_altura, sim_altura],
            'age': [sim_idade, sim_idade, sim_idade],
            'surface': ['Grass', 'Hard', 'Clay'],
            'rank': [sim_rank, sim_rank, sim_rank]
        }, index=['Grama (Grass)', 'Rápida (Hard)', 'Saibro (Clay)'])
        
        df_comparativo['Previsão (%)'] = resultados_regressao['modelo'].predict(df_comparativo)
        
        fig, ax = plt.subplots(figsize=(6, 2.8))
        cores = ['#10B981', '#3B82F6', '#EF4444']
        bars = ax.barh(df_comparativo.index, df_comparativo['Previsão (%)'], color=cores, height=0.55)
        ax.set_xlim(50, 80)
        ax.axvline(media_geral, color='gray', linestyle='--', label=f'Média Geral ATP ({media_geral:.1f}%)')
        ax.set_xlabel("% Estimada de Pontos de Saque Ganhos")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontweight='bold', fontsize=9)
        ax.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# -------------------------------------------------------------
# ABA 5: METODOLOGIA E DADOS
# -------------------------------------------------------------
with tab_sobre:
    st.subheader("Sobre o Projeto e Pipeline de Dados")
    
    st.markdown("""
    ### 📌 Contexto e Objetivos
    Este projeto tem como finalidade investigar quantitativamente os determinantes do **sucesso no saque** no tênis masculino de alto rendimento (ATP Tour), cobrindo a era de 2010 a 2024.

    ### 🧹 Pipeline de Limpeza dos Dados
    1. **Filtro de Torneios**: Exclusão de Davis Cup (`D`) e Olimpíadas (`O`) para manter a homogeneidade do circuito individual.
    2. **Reshape Jogador-Partida**: Cada partida foi desdobrada em 2 registros (um para o vencedor e outro para o perdedor), evitando viés de seleção da vitória.
    3. **Tratamento de Inconsistências**: Exclusão de partidas sem pontos de saque registrados (`svpt == 0`) e inconsistências lógicas (`1stIn > svpt`).
    4. **Estabilização de Proporções**: Filtro de volume mínimo de pelo menos 20 pontos de saque disputados (`svpt >= 20`) para eliminar distorções de desistências prematuras por lesão.

    ### 📐 Variáveis Centrais
    - `pct_pontos_saque_ganhos` = $\\frac{\\text{firstWon} + \\text{secondWon}}{\\text{svpt}} \\times 100$
    - `pct_primeiro_saque_dentro` = $\\frac{\\text{firstIn}}{\\text{svpt}} \\times 100$
    - `aces_por_100_saques` = $\\frac{\\text{ace}}{\\text{svpt}} \\times 100$
    
    ---
    *Projeto desenvolvido para a disciplina de Modelagem Estatística.*
    """)
