import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)


df = pd.read_csv('Data_trabalho_ebac_lmp1.csv', dtype_backend='pyarrow')



print("\n" + "="*60)
print("="*17+"ESTATISTICAS_DO_DATA_FRAME"+"="*17)
print("="*60)
print(df.describe().T)
print("="*60+"\n")

print("\n"+"="*60)
print("="*21+"CORRELACAO_GERAL"+"="*21)
print("="*60)
print(df.drop(columns=['id']).corr(numeric_only=True))
print("="*60+"\n")


print("\n" + "="*60)
print("="*14+'CORRELACAO_POSITIVA_ACIMA_DE_60'+"="*14)
print("="*60)
corr = df.drop(columns=['id']).corr(numeric_only=True)

alta_correlacao = corr.where((corr > 0.60) & (corr < 1.00)).stack().sort_values(ascending=False)

resultado = alta_correlacao.reset_index()
resultado.columns = ['Variavel_1', 'Variavel_2', 'Correlacao']

resultado['Par'] = resultado[['Variavel_1', 'Variavel_2']].apply(lambda x: tuple(sorted(x)), axis=1)
resultado_sem_duplicatas = resultado.drop_duplicates(subset='Par').drop(columns=['Par'])

print(resultado_sem_duplicatas)
print("="*60)

colunas_corr_positiva = [
    'H_uso_do_dispo_p_dia','pontuacao_de_dependencia_digital',
    'desbloqueios_do_celular','pontuacao_de_depressao',
    'pontuacao_de_ansiedade']

pearson_corr = df[colunas_corr_positiva].corr()
spearman_corr = df[colunas_corr_positiva].corr(method='spearman')

print("="*60)
print("="*12+'MATRIZ_CORRELACAO_POSITIVA(pearson)'+"="*12)
print("="*60)
print(pearson_corr)

print("="*60)
print("="*12+'MATRIZ_CORRELACAO_POSITIVA(spearman)'+"="*12)
print("="*60)
print(spearman_corr)
print("="*60)



print("\n" + "="*60)
print("="*12+'CORRELACAO_NEGATIVA_ACIMA_DE_60'+"="*12)
print("="*60)
corr = df.drop(columns=['id']).corr(numeric_only=True)

alta_correlacao_neg = corr.where(corr < -0.60).stack().sort_values(ascending=True)

resultado_neg = alta_correlacao_neg.reset_index()
resultado_neg.columns = ['Variavel_1', 'Variavel_2', 'Correlacao']

resultado_neg['Par'] = resultado_neg[['Variavel_1', 'Variavel_2']].apply(lambda x: tuple(sorted(x)), axis=1)
resultado_neg_sem_duplicatas = resultado_neg.drop_duplicates(subset='Par').drop(columns=['Par'])

print(resultado_neg_sem_duplicatas)
print("="*60+"\n")

colunas_corr_negativa =  [
    'notificacoes_p_dia','pontuacao_de_depressao',
    'qualidade_do_sono','pontuacao_de_foco',
    'H_uso_do_dispo_p_dia','nivel_de_estresse',
    'pontuacao_de_felicidade']

pearson_corr_negativa = df[colunas_corr_negativa].corr()
spearman_corr_negativa = df[colunas_corr_negativa].corr(method='spearman')

print("="*60)
print("="*12+'MATRIZ_CORRELACAO_NEGATIVA(pearson)'+"="*12)
print("="*60)
print(pearson_corr_negativa)
print("="*60)

print("="*60)
print("="*12+'MATRIZ_CORRELACAO_NEGATIVA(spearman)'+"="*12)
print("="*60)
print(spearman_corr_negativa)
print("="*60)


print("\n"+"="*60)



# ============================================
# GRAFICOS CORRELACOES POSITIVAS
# ============================================
import os  # Adicionado aqui para garantir que funcione direto


if not os.path.exists('resultados_graficos'):
    os.makedirs('resultados_graficos')

# ============================================
# CONFIGURACOES PROFISSIONAIS
# ============================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ============================================
# 1. HEATMAP (APENAS AS 7 CORRELAÇÕES)
# ============================================

colunas_heatmap = [
    'H_uso_do_dispo_p_dia',
    'desbloqueios_do_celular',
    'pontuacao_de_dependencia_digital',
    'pontuacao_de_depressao',
    'pontuacao_de_ansiedade'
]

matriz_corr = df[colunas_heatmap].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    matriz_corr,
    annot=True,
    fmt=".2f",
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=2,
    linecolor='white',
    cbar_kws={'shrink': 0.8, 'label': 'Correlação'}
)
plt.title('Mapa de Calor: Correlações entre Hábitos Digitais e Saúde Mental',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/1_heatmap_saude_mental.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# HEATMAP GERAL (TODAS AS COLUNAS NUMÉRICAS)
# ============================================

coluna_id = 'id'  # Substitua pelo nome da sua coluna ID, ou deixe '' se não houver

df_numerico = df.select_dtypes(include=[np.number])

if coluna_id in df_numerico.columns:
    df_numerico = df_numerico.drop(columns=[coluna_id])

matriz_corr_geral = df_numerico.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(
    matriz_corr_geral,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Correlação'}
)
plt.title('Mapa de Calor: Correlação entre Todas as Variáveis Numéricas',
          fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/2_heatmap_geral.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 3. REGRESSÃO: HORAS DE TELA × DEPENDENCIA DIGITAL (0.87)
# ============================================

plt.figure(figsize=(10, 6))
sns.regplot(
    x='H_uso_do_dispo_p_dia',
    y='pontuacao_de_dependencia_digital',
    data=df,
    line_kws={'color': 'red', 'linewidth': 2},
    scatter_kws={'alpha': 0.4, 'color': 'midnightblue'}
)
plt.title('Tendência: Horas de Tela × Dependência Digital (r = 0.87)', fontsize=14)
plt.xlabel('Horas de Uso de Dispositivo por Dia')
plt.ylabel('Pontuação de Dependência Digital')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/3_regressao_tela_dependencia.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 4. DISPERSÃO: DESBLOQUEIOS × HORAS DE TELA (0.85)
# ============================================

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='desbloqueios_do_celular',
    y='H_uso_do_dispo_p_dia',
    data=df,
    color='green',
    alpha=0.6
)
plt.title('Dispersão: Desbloqueios de Celular × Horas de Tela (r = 0.85)', fontsize=14)
plt.xlabel('Desbloqueios de Celular por Dia')
plt.ylabel('Horas de Uso de Dispositivo por Dia')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/4_dispersao_desbloqueios_tela.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# DISPERSÃO: DESBLOQUEIOS × DEPENDÊNCIA DIGITAL (0.84)
# ============================================

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='desbloqueios_do_celular',
    y='pontuacao_de_dependencia_digital',
    data=df,
    color='orange',
    alpha=0.6
)
plt.title('Dispersão: Desbloqueios de Celular × Dependência Digital (r = 0.84)', fontsize=14)
plt.xlabel('Desbloqueios de Celular por Dia')
plt.ylabel('Pontuação de Dependência Digital')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/5_dispersao_desbloqueios_dependencia.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# GRAFICOS CORRELACOES NEGATIVAS
# ============================================

# ============================================
# 1. HEATMAP  CORRELAÇÕES NEGATIVAS
# ============================================

colunas_heatmap_neg = [
    'notificacoes_p_dia',
    'pontuacao_de_foco',
    'qualidade_do_sono',
    'H_uso_do_dispo_p_dia',
    'pontuacao_de_depressao',
    'nivel_de_estresse',
    'pontuacao_de_felicidade'
]

matriz_corr_neg = df[colunas_heatmap_neg].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    matriz_corr_neg,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=2,
    linecolor='white',
    cbar_kws={'shrink': 0.8, 'label': 'Correlação'}
)
plt.title('Mapa de Calor: Correlações Negativas entre Hábitos Digitais e Saúde Mental',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/6_heatmap_negativo.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. REGRESSÃO: NOTIFICAÇÕES × FOCO (-0.77)
# ============================================

plt.figure(figsize=(10, 6))
sns.regplot(
    x='notificacoes_p_dia',
    y='pontuacao_de_foco',
    data=df,
    line_kws={'color': 'darkblue', 'linewidth': 2},
    scatter_kws={'alpha': 0.4, 'color': 'blue'}
)
plt.title('Tendência: Notificações por Dia × Pontuação de Foco (r = -0.77)', fontsize=14)
plt.xlabel('Notificações por Dia')
plt.ylabel('Pontuação de Foco')
plt.grid(True, linestyle='--', alpha=0.6)
plt.text(0.05, 0.95, 'Correlação Negativa Forte',
         transform=plt.gca().transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/7_regressao_notificacoes_foco.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 3. DISPERSÃO: DEPRESSÃO × QUALIDADE DO SONO (-0.69)
# ============================================

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='qualidade_do_sono',
    y='pontuacao_de_depressao',
    data=df,
    color='purple',
    alpha=0.6
)
plt.title('Dispersão: Qualidade do Sono × Depressão (r = -0.69)', fontsize=14)
plt.xlabel('Qualidade do Sono')
plt.ylabel('Pontuação de Depressão')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# SALVA O GRÁFICO ANTES DE EXIBIR
plt.savefig('resultados_graficos/8_dispersao_sono_depressao.png', dpi=300, bbox_inches='tight')
plt.show()

# FECHA TODOS OS GRÁFICOS DA MEMÓRIA DO COMPUTADOR
plt.close('all')

print("\nAnálise executada com sucesso! Todos os gráficos foram gerados e salvos na pasta 'resultados_graficos'.")
