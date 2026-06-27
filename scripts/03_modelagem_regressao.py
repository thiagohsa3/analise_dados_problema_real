import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# CRIAR PASTA PARA GRÁFICOS (SE NÃO EXISTIR)
# ==============================================================================
if not os.path.exists('resultados_graficos'):
    os.makedirs('resultados_graficos')

# Configurações de estilo para os gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ==============================================================================
# 1. CARREGAR DADOS (BASE ÚNICA)
# ==============================================================================
df = pd.read_csv('Data_trabalho_ebac_lmp1.csv')


# ==============================================================================
# COMPONENTES DO MODELO A: PREDIÇÃO DE DEPRESSÃO
# ==============================================================================

X_A = df[['H_uso_do_dispo_p_dia', 'desbloqueios_do_celular', 'pontuacao_de_dependencia_digital']]
Y_A = df['pontuacao_de_depressao']

X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(
    X_A, Y_A, test_size=0.2, random_state=42)

# Treino
modelo_A = LinearRegression()
modelo_A.fit(X_train_A, Y_train_A)
Y_prev_A = modelo_A.predict(X_test_A)

# Métricas
r2_A = r2_score(Y_test_A, Y_prev_A)
rmse_A = root_mean_squared_error(Y_test_A, Y_prev_A)

print("\n" + "="*80)
print(" MODELO A: PREDIÇÃO DE DEPRESSÃO ".center(80, "="))
print("="*80)
print(f'Coeficiente de Determinação (R²): {r2_A:.2f}')
print(f'RMSE: {rmse_A:.2f}')
print(f'Desvio padrão da pontuação de depressão: {df["pontuacao_de_depressao"].std():.2f}')
print(f'Intercepto (β0): {modelo_A.intercept_:.4f}')
print("\nCoeficientes do modelo:")
coef_A = pd.DataFrame({'Variavel': X_A.columns, 'Coeficiente': modelo_A.coef_})
print(coef_A)
print("="*80 + "\n")

joblib.dump(modelo_A, 'modelo_regressao_depressao.pkl')

# ==============================================================================
# GRÁFICOS DO MODELO A - SALVANDO E EXIBINDO
# ==============================================================================

# Gráfico 1: Valores Reais vs Previstos (Modelo A)
plt.figure()
sns.scatterplot(x=Y_test_A, y=Y_prev_A, alpha=0.6, color='teal')
plt.plot([Y_test_A.min(), Y_test_A.max()], [Y_test_A.min(), Y_test_A.max()], '--r', linewidth=2)
plt.title(f'Modelo A: Valores Reais vs. Previstos (R² = {r2_A:.2f})', fontweight='bold')
plt.xlabel('Pontuação de Depressão Real (Teste)')
plt.ylabel('Pontuação de Depressão Prevista')
plt.tight_layout()
plt.savefig('resultados_graficos/9_regressao_modeloA_reais_previstos.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Gráfico 2: Resíduos (Modelo A)
residuos_A = Y_test_A - Y_prev_A
plt.figure()
sns.histplot(residuos_A, kde=True, color='darkred')
plt.axvline(0, color='blue', linestyle='--')
plt.title('Modelo A: Distribuição dos Resíduos (Erros)', fontweight='bold')
plt.xlabel('Erro (Real - Previsto)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig('resultados_graficos/10_regressao_modeloA_residuos.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# ==============================================================================
# COMPONENTES DO MODELO B: ANÁLISE DE DEGRADRAÇÃO DO SONO
# ==============================================================================

X_B = df[['H_uso_do_dispo_p_dia', 'pontuacao_de_depressao']]
Y_B = df['qualidade_do_sono']

X_train_B, X_test_B, Y_train_B, Y_test_B = train_test_split(
    X_B, Y_B, test_size=0.2, random_state=42)

# Treino
modelo_B = LinearRegression()
modelo_B.fit(X_train_B, Y_train_B)
Y_prev_B = modelo_B.predict(X_test_B)

# Métricas
r2_B = r2_score(Y_test_B, Y_prev_B)
rmse_B = root_mean_squared_error(Y_test_B, Y_prev_B)

# PRINT DAS MÉTRICAS DO MODELO B NO TERMINAL
print("\n" + "="*80)
print(" MODELO B: ANÁLISE DE DEGRADAÇÃO DO SONO ".center(80, "="))
print("="*80)
print(f'Coeficiente de Determinação (R²): {r2_B:.2f}')
print(f'RMSE: {rmse_B:.2f}')
print(f'Desvio padrão da qualidade do sono: {df["qualidade_do_sono"].std():.2f}')
print(f'Intercepto (β0): {modelo_B.intercept_:.4f}')
print("\nCoeficientes do modelo:")
coef_B = pd.DataFrame({'Variavel': X_B.columns, 'Coeficiente': modelo_B.coef_})
print(coef_B)
print("="*80 + "\n")

joblib.dump(modelo_B, 'modelo_regressao_sono.pkl')

# ==============================================================================
# GRÁFICOS DO MODELO B - SALVANDO E EXIBINDO
# ==============================================================================

# Gráfico 3: Valores Reais vs Previstos (Modelo B)
plt.figure()
sns.scatterplot(x=Y_test_B, y=Y_prev_B, alpha=0.6, color='darkgreen')
plt.plot([Y_test_B.min(), Y_test_B.max()], [Y_test_B.min(), Y_test_B.max()], '--r', linewidth=2)
plt.title(f'Modelo B: Valores Reais vs. Previstos (R² = {r2_B:.2f})', fontweight='bold')
plt.xlabel('Qualidade do Sono Real (Teste)')
plt.ylabel('Qualidade do Sono Prevista')
plt.tight_layout()
plt.savefig('resultados_graficos/11_regressao_modeloB_reais_previstos.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Gráfico 4: Resíduos (Modelo B)
residuos_B = Y_test_B - Y_prev_B
plt.figure()
sns.histplot(residuos_B, kde=True, color='indigo')
plt.axvline(0, color='red', linestyle='--')
plt.title('Modelo B: Distribuição dos Resíduos (Erros)', fontweight='bold')
plt.xlabel('Erro (Real - Previsto)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig('resultados_graficos/12_regressao_modeloB_residuos.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Processo finalizado com sucesso! Modelos A e B executados.")
