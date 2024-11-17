# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Função para converter valores com sufixos k, m, b em números reais
def convert_to_number(value):
    if isinstance(value, str):
        value = value.replace(',', '').lower()
        if 'k' in value:
            return float(value.replace('k', '')) * 1e3
        elif 'm' in value:
            return float(value.replace('m', '')) * 1e6
        elif 'b' in value:
            return float(value.replace('b', '')) * 1e9
    return float(value)

# Carregando o dataset
dados = pd.read_csv("top_insta_influencers_data.csv")

# Tratando valores numéricos com sufixos
for coluna in ['posts', 'followers', 'avg_likes', 'new_post_avg_like', 'total_likes']:
    dados[coluna] = dados[coluna].apply(convert_to_number)

# Removendo '%' e convertendo a coluna `60_day_eng_rate` para float
dados['60_day_eng_rate'] = dados['60_day_eng_rate'].str.replace('%', '').astype(float) / 100

# Excluindo colunas irrelevantes ou não numéricas
dados.drop(columns=['rank', 'channel_info', 'country'], inplace=True)

# Removendo linhas com valores nulos (se houver)
dados.dropna(inplace=True)

# Exibindo informações do dataset tratado
print("\nDados Após Tratamento:")
print(dados.info())

# Análise Exploratória: Visualização de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(dados.corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de Correlação (Apenas Variáveis Numéricas)")
plt.show()

# Dividindo as variáveis dependentes e independentes
X = dados.drop(columns=["60_day_eng_rate"])  # Substituir pelo nome correto do target
y = dados["60_day_eng_rate"]  # Substituir pelo nome correto do target

# Divisão em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos Dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regressão Linear
modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)

# Predição no conjunto de teste
y_pred = modelo.predict(X_test_scaled)

# Avaliação do Modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMétricas de Avaliação do Modelo Linear:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Regularização com Ridge e Lasso
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, ridge_pred)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\nAvaliação com Regularização:")
print(f"Ridge R²: {ridge_r2:.4f}")
print(f"Lasso R²: {lasso_r2:.4f}")

# Visualização: Reais vs Preditos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predito")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2, label="Ideal")
plt.xlabel("Valores Reais")
plt.ylabel("Valores Preditos")
plt.title("Regressão Linear: Reais vs Preditos")
plt.legend()
plt.show()

# Coeficientes do Modelo
coeficientes = pd.DataFrame({"Variável": X.columns, "Coeficiente": modelo.coef_})
print("\nCoeficientes do Modelo:")
print(coeficientes)

# Validação Cruzada
scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")
print("\nValidação Cruzada (R² médio):")
print(f"R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
