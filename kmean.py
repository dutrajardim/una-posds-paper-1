# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from source.ml import kmeans as km
from source.helpers import snake_case
import source.ml.helpers as mlh
import source.ml.pca as mlp 

# %%
plt.style.use('ggplot')
cm = plt.get_cmap('Paired')
NUM_COLORS = 12
colors = [cm(i) for i in range(NUM_COLORS)]

# %%
columns = (
    ('Beta','beta'),
    ('Volatilidade anualizada','annualyzed_volatility'),
    ('Taxa anual de retorno', 'annual_rate_return'),
    ('Retorno acumulado', 'cumulative_return'),
)

df_sector = pd.read_csv('./data/setorial_b3_15_01_2021.csv')
df_sector.columns = [snake_case(column) for column in df_sector.columns]
df_sector.index = df_sector.codigo

def get_centroids_list (X, max=4, cached=False):
    if cached:
        return np.load("./data/initial_centroids.npy", allow_pickle=True)
    centroids = [km.init_centroids(X, K) for K in range(2, max)]
    np.save("./data/initial_centroids.npy", centroids)
    return centroids

# %% 
analisys_data = [
    ('./data/results_2019.csv', '2019'),
    ('./data/results_2020.csv', '2020')
]
classifications = {}
costs = []

for path, title in analisys_data:
    df = pd.read_csv(path, index_col='ticker').dropna()
    df = df.loc[:, [column for _, column in columns]]
    df_norm = (df - df.min())/(df.max()-df.min())
    X = df_norm.to_numpy()

    centroids_list = get_centroids_list(X, max=15, cached=True)

    classifications[title] = {
        'df': df,
        'norm': df_norm,
        'labels': []
    }

    for initial_centroids in centroids_list:
        centroids, idx, cost = km.k_means(X, initial_centroids)
        classifications[title]['labels'].append(idx + 1)
        costs.append({ "title": title, "K": len(initial_centroids), "cost": cost })
        
# %% Analisa custos
df_costs = pd.DataFrame(costs)
df_costs.columns = ['Período', 'K', 'Custo']
pd.pivot_table(df_costs, index=['K'], columns=['Período'], values=['Custo'], margins=True, margins_name="Total")

# %% Configura grupo
c = 6

# %% Características (Anexo 1)

size = len(columns)
fig, ax = plt.subplots(size, 2, figsize=(15, size*4), sharey='row')

for i, (key, value) in enumerate(classifications.items()):
    labels = np.unique(value['labels'][c])
    labels.sort()

    for j, (title, column) in enumerate(columns):
        data = [value['df'].loc[value['labels'][c] == label, column] for label in labels]
        ax[j][i].boxplot(data)
        ax[j][i].title.set_text(key)
        ax[j][i].set_ylabel(title)
        ax[j][i].set_xlabel('Grupos')
        
        for z in range(0, len(data)):
            y = data[z]
            x = np.random.normal(z+1, 0.12, size=len(y))
            ax[j][i].plot(x, y,'.', alpha=.3, color=colors[z])

fig.suptitle("Características (Anexo 1)", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% Grupos (Anexo 2)

size = len(centroids_list[c])
fig, ax = plt.subplots(size, 2, figsize=(15, size*4), sharey=True)

for i, (key, value) in enumerate(classifications.items()):
    for j in range(0, size):
        data = [value['norm'].loc[value['labels'][c] == (j + 1), column] for _, column in columns]
        labels = [legend for legend,_ in columns]
        ax[j][i].boxplot(data,labels=labels)
        ax[j][i].title.set_text(key)
        ax[j][i].set_ylabel("Grupo {}".format(j + 1))

        # color = "C{}".format(j)
        for z in range(0, len(data)):
            y = data[z]
            x = np.random.normal(z+1, 0.12, size=len(y))
            ax[j][i].plot(x, y,'.',c=colors[j], alpha=.3)

fig.suptitle("Grupos (Anexo 2)", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %%
K=1 

fig, ax = plt.subplots(1,2, figsize=(13, 5), sharey=True, sharex=True)

for i, (key, value) in enumerate(classifications.items()):
    
    X = value['df'][['cumulative_return', 'annual_rate_return']].to_numpy()
    X_norm, sigma, mu = mlh.feature_normalize(X)
    # X_norm = value['norm'][['cumulative_return', 'annual_rate_return']].to_numpy()
    U, S, V = mlp.pca(X_norm)
    Z_x = mlp.project_data(X_norm, U, K)

    Y = value['df'][['beta', 'annualyzed_volatility']].to_numpy()
    Y_norm, sigma, mu = mlh.feature_normalize(Y)
    # Y_norm = value['norm'][['beta', 'annualyzed_volatility']].to_numpy()
    U, S, V = mlp.pca(Y_norm)
    if i == 0:
        U = U * np.array([[-1,1], [1,1]])
    Z_y = mlp.project_data(Y_norm, U, K)

    labels = np.unique(value['labels'][c])

    for j, label in enumerate(labels):
        filter = value['labels'][c] == label
        ax[i].scatter(Z_x[filter], Z_y[filter], label=label, color=colors[j])
    
    ax[i].title.set_text(key)
    ax[i].set_xlabel("PCA (oportunidade)")
    ax[i].set_ylabel("PCA (risco)")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.suptitle("Analise por PCA (Gráfico 2)", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% Analise por PCA - 3D

fig = plt.figure(figsize=(15,5))
for i, (key, value) in enumerate(classifications.items()):
    ax = fig.add_subplot(1, 2, (i + 1), projection='3d')

    X = value['df'][['cumulative_return', 'annual_rate_return']].to_numpy()
    X_norm, sigma, mu = mlh.feature_normalize(X)
    U, S, V = mlp.pca(X_norm)
    # if i == 1:
    #     U = U * -1
    X_pca = mlp.project_data(X_norm, U, K)

    Y = value['df'][['beta']].to_numpy()
    # Y_norm, sigma, mu = mlh.feature_normalize(Y)

    Z = value['df'][['annualyzed_volatility']].to_numpy()
    # Z_norm, sigma, mu = mlh.feature_normalize(Z)

    for j, label in enumerate(labels):
        filter = value['labels'][c] == label
        ax.scatter(X_pca[filter], Y[filter], Z[filter], label=label, color=colors[j])
        ax.set_xlabel('PCA')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Volatilidade anual')

    # ax.legend()
    ax.title.set_text(key)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.suptitle("Analise por PCA - 3D (Gráfico 3)", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%

df_g18 = []
for i, (key, value) in enumerate(classifications.items()):
    df_temp = value['df'].copy()
    df_temp['labels'] = value['labels'][c]
    df_temp['year'] = key
    filter = np.isin(value['labels'][c], [1,8])
    df_temp.columns = ['Beta','Volatilidade','Taxa de retorno anual','Retorno acumulado', 'Grupo', 'Ano']
    df_g18.append(df_temp.loc[filter, :])

df_g18 = pd.concat(df_g18).reset_index()
p_g18 = df_g18.pivot_table(index=['Ano', 'Grupo', 'ticker'])
p_g18

#%%
print(p_g18.to_latex())
# %%
ls_qtd = []
for i, (key, value) in enumerate(classifications.items()):
    df_temp = value['df'].copy().reset_index()
    df_temp['labels'] = value['labels'][c]
    df_temp['sector'] = df_temp.ticker.apply(lambda x: df_sector.loc[x[:-4], 'setor_economico'])
    df_temp['year'] = key
    ls_qtd.append(df_temp[['ticker','labels','sector','year']])

df_qtd = pd.concat(ls_qtd)
df_qtd.columns = ['Ticker','Grupo','Setor','Ano']
pv_qtd = df_qtd.pivot_table(index=['Setor', 'Ano'], columns=['Grupo'], aggfunc='count', margins=True, margins_name='Total', fill_value=0)
# %%
print(pv_qtd.to_latex())
# %%
def c_year (x):
    return pd.Series({
        '2019': x.loc[x['Ano'] == '2019'].reset_index().Grupo.get(0),
        '2020': x.loc[x['Ano'] == '2020'].reset_index().Grupo.get(0),
    })

np_qtd = df_qtd \
    .groupby('Ticker') \
    .apply(c_year) \
    .reset_index() \
    .pivot_table(index=['2019'], columns=['2020'], aggfunc='count', margins=True, margins_name='Total',fill_value=0) \
    .to_numpy()

data = np_qtd / np_qtd[:,8].reshape((9,1))
pd_class = pd.DataFrame(data=data[:-1, :-1], index=range(1,9), columns=range(1,9))

# %%
fig, ax = plt.subplots(1,1,figsize=(6,6))
sbn.heatmap(pd_class, annot=True, fmt='.0%', ax=ax)

ax.set_xlabel("Grupo em 2020")
ax.set_ylabel("Grupo em 2019")
plt.suptitle("Reclassificação (Gráfico 4)", fontsize=14)
plt.show()
# %%
data[7,:]