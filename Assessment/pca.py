import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 加载数据 (Load Data)
# ==========================================
df_culture = pd.read_csv('London_Cultural_Infrastructure_Map.csv')
df_pop = pd.read_csv('London_Borough_Population.csv')

# ==========================================
# 2. 预处理 (Preprocessing)
# ==========================================
# 清理行政区名称 (去除空格)
df_culture['Borough'] = df_culture['Borough'].str.strip()
df_pop['Borough'] = df_pop['Borough'].str.strip()

# 合并数据
df_merged = pd.merge(df_culture, df_pop, on='Borough', how='inner')
df_merged.set_index('Borough', inplace=True)

# 分离特征和人口
# 排除 'Population' 列，其他都是文化设施
features = [col for col in df_merged.columns if col != 'Population']
X_counts = df_merged[features]
population = df_merged['Population']

# ***关键步骤***: 计算密度 (每 10,000 人)
# 这是 Practical 中没有但对城市数据必须做的一步
X_density = X_counts.div(population, axis=0) * 10000

# ==========================================
# 3. 标准化 (Standardization - 参考 Week 9)
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_density)

# 转回 DataFrame 方便查看
X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df_merged.index)

# ==========================================
# 4. 运行 PCA (PCA Analysis - 参考 Week 9)
# ==========================================
# 提取前 2 个主成分
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# 创建结果表
pca_df = pd.DataFrame(data=principal_components,
                      columns=['PC1', 'PC2'],
                      index=df_merged.index)

# ==========================================
# 5. 结果解释 (Interpretation)
# ==========================================
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

# 查看载荷 (Loadings) - 谁决定了 PC1 和 PC2？
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
print("\nTop positive loadings for PC1 (Intensity):")
print(loadings['PC1'].sort_values(ascending=False).head(5))

print("\nTop positive loadings for PC2 (Production/Creative):")
print(loadings['PC2'].sort_values(ascending=False).head(5))

# ==========================================
# 6. 可视化 (Visualization)
# ==========================================
plt.figure(figsize=(12, 10))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=100)

# 添加标签
for i in range(pca_df.shape[0]):
    plt.text(pca_df.PC1[i]+0.1, pca_df.PC2[i], pca_df.index[i], fontsize=9)

plt.title('London Cultural PCA: Intensity (PC1) vs Production (PC2)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.show()

# 保存 PCA 得分供下一步回归使用
pca_df.to_csv('London_Borough_PCA_Scores.csv')
print("\nPCA scores saved to 'London_Borough_PCA_Scores.csv'")