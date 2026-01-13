import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.formula.api as smf

# ==========================================
# 1. 读取并合并数据
# ==========================================
# 读取你上传的四个文件
df_pca_scores = pd.read_csv("London_Borough_PCA_Scores.csv")
df_crime = pd.read_csv("London_Crime_Data_2023.csv")
df_pop = pd.read_csv("London_Borough_Population.csv")
df_culture = pd.read_csv("London_Cultural_Infrastructure_Map.csv")

# 统一索引为 Borough
for df in [df_pca_scores, df_crime, df_pop, df_culture]:
    cols = df.columns
    if 'BoroughName' in cols: df.rename(columns={'BoroughName': 'Borough'}, inplace=True)
    if 'Borough' in cols:
        df['Borough'] = df['Borough'].str.strip()
        df.set_index('Borough', inplace=True)

# ==========================================
# 2. 复现 PCA Loadings (为了画条形图)
# ==========================================
# 筛选数值列并标准化
numeric_cols = df_culture.select_dtypes(include=[np.number]).columns
df_culture_dens = df_culture[numeric_cols].join(df_pop['Population'], how='inner')

# 计算每千人拥有量 (Rate)
for col in numeric_cols:
    df_culture_dens[col] = (df_culture_dens[col] / df_culture_dens['Population']) * 1000
df_culture_model = df_culture_dens.drop(columns=['Population'])

# 运行 PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_culture_model)
pca = PCA(n_components=2)
pca.fit(X_scaled)
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=numeric_cols)

# ==========================================
# 3. 画图 A: "Maker Effect" 散点图
# ==========================================
# 准备数据
df_scatter = df_pca_scores.join([df_crime, df_pop], how='inner')
df_scatter['Total_Crime_Rate'] = (df_scatter['Total_Crime_2023'] / df_scatter['Population']) * 1000

plt.figure(figsize=(10, 6))
# 绘制带回归线的散点图
sns.regplot(x='PC2', y='Total_Crime_Rate', data=df_scatter,
            scatter_kws={'s': 100, 'alpha': 0.7, 'color': '#2c3e50'},
            line_kws={'color': '#e74c3c'}) # 红色拟合线

# 标注重要区域
for i in range(df_scatter.shape[0]):
    # 只标注比较极端的点，避免拥挤
    if abs(df_scatter['PC2'].iloc[i]) > 1.5 or df_scatter['Total_Crime_Rate'].iloc[i] > 35:
        plt.text(df_scatter['PC2'].iloc[i]+0.2, df_scatter['Total_Crime_Rate'].iloc[i],
                 df_scatter.index[i], fontsize=9)

plt.title('The "Maker Effect": Creative Production vs Total Crime Rate', fontsize=14, fontweight='bold')
plt.xlabel('Creative Production Intensity (PC2 Score)', fontsize=12)
plt.ylabel('Total Crime Rate (per 1,000 people)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot.png', dpi=300)
plt.show()

# ==========================================
# 4. 画图 B: PC2 载荷图 (解释什么是 "Production")
# ==========================================
loadings_sorted = loadings.sort_values(by='PC2', ascending=False)

plt.figure(figsize=(10, 8))
# 使用红蓝配色：红色代表正相关(Production)，蓝色代表负相关(Consumption)
colors = ['#e74c3c' if x > 0 else '#3498db' for x in loadings_sorted['PC2']]
sns.barplot(x=loadings_sorted['PC2'], y=loadings_sorted.index, palette=colors)

plt.title('Decoding PC2: Production (+) vs Consumption (-)', fontsize=14, fontweight='bold')
plt.xlabel('Contribution to PC2 Score', fontsize=12)
plt.ylabel('Type', fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig('loadings_plot.png', dpi=300)
plt.show()

# ==========================================
# [修改版] 第三步：绘制清晰的偏回归图
# 目的：只标注重要点，防止文字重叠
# ==========================================
df_house = pd.read_csv("London_Borough_House_Prices_2023.csv")
if 'BoroughName' in df_house.columns: df_house.rename(columns={'BoroughName': 'Borough'}, inplace=True)
df_house['Borough'] = df_house['Borough'].str.strip()
df_house.set_index('Borough', inplace=True)
df_house.rename(columns={'House_Price_2023': 'House_Price'}, inplace=True)
df_scatter = df_scatter.join(df_house['House_Price'], how='inner')

# 1. 手动计算“净效应” (残差)
# ----------------------------------------
# 算出 Y (犯罪率) 剔除 PC1 和 房价 后的剩余部分
resid_y = smf.ols('Total_Crime_Rate ~ PC1 + House_Price', data=df_scatter).fit().resid
# 算出 X (PC2) 剔除 PC1 和 房价 后的剩余部分
resid_x = smf.ols('PC2 ~ PC1 + House_Price', data=df_scatter).fit().resid

# 把残差装进临时表，方便画图
df_plot = pd.DataFrame({
    'x_partial': resid_x,
    'y_partial': resid_y,
    'Borough': df_scatter.index
})

# 2. 开始画图
# ----------------------------------------
plt.figure(figsize=(10, 7))

# 画散点和回归线 (使用 Seaborn，更好看)
sns.regplot(
    x='x_partial',
    y='y_partial',
    data=df_plot,
    scatter_kws={'s': 80, 'alpha': 0.7, 'color': '#2c3e50'},  # 深灰色点
    line_kws={'color': '#c0392b', 'linewidth': 2}  # 红色线 (这根线一定是向下倾斜的！)
)

# 3. 智能标注 (这是解决"糊成一团"的关键)
# ----------------------------------------
# 设定阈值：只标注离中心较远的点 (异常值/典型值)
# 你可以根据实际出来的图调整这个 1.5 或 40
for i in range(len(df_plot)):
    x_val = df_plot['x_partial'][i]
    y_val = df_plot['y_partial'][i]
    label = df_plot['Borough'][i]

    # 只有当 X 或 Y 的绝对值比较大时，才标注名字
    # if abs(x_val) > 1.5 or abs(y_val) > 40:
    plt.text(x_val + 0.05, y_val, label, fontsize=9, fontweight='bold', color='#333333')

# 4. 美化与保存
# ----------------------------------------
plt.title('The "Maker Effect" Revealed: Partial Regression Plot\n(Controlling for Intensity and Wealth)', fontsize=14,
          fontweight='bold')
plt.xlabel('Creative Production (Adjusted for Intensity & Wealth)', fontsize=11)
plt.ylabel('Total Crime Rate (Adjusted)', fontsize=11)
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('maker_effect_partial.png', dpi=300)
plt.show()