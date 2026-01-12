import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as ols
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 加载所有数据 (Load All Datasets)
# ==========================================
print("📥 正在加载四份关键数据...")
# 读取之前步骤生成的四个 CSV 文件
df_pca = pd.read_csv("London_Borough_PCA_Scores.csv")       # 自变量 (X1, X2)
df_crime = pd.read_csv("London_Crime_Data_2023.csv")        # 因变量 (Y - 原始计数)
df_pop = pd.read_csv("London_Borough_Population.csv")       # 分母 (人口)
df_housing = pd.read_csv("London_Borough_House_Prices_2023.csv") # 控制变量 (Control)

# ==========================================
# 2. 数据清洗与合并 (Cleaning & Merging)
# ==========================================
# 统一将 'Borough' 设为索引，并去除可能存在的空格
for df in [df_pca, df_crime, df_pop, df_housing]:
    # 查找列名是 'Borough' 还是 'BoroughName'
    col_name = 'Borough' if 'Borough' in df.columns else 'BoroughName'
    if col_name in df.columns:
        df[col_name] = df[col_name].str.strip() # 去除空格
        df.set_index(col_name, inplace=True)
        df.index.name = 'Borough' # 统一索引名为 Borough

# 执行大合并 (Inner Join)
# 只有在四张表里都存在的区才会被保留 (N=32/33)
print("🔄 正在合并数据表...")
df_master = df_pca.join([df_crime, df_pop, df_housing], how='inner')

# ==========================================
# 3. 特征工程：计算犯罪率 (Feature Engineering)
# ==========================================
print("🧮 计算每千人犯罪率...")
# 核心公式: (罪案数 / 人口) * 1000
df_master['Total_Crime_Rate'] = (df_master['Total_Crime_2023'] / df_master['Population']) * 1000
df_master['Violence_Rate'] = (df_master['Violence_2023'] / df_master['Population']) * 1000

# ==========================================
# 4. 运行回归模型 (Regression Analysis)
# ==========================================
print("\n🤖 正在运行 OLS 回归模型...")

# --- 模型 A: 总体犯罪率 (Total Crime) ---
# 解释: 总体治安 = 繁荣度(PC1) + 生产性(PC2) + 富裕程度(房价)
model_a = ols.ols('Total_Crime_Rate ~ PC1 + PC2 + House_Price_2023', data=df_master).fit()

# --- 模型 B: 暴力犯罪率 (Violence) ---
# 解释: 暴力程度 = 繁荣度(PC1) + 生产性(PC2) + 富裕程度(房价)
model_b = ols.ols('Violence_Rate ~ PC1 + PC2 + House_Price_2023', data=df_master).fit()

# ==========================================
# 5. 输出与保存结果 (Output & Save)
# ==========================================
# 打印模型 A 摘要
print("\n" + "="*60)
print("📊 模型 A: 总体犯罪率 (TOTAL CRIME RATE)")
print("="*60)
print(model_a.summary())

# 打印模型 B 摘要
print("\n" + "="*60)
print("📊 模型 B: 暴力犯罪率 (VIOLENCE RATE)")
print("="*60)
print(model_b.summary())

# 保存最终总表 (用于附录展示或后续画图)
df_master.to_csv("Final_Analysis_Table.csv")
print("\n💾 最终分析总表已保存为: Final_Analysis_Table.csv")

# ==========================================
# 6. 可视化 (Visualization - Optional)
# ==========================================
# 画一张 "预测值 vs 真实值" 的图，展示模型有多准
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model_b.fittedvalues, y=df_master['Violence_Rate'], s=100, alpha=0.8)

# 画对角线 (完美预测线)
min_val = df_master['Violence_Rate'].min()
max_val = df_master['Violence_Rate'].max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

# 添加标签
for i in range(len(df_master)):
    plt.text(model_b.fittedvalues[i]+0.5, df_master['Violence_Rate'][i],
             df_master.index[i], fontsize=8)

plt.title('Model Accuracy: Predicted vs Actual Violence Rate')
plt.xlabel('Predicted Violence Rate (Model Output)')
plt.ylabel('Actual Violence Rate (Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('regression_accuracy_plot.png')
print("📈 模型准确度图表已保存为: regression_accuracy_plot.png")

# ==========================================
# 6. 可视化 (Visualization) - 修改版
# ==========================================
# 我们改用 Model A (Total Crime)，因为它的 R² 最高，展示效果最好
plt.figure(figsize=(10, 6))

# 修改 x 为 model_a.fittedvalues (预测值)
# 修改 y 为 df_master['Total_Crime_Rate'] (真实值)
sns.scatterplot(x=model_a.fittedvalues, y=df_master['Total_Crime_Rate'], s=100, alpha=0.8, color='blue')

# 画对角线 (完美预测线)
min_val = df_master['Total_Crime_Rate'].min()
max_val = df_master['Total_Crime_Rate'].max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line')

# 添加标签 (标注行政区名字)
# 这里的循环也需要对应修改数据源
for i in range(len(df_master)):
    # 为了避免字叠在一起，稍微错开一点位置 (+2)
    plt.text(model_a.fittedvalues[i]+2, df_master['Total_Crime_Rate'][i],
             df_master.index[i], fontsize=8, alpha=0.7)

plt.title(f'Model Accuracy: Predicted vs Actual Total Crime Rate')
plt.xlabel('Predicted Crime Rate (Model Output)')
plt.ylabel('Actual Crime Rate (Data)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 保存图片
plt.savefig('regression_accuracy_plot_TotalCrime.png', dpi=300)
print("📈 总体犯罪率模型图表已保存为: regression_accuracy_plot_TotalCrime.png")
plt.show()