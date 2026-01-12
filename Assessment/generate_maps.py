import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取地图数据 (请修改路径为你下载的文件)
gdf = gpd.read_file("London_Boroughs.geojson")

# 2. 读取你的 PCA 分数
df_scores = pd.read_csv("London_Borough_PCA_Scores.csv")

# 3. 合并数据 (注意地图文件里的名字通常叫 'name' 或 'BOROUGH')
gdf = gdf.merge(df_scores, left_on='BOROUGH', right_on='Borough', how='left')

# 4. 画 PC1 (Intensity) 地图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
gdf.plot(column='PC1', ax=ax, legend=True,
         cmap='OrRd', # 红色系，代表"热度"
         legend_kwds={'label': "Cultural Intensity Score (PC1)"})
ax.set_title("London's Cultural Intensity (The Consumer City)", fontsize=15)
ax.axis('off')
plt.savefig('map_pc1_intensity.png', dpi=300)

# 5. 画 PC2 (Production) 地图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
gdf.plot(column='PC2', ax=ax, legend=True,
         cmap='GnBu', # 蓝绿色系，代表"冷静/生产"
         legend_kwds={'label': "Creative Production Score (PC2)"})
ax.set_title("London's Creative Production (The Maker City)", fontsize=15)
ax.axis('off')
plt.savefig('map_pc2_production.png', dpi=300)