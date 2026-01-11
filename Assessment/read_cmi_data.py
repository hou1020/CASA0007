import pandas as pd
import os

# ==========================================
# 1. 配置文件映射 (Configuration)
# ==========================================
# 音乐类文件 (需要特殊处理去重)
MUSIC_FILES = {
    "All": "CIM 2024 Music_Venues_All.csv",
    "Nightclubs": "CIM 2024 Music_Nightclubs.csv",
    "Grassroots": "CIM 2024 Music_Venues_Grassroots.csv"
}

# 标准文件 (文件名 -> 最终变量名)
STANDARD_FILES = {
    # High Culture / Day
    "CIM 2023 Museums and public galleries (Nov 2023).csv": "Museums_and_Public_Galleries",
    "CIM 2023 Commercial galleries (Nov 2023).csv": "Commercial_Galleries",
    "CIM 2023 Theatres.csv": "Theatres",
    "CIM 2023 Arts centres.csv": "Arts_Centres",
    "CIM 2023 Libraries (Nov 2023).csv": "Libraries",
    "CIM 2023 Archives (Nov 2023).csv": "Archives",
    "CIM 2023 Cinemas.csv": "Cinemas",

    # Night / Vibrancy
    "CIM 2023 Lgbt venues.csv": "LGBT_Venues",
    "CIM 2023 Dance performance venues.csv": "Dance_Performance_Venues",

    # Creative / Design / Gentrification
    "CIM 2023 Artists workspaces (Nov 2023).csv": "Artist_Workspaces",
    "CIM 2023 Creative coworking desk space.csv": "Creative_Coworking_Desk_Spaces",
    "CIM 2023 Creative workspaces (Nov 2023).csv": "Creative_Workspaces",
    "CIM 2023 Fashion and design.csv": "Fashion_and_Design",
    "CIM 2023 Textile design.csv": "Textile_Design",
    "CIM 2023 Jewellery design (Nov 2023).csv": "Jewellery_Design",
    "CIM 2023 Makerspaces (Nov 2023).csv": "Makerspaces",
    "CIM 2023 Making and manufacturing.csv": "Making_and_Manufacturing",
    "CIM 2023 Music recording studios (Nov 2023).csv": "Music_Recording_Studios",
    "CIM 2023 Music rehearsal studios (Nov 2023).csv": "Music_Rehearsal_Studios",
    "CIM 2023 Theatre rehearsal studios (Nov 2023).csv": "Theatre_Rehearsal_Studios",
    "CIM 2023 Dance rehearsal studios.csv": "Dance_Rehearsal_Studios",
    "CIM 2023 Prop and costume making.csv": "Prop_and_Costume_Making",
    "CIM 2023 Set and exhibition building.csv": "Set_and_Exhibition_Building"
}


# ==========================================
# 2. 辅助函数
# ==========================================
def get_borough_counts(df, col_name):
    """从DataFrame中智能查找行政区列并统计数量"""
    b_col = None
    for col in df.columns:
        if 'borough' in col.lower():
            b_col = col
            break
    if not b_col: return None

    # 清洗名称: 去空格, Title Case
    df['Clean_Borough'] = df[b_col].astype(str).str.strip().str.title()
    return df.groupby('Clean_Borough').size().reset_index(name=col_name)


# ==========================================
# 3. 核心处理逻辑
# ==========================================
dfs_to_merge = []

# --- A. 处理音乐数据 (去重逻辑) ---
if all(os.path.exists(f) for f in MUSIC_FILES.values()):
    print("🎵 处理音乐数据...")
    df_all = pd.read_csv(MUSIC_FILES["All"])
    df_nc = pd.read_csv(MUSIC_FILES["Nightclubs"])
    df_gr = pd.read_csv(MUSIC_FILES["Grassroots"])

    # 找出核心类别
    dfs_to_merge.append(get_borough_counts(df_nc, "Music_Nightclubs"))
    dfs_to_merge.append(get_borough_counts(df_gr, "Music_Grassroots"))

    # 计算 Others (在 All 中，但不在 Nightclubs 或 Grassroots 中的)
    known_names = set(df_nc['name'].str.strip().str.lower()) | set(df_gr['name'].str.strip().str.lower())
    df_others = df_all[~df_all['name'].astype(str).str.strip().str.lower().isin(known_names)].copy()
    dfs_to_merge.append(get_borough_counts(df_others, "Music_Others"))
else:
    print("⚠️ 警告: 音乐文件缺失，跳过音乐数据处理。")

# --- B. 处理标准文件 ---
print(f"📂 处理其他 {len(STANDARD_FILES)} 个标准文件...")
for filename, var_name in STANDARD_FILES.items():
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            counts = get_borough_counts(df, var_name)
            if counts is not None:
                dfs_to_merge.append(counts)
        except Exception as e:
            print(f"❌ 错误 {filename}: {e}")

# ==========================================
# 4. 合并与清洗
# ==========================================
print("🔄 正在合并...")
df_final = dfs_to_merge[0]
for df in dfs_to_merge[1:]:
    df_final = pd.merge(df_final, df, on='Clean_Borough', how='outer')

# 填充缺失值并设置索引
df_final = df_final.fillna(0).set_index('Clean_Borough').astype(int)
df_final.index.name = 'Borough'

# 标准化行政区名称 (解决 GLA 数据命名不一致问题)
rename_map = {
    "City Of Westminster": "Westminster",
    "Kensington And Chelsea": "Kensington and Chelsea",
    "Hammersmith And Fulham": "Hammersmith and Fulham",
    "Richmond Upon Thames": "Richmond upon Thames",
    "Kingston Upon Thames": "Kingston upon Thames",
    "Barking And Dagenham": "Barking and Dagenham",
    "City Of London": "City of London"
}
# 统一重命名并聚合 (防止出现重名行)
df_final.rename(index=rename_map, inplace=True)
df_final = df_final.groupby(level=0).sum()

# ==========================================
# 5. 保存结果
# ==========================================
output_file = "London_Cultural_Infrastructure_Map.csv"
df_final.to_csv(output_file)

print(f"✅ 成功! 数据包含 {df_final.shape[0]} 个行政区, {df_final.shape[1]} 个变量。")
print(f"📁 已保存为: {output_file}")
print("前5行预览:")
print(df_final.head())