import pandas as pd
import numpy as np


def process_crime_2023():
    print("🕵️ 开始处理 2023 年犯罪数据...")

    # ==========================================
    # 1. 配置文件名
    # ==========================================
    file_hist = "MPS Borough Level Crime (Historical).csv"
    file_recent = "MPS Borough Level Crime (most recent 24 months).csv"

    output_file = "London_Crime_Data_2023.csv"

    try:
        # ==========================================
        # 2. 处理 MPS 犯罪数据 (合并 Historical 和 Recent)
        # ==========================================
        print("   - 读取 MPS 犯罪文件...")
        df_hist = pd.read_csv(file_hist)
        df_recent = pd.read_csv(file_recent)

        # 查找 2023 年的列
        # 历史文件通常包含 202301 - 202311
        cols_2023_hist = [c for c in df_hist.columns if str(c).startswith('2023')]
        # 新文件通常包含 202312
        cols_2023_recent = [c for c in df_recent.columns if str(c).startswith('2023')]

        print(f"     历史文件包含月份: {cols_2023_hist}")
        print(f"     新文件包含月份: {cols_2023_recent}")

        # --- A. 计算 2023 总体犯罪 (Total Crime) ---
        # 历史数据汇总
        df_hist['Crime_2023_Part1'] = df_hist[cols_2023_hist].sum(axis=1)
        grp_hist = df_hist.groupby('BoroughName')['Crime_2023_Part1'].sum()

        # 新数据汇总
        df_recent['Crime_2023_Part2'] = df_recent[cols_2023_recent].sum(axis=1)
        grp_recent = df_recent.groupby('BoroughName')['Crime_2023_Part2'].sum()

        # 合并 (Part1 + Part2)
        df_total = pd.concat([grp_hist, grp_recent], axis=1).fillna(0)
        df_total['Total_Crime_2023'] = df_total['Crime_2023_Part1'] + df_total['Crime_2023_Part2']

        # --- B. 计算 2023 暴力犯罪 (Violence Only) ---
        # 筛选 MajorText 为 'VIOLENCE AGAINST THE PERSON'
        violence_filter_hist = df_hist['MajorText'] == 'VIOLENCE AGAINST THE PERSON'
        violence_filter_recent = df_recent['MajorText'] == 'VIOLENCE AGAINST THE PERSON'

        grp_vio_hist = df_hist[violence_filter_hist].groupby('BoroughName')['Crime_2023_Part1'].sum()
        grp_vio_recent = df_recent[violence_filter_recent].groupby('BoroughName')['Crime_2023_Part2'].sum()

        df_violence = pd.concat([grp_vio_hist, grp_vio_recent], axis=1).fillna(0)
        df_total['Violence_2023'] = df_violence['Crime_2023_Part1'] + df_violence['Crime_2023_Part2']

        # 清理中间列
        final_df = df_total[['Total_Crime_2023', 'Violence_2023']].copy()

        # # ==========================================
        # # 3. 处理 ASB 数据
        # # ==========================================
        # print("   - 读取 ASB 数据 (可能需要一点时间)...")
        # try:
        #     # ASB 文件较大，只读取需要的列
        #     df_asb = pd.read_csv(file_asb, usecols=['Date', 'Safer_Neighborhood_Team_Borough_Name'])
        #
        #     # 转换日期格式
        #     df_asb['Date'] = pd.to_datetime(df_asb['Date'], errors='coerce')
        #
        #     # 筛选 2023 年
        #     df_asb_2023 = df_asb[df_asb['Date'].dt.year == 2023]
        #
        #     # 统计每个区的 ASB 数量
        #     asb_counts = df_asb_2023.groupby('Safer_Neighborhood_Team_Borough_Name').size()
        #     asb_counts.name = 'ASB_2023'
        #
        #     # 合并到主表
        #     # 注意: ASB 数据里的区名可能与 MPS 不完全一致 (e.g., 'Westminster' vs 'City of Westminster')
        #     # 我们先尝试直接合并，之后再统一清洗
        #     final_df = final_df.merge(asb_counts, left_index=True, right_index=True, how='left').fillna(0)
        #     print(f"     成功提取 ASB 记录: {len(df_asb_2023)} 条")
        #
        # except Exception as e:
        #     print(f"⚠️ ASB 处理出错 (可能是文件缺失或格式问题): {e}")
        #     final_df['ASB_2023'] = 0  # 设为0以防万一

        # ==========================================
        # 4. 最终清洗与保存
        # ==========================================
        # 将索引列的名称改为 "Borough"
        final_df.index.name = 'Borough'
        # 统一行政区名称 (Standardisation)
        final_df.index = final_df.index.str.strip().str.title()

        rename_map = {
            "City Of Westminster": "Westminster",
            "Kensington And Chelsea": "Kensington and Chelsea",
            "Hammersmith And Fulham": "Hammersmith and Fulham",
            "Richmond Upon Thames": "Richmond upon Thames",
            "Kingston Upon Thames": "Kingston upon Thames",
            "Barking And Dagenham": "Barking and Dagenham",
            "City Of London": "City of London"
        }

        # 重命名并再次聚合
        final_df.rename(index=rename_map, inplace=True)
        final_df = final_df.groupby(level=0).sum()

        # 保存
        final_df.to_csv(output_file)
        print(f"\n✅ 处理完成! 数据已保存为: {output_file}")
        print("前 5 行预览:")
        print(final_df.head())

        return final_df

    except Exception as e:
        print(f"❌ 严重错误: {e}")
        return None


# 运行处理
df_crime_2023 = process_crime_2023()