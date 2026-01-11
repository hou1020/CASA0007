import pandas as pd


def process_house_prices_2023():
    print("🏠 正在处理 2023 年房价数据...")
    file_path = "UK House price index.xlsx"
    sheet_name = "Average price"

    try:
        # 1. 读取数据 (注意 header 在第一行，且第一列包含无关代码)
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.iloc[1:].copy()  # 删除第一行 (代码行)

        # 2. 处理日期列
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # 3. 筛选 2023 年数据
        df_2023 = df[df['Date'].dt.year == 2023].copy()

        if df_2023.empty:
            print("⚠️ 警告: 2023 年无数据")
            return None

        # 4. 计算平均房价
        df_2023.set_index('Date', inplace=True)
        # 仅保留包含数据的列 (去除 Unnamed 列)
        df_2023 = df_2023.loc[:, ~df_2023.columns.str.contains('^Unnamed')]

        # 强制转为数值型
        for col in df_2023.columns:
            df_2023[col] = pd.to_numeric(df_2023[col], errors='coerce')

        # 计算每列均值 (即每个区的年度均价)
        avg_prices = df_2023.mean(axis=0)

        # 5. 构建结果表格
        df_final = pd.DataFrame(avg_prices, columns=['House_Price_2023'])
        df_final.index.name = 'Borough'

        # 6. 清洗 Borough 名称 (Standardisation)
        # 这一步至关重要，否则无法和其他表合并
        new_index = []
        for borough in df_final.index:
            b = str(borough).strip()
            if '&' in b: b = b.replace('&', 'and')  # Barking & Dagenham -> Barking and Dagenham

            # 特殊修正
            if b == "City of Westminster":
                b = "Westminster"
            elif "Richmond" in b:
                b = "Richmond upon Thames"
            elif "Kingston" in b and "Hull" not in b:
                b = "Kingston upon Thames"

            new_index.append(b)
        df_final.index = pd.Index(new_index, name='Borough')

        # 7. 剔除无效行 (NaN 和 区域汇总行)
        df_final.dropna(inplace=True)
        regions_to_drop = [
            "Inner London", "Outer London", "London", "England",
            "North East", "North West", "Yorks and The Humber",
            "East Midlands", "West Midlands", "East of England",
            "South East", "South West", "United Kingdom"
        ]
        # 反向筛选 (不包含在排除列表中的)
        df_final = df_final[~df_final.index.str.title().isin([x.title() for x in regions_to_drop])]

        return df_final.astype(int)

    except Exception as e:
        print(f"❌ 处理出错: {e}")
        return None


# 运行并保存
df_housing = process_house_prices_2023()

if df_housing is not None:
    print(f"\n✅ 成功! 提取了 {len(df_housing)} 个区域的房价。")
    print(df_housing.head())
    df_housing.to_csv("London_Borough_House_Prices_2023.csv")
    print("📁 已保存为: London_Borough_House_Prices_2023.csv")