import pandas as pd
import os


def process_population_data():
    # 1. 设置文件名 (确保这个 .xlsx 文件在你的脚本同一目录下)
    file_path = "mye23tablesew.xlsx"
    sheet_name = "MYE5"  # 我们读取 MYE5 这个包含人口估算的工作表

    print(f"📖 正在从 Excel 读取数据: {file_path} (Sheet: {sheet_name})")

    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'。请确保文件在当前目录下。")
        return None

    try:
        # 2. 读取 Excel
        # header=7 表示标题在第8行 (索引为7)
        # engine='openpyxl' 用于读取 .xlsx 格式
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=7, engine='openpyxl')

        # 3. 筛选伦敦行政区
        # 伦敦行政区的代码 (Code) 都是以 "E09" 开头的
        # 先确保 Code 列是字符串格式
        df['Code'] = df['Code'].astype(str)
        df_london = df[df['Code'].str.startswith('E09')].copy()

        # 4. 提取需要的列 (名称和 2023 年人口)
        df_london.columns = df_london.columns.str.strip()  # 清除列名空格

        # 自动寻找包含 "2023" 和 "Population" 的列，或者直接指定
        target_col = 'Estimated Population mid-2023'
        if target_col not in df_london.columns:
            print("⚠️ 未找到标准列名，尝试模糊搜索...")
            for col in df_london.columns:
                if '2023' in str(col) and 'Population' in str(col):
                    target_col = col
                    break
        print(f"   - 锁定目标列: {target_col}")

        # 5. 创建最终表格
        df_pop = df_london[['Name', target_col]].copy()
        df_pop.columns = ['Borough', 'Population']

        # 6. 名称标准化 (Standardisation)
        df_pop['Borough'] = df_pop['Borough'].astype(str).str.strip().str.title()

        # 修正特殊的连接词 (与之前的文化数据保持一致)
        rename_map = {
            "City Of Westminster": "Westminster",
            "Kensington And Chelsea": "Kensington and Chelsea",
            "Hammersmith And Fulham": "Hammersmith and Fulham",
            "Richmond Upon Thames": "Richmond upon Thames",
            "Kingston Upon Thames": "Kingston upon Thames",
            "Barking And Dagenham": "Barking and Dagenham",
            "City Of London": "City of London"
        }
        df_pop['Borough'] = df_pop['Borough'].replace(rename_map)

        # 7. 设置索引并确保数值格式
        df_pop.set_index('Borough', inplace=True)

        return df_pop

    except Exception as e:
        print(f"❌ 读取 Excel 数据出错: {e}")
        print("💡 提示: 如果报错 'No module named openpyxl'，请运行: pip install openpyxl")
        return None


# ==========================================
# 执行
# ==========================================
df_population = process_population_data()

if df_population is not None:
    print(f"\n✅ 成功! 提取了 {len(df_population)} 个行政区的人口数据。")
    print(df_population.head())

    # 保存为 CSV 供后续步骤使用
    output_file = "London_Borough_Population.csv"
    df_population.to_csv(output_file)
    print(f"\n📁 已保存结果到: {output_file}")