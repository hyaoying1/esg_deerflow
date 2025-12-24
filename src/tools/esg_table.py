import ast
import pymysql
from decimal import Decimal
from collections import defaultdict

# 建立连接
def get_connection():
    return pymysql.connect(
        host="124.221.106.193",
        port=3306,
        user="root",
        password="esg-ai-2025",
        database="esg_ai",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


def sub_title_tables(sub_title_topics, report_id, tenant_id):
    tables = []
    connection = get_connection()
    try:
    # try:
        with connection.cursor() as cursor:
            for sub_title_topic in sub_title_topics:
                if sub_title_topic.upper() != 'SASB':
                    table = []
                    sql_indicator_topic = f'''SELECT DISTINCT indicator_name FROM t_esg_indicator 
                    WHERE report_id = {abs(report_id)} AND tenant_id = {tenant_id} and deleted = 0 
                    AND topic_tags LIKE '%{sub_title_topic}%'
                    '''
                    cursor.execute(sql_indicator_topic)
                    raw_indicator_results = cursor.fetchall()

                    if not raw_indicator_results:
                        print(f"⚠️ 未找到匹配议题：{sub_title_topics}")
                        continue

                    indicator_results = [list(d.values())[0] for d in raw_indicator_results]

                    for indicator_result in indicator_results:
                        sql_data = f'''SELECT * FROM t_esg_indicator_data 
                        WHERE report_id = {abs(report_id)} AND tenant_id = {tenant_id} and deleted = 0 AND name = '{indicator_result}'
                        '''
                        cursor.execute(sql_data)
                        raw_data_results = cursor.fetchall()

                        if not raw_data_results:
                            print(f"⚠️ 议题 {indicator_result} 无数据")
                            continue

                        for data_result in raw_data_results:
                            table.append({'name': indicator_result, 'year': data_result['year'], 'value': data_result['value']})

                    if not table:
                        print(f"⚠️ 议题 {sub_title_topic} 无可用表格数据")
                        return ""

                    years = sorted(set(item['year'] for item in table), reverse=True)
                    table_data = defaultdict(dict)

                    def normalize_value(value):
                        if value is None:
                            return ""
                        if isinstance(value, (int, float, Decimal)):
                            return f"{float(value):.2f}"
                        return str(value)

                    for item in table:
                        table_data[item['name']][item['year']] = normalize_value(item.get('value'))

                    header = "| 指标名称 | " + " | ".join(str(y) for y in years) + " |"
                    separator = "|-----------|" + "|".join(["-----------"] * len(years)) + "|"

                    rows = []
                    for name, values in table_data.items():
                        row = [name] + [values.get(y, "") for y in years]
                        rows.append("| " + " | ".join(row) + " |")

                    markdown_table = "\n".join([header, separator] + rows)
                    print(f"————————————{sub_title_topic}————————————")
                    print(markdown_table)
                    tables.append(markdown_table)
    finally:
        connection.close()

    return tables

if __name__ == '__main__':
    sub_title_topics = ['员工健康与安全']
    tables = sub_title_tables(sub_title_topics, -47, 1)