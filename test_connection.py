import ast
import pymysql
from decimal import Decimal
from collections import defaultdict

connection = pymysql.connect(
    host="124.221.106.193",  # 只写 IP 或域名
    port=3306,  # 数据库主机名
    user="root",  # 用户名
    password="esg-ai-2025",  # 密码
    database='esg_ai',
    charset="utf8mb4",  # 字符集
    cursorclass=pymysql.cursors.DictCursor  # 返回字典形式结果
)
try:
    with connection.cursor() as cursor:
        sql_text_topic = f'''
        SELECT distinct *
                    FROM t_esg_content_text where title_id = 11
                    '''
        cursor.execute(sql_text_topic)
        text_results = cursor.fetchall()
finally:
    connection.close()
print(text_results)