import pymysql
from collections import deque 
from src.config.loader import get_bool_env, get_int_env, get_str_env
from pymysql.cursors import DictCursor

_CURSOR_CLASS_MAP = {
    "DictCursor": DictCursor,
}

def get_connection():
    cursor_name = get_str_env("DB_CURSORCLASS", "DictCursor")
    cursor_class = _CURSOR_CLASS_MAP.get(cursor_name, DictCursor)

    return pymysql.connect(
        host=get_str_env("DB_HOST"),
        port=get_int_env("DB_PORT", 3306),
        user=get_str_env("DB_USER"),
        password=get_str_env("DB_PASSWORD"),
        database=get_str_env("DB_DATABASE"),
        charset=get_str_env("DB_CHARSET", "utf8mb4"),
        cursorclass=cursor_class,
        autocommit=get_bool_env("DB_AUTOCOMMIT", True),
    )

def get_title_contents(report_id, tenant_id, title_id):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            sql_text_topic = f'''
            select distinct tenant_name from t_esg_system_settings  where tenant_id = {tenant_id}
            '''
            cursor.execute(sql_text_topic)
            company_name = cursor.fetchall()

            text_results = []
            # 取一级标题内容
            if report_id:
                sql_text_topic = f'''
                SELECT DISTINCT
                    c.title as title,
                    c.title_id as title_id,
                    c.parent_id as parent_id,
                    c.content as content,
                    c.sub_title_name as sub_title_name,
                    c.sub_title_topic as sub_title_topic,
                    c.sub_title_content as sub_title_content,
                    c.topic_info_rules as topic_info_rules,
                    CASE
                        WHEN t.status = 'PASSED' THEN c.sub_title_raw_data
                        ELSE ''
                    END AS sub_title_raw_data
                FROM t_esg_content_text c
                LEFT JOIN t_esg_task t
                    ON t.target_id = c.id
                WHERE c.report_id = {report_id}
                  AND c.tenant_id = {tenant_id}
                  AND c.deleted = 0
                  AND c.title_id = {title_id}
                '''
            else:
                sql_text_topic = f'''
                SELECT DISTINCT
                    c.title as title,
                    c.title_id as title_id,
                    c.parent_id as parent_id,
                    c.content as content,
                    c.sub_title_name as sub_title_name,
                    c.sub_title_topic as sub_title_topic,
                    c.sub_title_content as sub_title_content,
                    c.topic_info_rules as topic_info_rules,
                    CASE
                        WHEN t.status = 'PASSED' THEN c.sub_title_raw_data
                        ELSE ''
                    END AS sub_title_raw_data
                FROM t_esg_content_text c
                LEFT JOIN t_esg_task t
                    ON t.target_id = c.id
                WHERE c.report_id is null
                  AND c.tenant_id = {tenant_id}
                  AND c.deleted = 0
                  AND c.title_id = {title_id}
                '''
            cursor.execute(sql_text_topic)
            contents = cursor.fetchall()
            sub_level_parent_ids = deque([title_id])
            text_results += contents

            # 取下级标题内容
            while sub_level_parent_ids:
                parent_id = sub_level_parent_ids.popleft()
                if report_id:
                    sql_text_topic = f'''
                    SELECT DISTINCT
                        c.title as title,
                        c.title_id as title_id,
                        c.parent_id as parent_id,
                        c.content as content,
                        c.sub_title_name as sub_title_name,
                        c.sub_title_topic as sub_title_topic,
                        c.sub_title_content as sub_title_content,
                        c.topic_info_rules as topic_info_rules,
                        CASE
                            WHEN t.status = 'PASSED' THEN c.sub_title_raw_data
                            ELSE ''
                        END AS sub_title_raw_data
                    FROM t_esg_content_text c
                    LEFT JOIN t_esg_task t
                        ON t.target_id = c.id
                    WHERE c.report_id = {report_id}
                      AND c.tenant_id = {tenant_id}
                      AND c.deleted = 0
                      AND c.parent_id = {parent_id}
                    '''
                else:
                    sql_text_topic = f'''
                    SELECT DISTINCT
                        c.title as title,
                        c.title_id as title_id,
                        c.parent_id as parent_id,
                        c.content as content,
                        c.sub_title_name as sub_title_name,
                        c.sub_title_topic as sub_title_topic,
                        c.sub_title_content as sub_title_content,
                        c.topic_info_rules as topic_info_rules,
                        CASE
                            WHEN t.status = 'PASSED' THEN c.sub_title_raw_data
                            ELSE ''
                        END AS sub_title_raw_data
                    FROM t_esg_content_text c
                    LEFT JOIN t_esg_task t
                        ON t.target_id = c.id
                    WHERE c.report_id is null
                      AND c.tenant_id = {tenant_id}
                      AND c.deleted = 0
                      AND c.parent_id = {parent_id}
                    '''
                cursor.execute(sql_text_topic)
                sub_level_contents = cursor.fetchall()
                for sub_level_content in sub_level_contents:
                    if sub_level_content['title_id'] not in sub_level_parent_ids:
                        sub_level_parent_ids.append(sub_level_content['title_id'])
                text_results += sub_level_contents
    finally:
        connection.close()
    return text_results, company_name[0]['tenant_name'] if company_name else company_name