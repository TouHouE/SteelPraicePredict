import datetime

import pymysql
import pandas as pd
import numpy as np
# import charts

class ToDoTabeManager(object):
    def __init__(self):
        self.db_settings = {
            "host": "124.218.89.163",
            "port": 13306,
            "user": "nkust",
            "db": "taya",
            "password": "(07)3814526",
            "charset": "utf8"
        }
    
    def connect_to_db(self):
        try:
            # 建立Connection物件
            self.conn = pymysql.connect(**self.db_settings)
            self.cursor = self.conn.cursor()
        except Exception as ex:
            print(ex)
    
    def show_db(self):
        sql ="show databases"
        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        return info

    def load_table(self, table_name) -> pd.DataFrame:
        sql ="select * from {}".format(table_name)
        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        df = pd.DataFrame(info)

        sql = f"describe {table_name}"
        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        col_names = np.array(info)[:, 0]
        df.columns = col_names
        try:
            df = df[['c_name', 'c_date', 'c_price', 'c_open', 'c_high', 'c_low']]
        except:
            pass

        return df


    def load_bom_tabe(self):
        sql = "select * from test0811" #TODO 編輯資料庫名稱bom_tabe

        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        return info

    def load_medium_tabe(self):
        sql = "select * from test0811"#TODO 編輯資料庫名稱medium_tabe

        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        return info

    def load_to_do_list_tabe(self):
        sql = "select * from test0811"#TODO 編輯資料庫名稱to_do_list

        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        return info

    def load_parts_inventory_tabe(self):
        sql = "select * from test0811"#TODO 編輯資料庫名稱parts_inventory_tabe

        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        return info

    def delete_all(self, table_name):
        command = f'delete from {table_name}'
        self.cursor.execute(command)

    def insert(self, table_name: str, info):
        y = info['y_hat'] if 'y_hat' in info.keys() else info['yhat']
        key_part = '(c_name, p_price, p_date, created_user, created_dt)'
        values_part = f"(\"{info['c_name']}\", \"{y}\", \"{info['ds']}\", \"31\", \"{datetime.datetime.now()}\")"
        command = f'insert into {table_name} {key_part} values {values_part}'

        self.cursor.execute(command)
        self.conn.commit()

keys = ['台灣鋼鐵指數', '道瓊鋼鐵指數', '鋼筋價格指數']

# if __name__ == '__main__':
    # dbmanager = ToDoTabeManager()
    # dbmanager.connect_to_db()
    # print("===============================")
    #
    # df = dbmanager.load_tabe("commodity_daily")
    # # df.to_csv("commodity_daily.csv")
    # print(df)
    #
    # df_tw = df[df['c_name'] == keys[0]]
    # df_dj = df[df['c_name'] == keys[1]]
    # df_fe = df[df['c_name'] == keys[2]]

