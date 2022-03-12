# How to install package:
   由於Prophet套件的安裝方式在windows與Linux上略有不同，因此詳細安裝方式請參考官方[github](https://github.com/facebook/prophet)
# How to use:
## predict.py
        python predict.py --start_date "2020-09-20" --predict_range "10D" --predicted_range "./for_iamge"

### What meaning with those arguments:
        --start_date:   預測起始日期，請勿超出DataBase裡的最後日期。
                     格式為"年年年年-月月-日日"。必須要加字串引號

        --predict_range:    預測未來時間的長度，後半部分為時間單位為天(D)、週(W)或月(M)
                        前半部分表示長度。必須要加字串引號
        --predicted_dir:   用來儲存預測數據及歷史數據的資料夾，若是要Demo趨勢圖請指定位置。必須要加字串引號
## showDemo.py
         python showDemo.py --predict_csv ./for_image/djusst_D.csv --history_csv ./for_image/djusst_D_DF.csv --image_name djusst_D.jpg
### What meaning with those arguments:
         --predict_csv:    預測數據的.csv表位置及檔名。無需使用字串引號
         --history_csv:    歷史數據的.csv表位置及檔名。無需使用字串引號
         --image_name:     使用這個名字儲存圖檔，若是不加則是執行完後跳出可互動視窗的顯示方式。無需使用字串引號
# Requirement 
1. sciki-learn
   - version: 0.23.2
2. pystan
   - version: 2.19.1.1
3. fbprophet
   - version: 0.7.1
4. pandas
   - version: 1.3.1
5. joblib
   - version: 1.0.1
6. pymysql
   - version: 1.0.2

# TODO: 修正期始日期不可超過DataBase中最後的日期
