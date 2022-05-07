
import pandas as pd
a = [37213388,
37624102,
39810655,
41327601,
41600519,
44998910,
46642627,
46995209,
51181393,
56380303]

for model_number in range(1, 12):
    if model_number != 4:
        
        #model_number = 25
        print(model_number)
        
        #load datasets from Excel file
        df_1 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientMean/Metrics.csv".format(model_number))
        df_2 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientSTD/Metrics.csv".format(model_number))
        df_3 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientVar/Metrics.csv".format(model_number))                 
        df_4 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientMax/Metrics.csv".format(model_number))
        df_5 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientMedian/Metrics.csv".format(model_number))
        df_6 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientMin/Metrics.csv".format(model_number))
        df_7 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientSem/Metrics.csv".format(model_number))
        df_8 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/gradientSkew/Metrics.csv".format(model_number))         
        df_9 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/LogisticFun/Metrics.csv".format(model_number))
        df_10 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/Metric/Metrics.csv".format(model_number))
        df_11 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/Norm/Metrics.csv".format(model_number))                 
        df_12 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputMean/Metrics.csv".format(model_number))
        df_13 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputSTD/Metrics.csv".format(model_number))
        df_14 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputVar/Metrics.csv".format(model_number))
        df_15 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputMax/Metrics.csv".format(model_number))
        df_16 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputMedian/Metrics.csv".format(model_number))
        df_17 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputMin/Metrics.csv".format(model_number))
        df_18 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputSem/Metrics.csv".format(model_number))
        df_19 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/outputSkew/Metrics.csv".format(model_number))                   
        df_20 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/TuneLearn/Metrics.csv".format(model_number))
        df_21 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/VanishingGradient/Metrics.csv".format(model_number))                    
        df_22 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightMean/Metrics.csv".format(model_number))
        df_23 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightSTD/Metrics.csv".format(model_number))
        df_24 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightVar/Metrics.csv".format(model_number))
        df_25 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightMax/Metrics.csv".format(model_number))
        df_26 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightMedian/Metrics.csv".format(model_number))
        df_27 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightMin/Metrics.csv".format(model_number))
        df_28 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightSem/Metrics.csv".format(model_number))
        df_29 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/weightSkew/Metrics.csv".format(model_number))                   
        df_30 = pd.read_csv("/new feature/Evaluate/GH{0}/padding/DeadNode/Metrics.csv".format(model_number))
        
        #merge datasets
        #df_outer = pd.concat([df_1, df_2,df_3,df_4, df_5,df_6,df_7, df_8,df_9,df_10, df_11,df_12,df_13, df_14], axis=1)
        df_outer = pd.concat([df_1, df_2,df_3,df_4, df_5,df_6,df_7, df_8,df_9,df_10,df_11, df_12,df_13,df_14, df_15,df_16,df_17, df_18,df_19,df_20,df_21, df_22,df_23,df_24, df_25,df_26,df_27, df_28,df_29,df_30], axis=1)
        
        #df_combine = df_combine.merge(df_3, on='id', how='outer')
        #output back into Excel
        df_outer.to_csv("/new feature/Evaluate/GH{0}/merge/Metrics.csv".format(model_number), header=False)
        
        
        df = pd.read_csv("/new feature/Evaluate/GH{0}/merge/Metrics.csv".format(model_number))
        # If you know the name of the column skip this
        first_column = df.columns[0]
        # Delete first
        df = df.drop([first_column], axis=1)
        df.to_csv("/new feature/Evaluate/GH{0}/merge/Metrics.csv".format(model_number), index=False, header=False)