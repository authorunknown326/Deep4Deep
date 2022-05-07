import csv
import codecs
import pandas as pd 
from numpy import genfromtxt
import tensorflow as tf
import runpy
import glob, os
import numpy as np
import os


#Model#8
#Model#71
#Model#171
#Model#111
#Model#221
#Model#81

Model_Number = 81

baisesMean_padding =  18
baisesSTD_padding = 18
baisesVar_padding = 18
DeadNode_padding = 15 
gradientMax_padding =18
gradientMean_padding =18
gradientMedian_padding =18
gradientMin_padding =18
gradientSTD_padding =18
gradientSem_padding =18
gradientSkew_padding =18
gradientVar_padding =18
ImproperData_padding = 4
LogisticFun_padding = 18
Metric_padding = 2
Norm_padding = 36
outputMax_padding = 36
outputMedian_padding = 36
outputMin_padding = 36
outputSem_padding = 36
outputMean_padding = 36
outputSkew_padding = 36
outputSTD_padding = 36 
outputVar_padding = 36
TuneLearn_padding = 18
VanishingGradient_padding = 18
weightMax_padding = 18
weightMin_padding = 18
weightMedian_padding = 18
weightMean_padding = 18
weightSTD_padding = 18
weightVar_padding = 18
weightSem_padding = 18
weightSkew_padding = 18


def max_padding(features):
    if features in 'baisesMean':
        return baisesMean_padding
    
    if features in 'baisesSTD':
        return baisesSTD_padding

    if features in 'baisesVar':
        return baisesVar_padding
    
    if features in 'DeadNode':
        return DeadNode_padding
    
    if features in 'gradientMax':
        return  gradientMax_padding
    
    
    if features in 'gradientMean':
        return  gradientMean_padding
    
    if features in 'gradientMedian':
        return  gradientMedian_padding
    
    if features in 'gradientMin':
        return  gradientMin_padding
    
    if features in 'gradientSTD':
        return gradientSTD_padding

    if features in 'gradientSem':
        return  gradientSem_padding
    
    if features in 'gradientSkew':
        return gradientSkew_padding
    
    
    if features in 'gradientVar':
        return gradientVar_padding
    
    if features in 'ImproperData':
        return ImproperData_padding
    
    if features in 'LogisticFun':
        return  LogisticFun_padding

    if features in 'Metric':
        return Metric_padding
    
    if features in 'Norm':
        return Norm_padding

    if features in 'outputMax':
        return outputMax_padding
    
    if features in 'outputMedian':
        return outputMedian_padding

    if features in 'outputMin':
        return outputMin_padding
    
    if features in 'outputSem':
        return outputSem_padding

    if features in 'outputMean':
        return outputMean_padding
    
    if features in 'outputSkew':
        return  outputSkew_padding
    
    
    if features in 'outputSTD':
        return  outputSTD_padding
    
    if features in  'outputVar':
        return outputVar_padding
    
    if features in  'TuneLearn':
        return TuneLearn_padding
    
    if features in  'VanishingGradient':
        return VanishingGradient_padding
    
    if features in  'weightMax':
        return weightMax_padding
    
    if features in  'weightMin':
        return weightMin_padding
    
    if features in  'weightMedian':
        return weightMedian_padding    
    
    if features in  'weightMean':
        return weightMean_padding
    
    if features in  'weightSTD':
        return weightSTD_padding
    
    if features in 'weightVar':
        return weightVar_padding

    if features in 'weightSem':
        return weightSem_padding
    
    if features in 'weightSkew':
        return weightSkew_padding

    


def read_file(PATHX, NAME, txtName):
    maxNumber = 0
    FWrite = open("/new feature/Model#{0}/init/{1}/{2}.txt".format(Model_Number, txtName,NAME ), 'a')
    with open('{0}/{1}.txt'.format(PATHX,txtName), 'r') as f:
        for post, line in enumerate(f):
        #print(last_line)
            if post < 50:
                #line = line.strip()
                line = line.replace("0.0\t", "0.0001\t")
                line = line.replace("nan", "-1.0")
                line = line.replace("-inf", "-9999.99")
                line = line.replace("inf", "9999.99")
                line = line.replace("\t\t", "\t")
                FWrite.write(str(line))
    f.close()
    with open('{0}/{1}.txt'.format(PATHX, txtName), 'r') as ff:
        last_line = ff.readlines()[-1]
        splitLine = last_line.split('\t')
    
        if len(splitLine) > maxNumber:
            maxNumber = len(splitLine)
    ff.close()
    FWrite.close()
    return maxNumber


def check_nan(file):
    my_data = genfromtxt(file, delimiter=',')
    arr = np.array(my_data)
    if np.isnan(arr.max()) or np.isnan(arr.min()):
        print('nan', file)
        
def check_empty(txt_file):
    size = os.stat(txt_file).st_size
    counter = 0
    with open(txt_file) as f:
        lines = f.readlines() # list containing lines of file
        counter += lines.count('\n')

    if size == counter:
        return True
    return False
    
def read_cvs(NAME, txtName):
    txt_file = r"/new feature/Model#{0}/init/{1}/{2}.txt".format(Model_Number, txtName, NAME)
    csv_file = r"/new feature/Model#{0}/cvs/{1}/{2}.csv".format(Model_Number, txtName, NAME)
    
    if check_empty(txt_file):
                # open the file in the write mode
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            for row in range(0,50):
                writer.writerow('0')
    else:
        with open(txt_file, "r") as in_text:
            in_reader = csv.reader(in_text, delimiter = '\t')
            with open(csv_file, "w") as out_csv:
                out_writer = csv.writer(out_csv)
                for row in in_reader:
                    out_writer.writerow(row)
    
 
def padding(NAME, txtName):
    
    #print(r"/new feature/Model#{0}/cvs/{1}/{2}.csv".format(Model_Number, txtName, NAME))
    data = pd.read_csv(r"/new feature/Model#{0}/cvs/{1}/{2}.csv".format(Model_Number, txtName, NAME))
    if not data.empty:   
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        # Preview the first 5 lines of the loaded data 
        #print(data.head())
        #print(data)
        
        
        #my_data = genfromtxt(r"./circle_cvs/{0}.csv".format(3), delimiter=',')
        #print(my_data[0])
        my_data = data.to_numpy()
        #data = loadtxt(file,delimiter = ",")
        if np.isnan(my_data.max()) or np.isnan(my_data.min()):
            print("{0}/{1}.csv".format(txtName, NAME))
        #print(data.shape)
        
        maxlen_padding  = max_padding(txtName)
        my_data = tf.keras.preprocessing.sequence.pad_sequences(my_data, maxlen=maxlen_padding, padding="post", dtype='float64', value=0.0)
        #print(my_data.shape)
        # save array into csv file
        np.savetxt(r"/new feature/Model#{0}/padding/{1}/{2}.csv".format(Model_Number, txtName, NAME), my_data, delimiter = ",", fmt='%1.9f')


def create_dir(directory):
    # Directory
    #directory = "weightVar"
      
    # Parent Directory path
    parent_dir = "/new feature/Model#{0}/init/".format(Model_Number)
      
    # Path
    path = os.path.join(parent_dir, directory)
      
    # Create the directory
    # 'GeeksForGeeks' in
    # '/home / User / Documents'
    os.mkdir(path)
    print("Directory '% s' created" % directory)
    
    # Directory
    #directory = "weightVar"
      
    # Parent Directory path
    parent_dir = "/new feature/Model#{0}/cvs/".format(Model_Number)
      
    # Path
    path = os.path.join(parent_dir, directory)
      
    # Create the directory
    # 'GeeksForGeeks' in
    # '/home / User / Documents'
    os.mkdir(path)
    print("Directory '% s' created" % directory)
    # Directory
    #directory = "weightVar"
      
    # Parent Directory path
    parent_dir = "/new feature/Model#{0}/padding/".format(Model_Number)
      
    # Path
    path = os.path.join(parent_dir, directory)
      
    # Create the directory
    # 'GeeksForGeeks' in
    # '/home / User / Documents'
    os.mkdir(path)
    print("Directory '% s' created" % directory)
    

def label(s):
    if s == 'Loss':
        return 1
    if s == 'BatchSize':
        return 2
    if s == 'activation':
        return 3
    if s == 'weights':
        return 4
    if s == 'optimizer':
        return 5
    if s == 'LearnRate':
        return 6
    if s == 'Dropout':
        return 7






if __name__ == '__main__':
    list_dir = ['baisesMean', 'baisesSTD', 'baisesVar', 'DeadNode', 'gradientMax', 'gradientMean', 'gradientMedian', 'gradientMin', 'gradientSem', 'gradientSkew', 
                'gradientSTD', 'gradientVar', 'ImproperData', 'LogisticFun', 'Metric', 'Norm', 'outputMax', 'outputMedian', 'outputMean', 'outputSem', 'outputMin','outputSkew' ,'outputSTD', 
                'outputVar', 'TuneLearn', 'VanishingGradient','weightMax', 'weightMean', 'weightMedian', 'weightMin','weightSem', 'weightSkew', 'weightSTD', 'weightVar']

    for index in list_dir:
        create_dir(index)
        counter = 0
        maxNumber = 0
        AllDir =      ['/New Featues/Model#{0}'.format(Model_Number)]
        for current in AllDir:
            output = [dI for dI in os.listdir(current ) if os.path.isdir(os.path.join(current,dI))]
            for x in output:            
                if 'Metrics' in x :
                    read_file('{0}/{1}/'.format(current, x),x, index)
                    read_cvs(x, index)
                    padding(x, index)

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
