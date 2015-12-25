

def RunRandomForest(X_train, y_train, X_test, y_test, counter, combo):
    #
    print("Run RandomForest with Combo: ", counter+1)
    rf_model = RandomForestClassifier(n_estimators=combo['n_estimators'], max_features=combo['max_features'], min_samples_leaf=combo['min_samples_leaf'], bootstrap =combo['bootstrap'], max_leaf_nodes = combo['max_leaf_nodes'], class_weight = combo['class_weight'])    
    rf_model.fit(X_train, y_train)
    
    pred_train = rf_model.predict_proba(X_train)
    pred_test = rf_model.predict_proba(X_test)
    
    auc_train = roc_auc_score(y_train, pred_train[:,1])
    auc_test = roc_auc_score(y_test, pred_test[:,1])
    print("Near end: ", counter+1)   
    print np.array([counter+1, combo['bootstrap'], combo['n_estimators'], combo['max_features'], combo['min_samples_leaf'], combo['max_leaf_nodes'], combo['class_weight'], auc_train, auc_test])
    return np.array([counter+1, combo['bootstrap'], combo['n_estimators'], combo['max_features'], combo['min_samples_leaf'], combo['max_leaf_nodes'], combo['class_weight'], auc_train, auc_test])
    


import pandas as pd
import numpy as np
#from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import ParameterGrid
import datetime
import os
import multiprocessing as mp


if __name__ == '__main__':
        
    #change work directory
    os.chdir("D:/Project Files2/Python Efficiency/test RF using another dataset")
    file_counter = 1 # this needs to be changed if you want to run a model data
    n_processes = 4
    
    class_weight = [None, 'auto', 'subsample',{0: 1, 1: 1}]

    param_grid = {'bootstrap': [True, False],'n_estimators': [300, 500, 800, 1500], 'max_features': [3, 5, 10, 20, 30, 40, 50, 60, 70, 80], 'min_samples_leaf': [1, 5, 10, 20, 30, 50], 'max_leaf_nodes': [None, 3, 6 ,10, 15, 20], 'class_weight': class_weight}  
    
    ######## parameter and option in random forest classifier######
    
    parms_combo = ParameterGrid(param_grid)
    parms_combo_list = list(parms_combo)
    
    #type(parms_combo_list)
    #list(parms_combo)
    
    x_train_file = 'X_train_%d.csv' %(file_counter)
    y_train_file = 'y_train_%d.csv' %(file_counter)
    x_test_file = 'X_test_%d.csv' %(file_counter)
    y_test_file = 'y_test_%d.csv' %(file_counter)

    X_train = np.loadtxt(x_train_file, delimiter=',', skiprows= 0 )
    y_train = np.loadtxt(y_train_file, delimiter=',' , skiprows=0) 
    
    X_test = np.loadtxt(x_test_file, delimiter=',', skiprows= 0 )
    y_test = np.loadtxt(y_test_file, delimiter=',' , skiprows=0)       
        
    
    #initialize a pool of processes
    pool = mp.Pool(processes=n_processes)  
    
    ######!!!start !!! ######
    
    start = datetime.datetime.now()          
    #process creation-- call RandomForestLoop functions and run thry iterations...
    results = []    
    
    for counter, combo in enumerate(parms_combo):
        #
        results.append(pool.apply_async(RunRandomForest,(X_train, y_train, X_test, y_test, counter, combo)))
        
        
    print("Start multiple processing......")
    pool.close()
    pool.join()        
        
    print("End of multiple processing.")        
    
    print("Appending results...")
    #fetch outcomes from RunRandomForest function
    the_outcome = [res.get() for res in results]   
    
    
    #print('Printing the outcome...')
    #print(the_outcome)    

    columns=["Simulation Num", "Bootstrap", "Number of Trees", "max features", "mini sample leaf", "max leaf nodes", "class_weight","Training AUC","Test AUC"]    
    final_outcome = pd.DataFrame(the_outcome, columns=columns)      
    final_outcome.to_csv('Python_RF_Performance_File%d.csv' %(file_counter))


    end = datetime.datetime.now()  
    print (str((end-start).seconds))    
    