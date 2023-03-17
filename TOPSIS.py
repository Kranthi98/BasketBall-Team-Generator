# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 06:59:11 2023

@author: 11482
"""
import numpy as np
import pandas as pd


class Topsis():
    
    """
        Class to implement the Topsis algorithm
        
        Inputs :: 
            
            data :  2D array that consists of decision matrix
            weights : weights if there are any
            
        
        Functions ::
            
            weights_initiation : Function to initiate the weights
            fit_transform : create ratings using the weights
            
    """

    def __init__(self, data, weights = None):
      
      
      
        """"Initiating the data and weights"""
          
        self.data = data 
        self.weights = weights
        
        """Checking the conditions"""
        
        if self.weights == None:
            print("Use weights_intiation method")
          
        elif(self.data.shape[1] == len(self.weights)) & (sum(self.weights) == 1):
          print("Weights are according to criteria \nUse fit_transform by giving series of benefical and cost criteria.")
        
        elif(self.data.shape[1] != len(self.weights)):
          print("Mismatch between weights and criteria. Check them again")
        
        else:
          print("Weights arent summing upto 1. Check")
    
          

    def fit_transform(self,weights,bc = None, p = 2, distance = "Minowski"):
        
        """
        Function : to use weights and get the ratings for the alternatives
        
        Inputs:
            
            weights : weights to use for ranking
            p : p value for the minowski distance formula
            distance : distance metric can be minowski or mahalanobis 
        
        """
        
        print("Generating Player ratings..!!")
        #Lambda function for normalising the data
        normalise = lambda df : (df - np.min(df,0))/(np.max(df, 0) - np.min(df, 0))
        
        #Minowski Distance
        if (bc != None) & (len(bc) == len(weights)) & (distance == "Minowski"):
        
          data1 = normalise(self.data)
        
          #Vector normalising the data
          data1 = data1/np.sqrt(np.sum(data1**2, axis = 0))
          weighted = np.multiply(data1, weights)
          
          max1 = np.max(weighted, axis = 0)
          min1 = np.min(weighted, axis = 0)
        
          bc1 = [1 if i == "b" else 0 for i in bc]
          bc2 = [0 if i == "b" else 0 for i in bc]
          
          #Positive Ideal Solution
          PIS = np.multiply(max1, bc1) + np.multiply(min1, bc2)
          #Negative Ideal Solution
          NIS = np.multiply(min1, bc1) + np.multiply(max1, bc2)
        
          #print(PIS, NIS)
          
        
          #Calculating the distances
          pos_dist = np.power(np.sum(np.power(np.abs(weighted - PIS),p), axis = 1),1/p)
          neg_dist = np.power(np.sum(np.power(np.abs(weighted - NIS),p), axis = 1), 1/p)
          
          #Generating the ranking using weights on positive distance and negative distances
          ranking = (0.1*neg_dist)/(0.1*neg_dist + 0.9*pos_dist)
        
          print("Larger the value better is the alternative")
        
          return ranking
        
        
        #Mahalanobis Distance
        elif(bc != None) & (len(bc) == len(weights)) & (distance == "Mahalanobis"):
            
            data1 = self.data
            
            #Vector normalising
            data2 = data1/np.sqrt(np.sum(np.power(data1,2),0))
            
            #Weight matrix
            wgt_matrix = np.diag(np.sqrt(weights))
            
            max1 = np.max(data2, axis = 0)
            min1 = np.min(data2, axis = 0)
            
            bc1 = [1 if i == "b" else 0 for i in bc]
            bc2 = [0 if i == "b" else 0 for i in bc]
            
            #Positive Ideal Solution
            PIS = np.multiply(max1, bc1) + np.multiply(min1, bc2)
            #Negative Ideal Solution
            NIS = np.multiply(min1, bc1) + np.multiply(max1, bc2)
            
            #Centered Matrix
            c_data2 = data2 - np.mean(data2, 0)
            s_matrix = (np.dot(np.transpose(c_data2), c_data2))/data2.shape[1]-1
            s_inv = np.linalg.inv(s_matrix)
            
            p_dis = np.dot(np.dot(np.dot(data2 - PIS, s_inv), np.transpose(wgt_matrix)), np.transpose(data2 - PIS))
            n_dis = np.dot(np.dot(np.dot(data2 - NIS, s_inv), np.transpose(wgt_matrix)), np.transpose(data2 - NIS))
            
            #Calculating the positive distance
            pos = np.diag(p_dis)
            #Calculating the negative distance
            neg = np.diag(n_dis)
            
            #Generating the rankings
            ranking = 0.1*neg/(0.1*neg+0.9*pos)
            
            return ranking
        else:
            
          print("Mismatch between criteria and Benefical cost array")
          
        
    

    
    def weights_initiation(self,type):
        
        """
            Function to initiate the weights
            
            Inputs ::
                
                type : type of weight to initiate
                
        """
        
        print("Initiating "+type+" Weights...!!")
          
        #Lambda function to normalise the data
        normalise = lambda df : (df - np.min(df,0))/(np.max(df, 0) - np.min(df, 0))
        
        df = self.data
        
        if type == "Equal":
          weights = np.ones(df.shape[1])
          weights = weights/weights.sum()
        elif(type == "Eigen"):
          pass
        elif(type == "ROC"):
          pass
        elif(type == "OWA"):
          pass
        elif(type == "Entropy1"):
          data1 = df/np.sum(df, 0)
          dff = np.sum(np.multiply(data1, np.log(data1)), 0)/np.log(data1.shape[0])
          weights = (1 - dff)/np.sum(1-dff)
        
          pass
        elif(type == "SD"):
          data_n = normalise(df)
          sds = np.std(data_n, axis = 0)
          weights = sds/sds.sum()
        elif(type == "CRITIC"):
        
          df1 = normalise(df)
          corr_coef = 1 - (np.corrcoef(np.transpose(df1)))
          sds = np.std(df, axis = 0 )
          wgts = np.multiply(sds, np.sum(corr_coef, 0))
          weights = wgts/wgts.sum()
        
        elif(type == "Entropy2"):
          pass
        elif(type == "Combinative"):
            
            #Critic
            df1 = normalise(df)
            corr_coef = 1 - (np.corrcoef(np.transpose(df1)))
            sds = np.std(df, axis = 0 )
            wgts = np.multiply(sds, np.sum(corr_coef, 0))
            weights1 = wgts/wgts.sum()
            
            #Entropy
            data1 = df/np.sum(df, 0)
            dff = np.sum(np.multiply(data1, np.log(data1)), 0)/np.log(data1.shape[0])
            weights2 = (1 - dff)/np.sum(1-dff)
            
            #Subjective Weights
            wts = np.array([1,1.2,1.5,3,3])
            weights3 = wts/wts.sum()
            
            w1w2 = np.multiply(weights1, weights2)
            w1w2w3 = np.multiply(w1w2, weights3)
            w1w2w3 = np.power(w1w2w3, 1/3)
            weights = w1w2w3/w1w2w3.sum()
            
        
        
        return weights
    



  