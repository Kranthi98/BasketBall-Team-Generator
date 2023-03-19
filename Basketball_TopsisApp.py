# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 08:21:04 2023

@author: 11482
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import itertools
import random
import os
import itertools


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
    


class team_combos(Topsis):
    
    def __init__(self,dt):
        
        self.data = dt
    

    
    def main_function(self, sport, wgt_inits, cols, bc, cc, num_overlap):
        
        

        
        data = self.data
        self.data["Name1"] = "("+self.data.Pos+") " +self.data.Name

        if len(cc)-1+len(bc) == len(cols):
            bb = list("b"*len(bc))+list("c"*(len(cc)-1))
            print("Benefit and Cost criteria noted..")
        else:
            print("Benefit and Cost criteria doesnt align with columns")

        df = data[cols]
        mcdm = Topsis(df)
        ratings = dict()
        weghts = dict()
        for i in wgt_inits:
            wghts = mcdm.weights_initiation(i)
            weghts[i] = wghts
            ratings[i] = mcdm.fit_transform(wghts, bb)
        
        df = pd.DataFrame(ratings)
        df["Name"] = data.Name
        df["Pos"] = data.Pos
        df["Credits"] = data.Credits
        df["Per"] = list(range(len(df)))
        df["Sel"] = 1
        df["Team"] = data.Team
        df["Projection"] = data.Projection

        if sport == "Basketball":
            mins = np.ones(5)
            maxs = np.zeros(5)+4
            tm_min = np.array([3,3])
            tm_max = np.array([5,5])
            n = 8
            Proj = data.Projection.values
        elif sport == "Football":
            mins = [3,1,3,1] 
            maxs = [5,1,5,3]
            tm_min = np.array([4,4])
            tm_max = np.array([7,7])
            Proj = data.Projection.values
            n = 11
        
        
        tc = team_combos(df)
        df_rat = dict()

        for i in wgt_inits:
            dd,dd1 = tc.generate(i, mins, maxs, tm_min, tm_max, n, Proj, num_overlap)

            df_rat[i] = dd1
            
        return df, df_rat
            
        

        
    def generate(self, Rating, mins, maxs, tm_min, tm_max, n, Proj, num_overlap):
        

      
        sel1 = self.data.Per.values
        selc = self.data.Sel.values
        permuts = itertools.combinations(sel1,n)
        self.data["Name1"] = "("+self.data.Pos+") " +self.data.Name
        names_dict = self.data.Name1.to_dict()
        


        pos, _ = pd.factorize(self.data.Pos.values)
        teams, _ = pd.factorize(self.data.Team.values)
        Crs = self.data.Credits.values
        Rtgs = self.data[Rating].values
        
        print("Filtering the combinations based on team and position constraints...!!")
        
        permuts1 = [x for x in permuts if len(set(pos[list(x)])) >= self.data.Pos.nunique() and len(set(teams[list(x)])) >= self.data.Team.nunique()]
        final_pers = []

        
        final_pers = list(set(final_pers))


        final_permuts = []
        Avg_team_ratings = []
        Credits_used = []
        Avg_team_ratings = []
        Sd_team_ratings = []
        Median_team_ratings = []
        Team_FP_var = []
        Total_ratings = []
        Team_FP = []

        for j in permuts1:
            i = list(j)
            selc1 = selc[i]
            pos1 = pos[i]
            teams1 = teams[i]
            Crs1 = Crs[i]
            Rtgs1 = Rtgs[i]
            FP1 = Proj[i]
        

            if (np.bincount(pos1,selc1) <= maxs).all() and (95 <= np.sum(Crs1)<=100) and (np.all(np.bincount(teams1, selc1) <= tm_max)) and (np.all(np.bincount(teams1, selc1) >= tm_min)):
                

                final_permuts.append(j)
                Median_team_ratings.append(np.median(Rtgs1))
                Team_FP_var.append(np.var(FP1))
                Total_ratings.append(np.sum(Rtgs1))
                Avg_team_ratings.append(np.mean(Rtgs1))
                Sd_team_ratings.append(np.std(Rtgs1))
                Credits_used.append(np.sum(Crs1))
                Team_FP.append(np.sum(FP1))
                
        
        
        df2 = pd.DataFrame(final_permuts)
        df2.replace(names_dict, inplace = True)
        df2 = df2.apply(lambda x : np.sort(x), axis = 1, raw = True)
        cols = ["Player"+str(i+1) for i in range(df2.shape[1])]
        df2.columns = cols

        df2["Credits_used"] = Credits_used
        df2["Total_Ratings"] = Total_ratings
        df2["Avg_team_ratings"] = Avg_team_ratings
        df2["Std_team_ratings"] = Sd_team_ratings
        df2["Team_FP_Variance"] = Team_FP_var
        df2["Median_team_ratings"] = Median_team_ratings
        df2["Team_FP"] = Team_FP
        df2["Value"] = df2["Team_FP"]/df2["Credits_used"]
        df2["Ratings_Prop"] = df2["Total_Ratings"]/df2["Total_Ratings"].max()
        df2 = df2.sort_values(by = "Total_Ratings", ascending = False)

        final_combos = [0]
        final_combos_players = []

        final_combos_players.append(df2.iloc[0,0:8].to_list())
        
        for i in range(1,len(df2)):
            
            comb1 = df2.iloc[i,0:8].to_list()
            
            lens = np.array([len(set(comb1)-set(x)) for x in final_combos_players])
            if np.all(lens >= num_overlap):
                final_combos_players.append(comb1)
                final_combos.append(i)
            
        dff = pd.DataFrame(final_combos_players, columns = ["P"+str(i) for i in range(1,9)])
        
        return df2,dff
        







st.title("Basketball Topsis Team Generator")
st.checkbox("Use container width", value=True, key="use_container_width")


data_path = st.file_uploader("Select the file with D11 information")
if data_path is not None:
    data = pd.read_excel(data_path)
    
    sport = st.sidebar.selectbox(label = "Select Sport", options = ["Basketball", "Football"])
    wgt_inits = st.sidebar.multiselect(label = "Pick the weight initiation types", options=["SD","Entropy1","CRITIC","Combinative"])
    cols = st.sidebar.multiselect(label = "Pick the criteria columns", options = data.columns)
    bc = st.sidebar.multiselect(label = "Pick the benefit criteria columns", options = cols)
    cc = st.sidebar.multiselect(label = "Pick the cost criteria columns.Select one from benefit", options = cols)
    num_overlap = int(st.sidebar.select_slider(label = "Enter the max common players in two different teams",options = [1,2,3,4,5,6,7,8]))

    


    if len(cols) > 0 and len(bc) > 0 and len(cc) > 0 and len(wgt_inits) > 0 and num_overlap > 1 :

        mg = team_combos(data)
        ratss, combos = mg.main_function(sport, wgt_inits, cols, bc, cc, num_overlap)
        
        display_results = st.sidebar.selectbox(label = "Select to display the results", options = wgt_inits, )
        
        ratings, combos1 = st.tabs(["Player Ratings","Team Combinations"])
        with ratings:
            st.dataframe(ratss.sort_values(by = "CRITIC", ascending = False))
            st.download_button(label = "Download ratings", data = ratss,
                              file_name = "Ratings.csv")

        
        with combos1:
            st.dataframe(combos[display_results[0]],use_container_width=st.session_state.use_container_width)
            st.download_button("Download Team Combinations", combos[wgt_inits[0]])

            
            
            
            
            
            
            
            
            
