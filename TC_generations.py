# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 07:07:12 2023

@author: 11482
"""



class team_combos(Topsis):
    
    def __init__(self,dt):
        
        self.data = dt
    

    
    def main_function(self, sport, wgt_inits, cols, bc, cc, num_overlap):
        
        
        import numpy as np
        import pandas as pd
        import easygui as eg
        from datetime import date
        
        data = self.data
        self.data["Name1"] = "("+self.data.Pos+") " +self.data.Name
        #data.sort_values(by = "Pos", inplace = True)

        # folder = eg.diropenbox(msg = "Select the folder to store the output")
        # path = folder+"\\"+"_".join(data.Team.unique())+"_"+str(date.today())+".xlsx"
        # sport = eg.choicebox(msg = "Select the sport", choices = ["Basketball", "Football"])  
        # wgt_inits = eg.multchoicebox(msg = "Pick the weight initiation types", choices=["SD","Entropy1","CRITIC","Combinative"])
        # cols = eg.multchoicebox(msg = "Pick the criteria columns", choices = data.columns)
        # bc = eg.multchoicebox(msg = "Pick the benefit criteria columns", choices = cols)
        # cc = eg.multchoicebox(msg = "Pick the cost criteria columns.Select one from benefit", choices = cols)
        # num_overlap = int(eg.enterbox(msg = "Enter the max common players in two different teams"))
        # man_players1 = eg.multchoicebox(msg = "Pick the mandatory players in the team", choices = self.data.Name1.values)
        

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
        # writer = pd.ExcelWriter(path)
        for i in wgt_inits:
            dd,dd1 = tc.generate(i, mins, maxs, tm_min, tm_max, n, Proj, num_overlap)
            # man_players = st.sidebar.multselect(label = "Pick the mandatory players in the team", options = dd.Name1.values)
            #cols = ["Player"+str(i+1) for i in range(1,9)]
            # dd.to_excel(writer, sheet_name = i, index = False)
            # dd1.to_excel(writer, sheet_name = i+str(1), index = False)
            df_rat[i] = dd1
            
        # df.to_excel(writer, sheet_name = "Player_Ratings", index = False)
        # pd.DataFrame(weghts).to_excel(writer, sheet_name = "weights", index = False)
        # writer.save()
        # writer.close()
        
        return df, df_rat
            
        

        
    def generate(self, Rating, mins, maxs, tm_min, tm_max, n, Proj, num_overlap):
        
        import easygui as eg
        import numpy as np
        import pandas as pd
        import itertools
        import random
      
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
        
      #  print("Running 25 Times..")
       # for _ in range(25):
        #    permuts2 = random.sample(permuts1,1)
         #   permuts1.remove(permuts2[0])
          #  ratings2 = [np.sum(Rtgs[list(permuts1[0])])]
            #Credits2 = [np.sum(Crs[list(permuts1[0])])]
           # for j in permuts1:     
            #    v = [1 if len(set(i).intersection(set(j))) <= num_overlap else 0 for i in permuts2]
             #   if sum(v) == len(permuts2):
              ##     ratings2.append(np.sum(Rtgs[list(j)]))
                    #Credits2.append(np.sum(Crs[list(j)]))
            
            #final_pers.extend(permuts2)
        
        final_pers = list(set(final_pers))
        #[x for x in permuts1[1:] if len(set(x).intersection(set(permuts1[0]))) < 6]
        #permuts2.append(permuts1[0])
        #pos_n = [set(pos[list(x)]) for x in permuts1]

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

        
        # man_players_count = df2[cols].apply(lambda x : sum([1 for i in man_players if i in list(x)]), axis = 1)
        # df2["man_players_count"] = man_players_count
        
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
       # df2 = df2.query(f"man_players_count >= {df2.man_players_count.max()-1}")
        #df3 = np.array(df2[cols])
        
        # num_overlap = 2

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
        
