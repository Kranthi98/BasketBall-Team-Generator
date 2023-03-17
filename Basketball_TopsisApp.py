# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 08:21:04 2023

@author: 11482
"""

import streamlit as st
import pandas as pd
import easygui as eg
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
    


# man_players1 = eg.multchoicebox(msg = "Pick the mandatory players in the team", choices = self.data.Name1.values)


    if len(cols) > 0 and len(bc) > 0 and len(cc) > 0 and len(wgt_inits) > 0 and num_overlap > 1 :
        code1 = r"C:\Users\11482\OneDrive - Kantar\Desktop\LeAAP\Batch-2_DS\WhitePaper\Topsis.py"
        code2 = r"C:\Users\11482\OneDrive - Kantar\Desktop\LeAAP\Batch-2_DS\WhitePaper\TC_generations.py"
        
        import os
        # execfile(code1)
        # execfile(code2)
              
        exec(open(code1).read())
        exec(open(code2).read())
        
        
        import itertools
        import pandas as pd
        import numpy as np
        #d2 = pd.read_clipboard()
        
        mg = team_combos(data)
        ratss, combos = mg.main_function(sport, wgt_inits, cols, bc, cc, num_overlap)
        
        ratings, combos1 = st.tabs(["Tab1","Tab2"])
        with ratings:
            st.dataframe(ratss.sort_values(by = "CRITIC"))
        
        with combos1:
            st.dataframe(combos[wgt_inits[0]],use_container_width=st.session_state.use_container_width)    