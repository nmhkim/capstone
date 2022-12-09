import pandas as pd 

df = pd.read_stata('gsp_data_part1.dta', convert_categoricals=False)

for y in range(2000, 2015):
    
    df_cut = df.loc[(df['syear'] == y)           & 
                    (df['isco88_2dg'] != -3)     & 
                    (df['lag_isco88_2dg'] != -3) & 
                    (df['unemployed2'] == 0)]
    
    df_tab = pd.crosstab(df_cut['isco88_2dg'], df_cut['lag_isco88_2dg'])
    
    max_range = df_tab.shape[1]

    for j in range(max_range):
        total_curr = df_tab.iloc[:, j].sum()
        isco_curr  = df_tab.iloc[j, j] / total_curr
        df.loc[(df['syear'] == y) & (df['isco_alt'] == j + 1), 'transition'] = isco_curr 

df.to_stata('gsp_data_part2.dta')