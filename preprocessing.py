import pandas as pd
from dataio import *
from sklearn.preprocessing import StandardScaler

def nullstats():
    for i in range(ntables):
        df = tables[i]
        nulllabels = (df.index.isna().sum(), df.columns.isna().sum())
        nullrows = df.shape[0] - df.dropna(0).shape[0]
        nullcols = df.shape[1] - df.dropna(1).shape[1]
        allnullrows = df.shape[0] - df.dropna(0, how='all').shape[0]
        allnullcols = df.shape[1] - df.dropna(1, how='all').shape[1]
        nullcolpercent = nullcols / df.shape[1] * 100
        nullpercent = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        print('Table:                    ', tablenames[i])
        print('Shape:                    ', df.shape)
        print('Contains null:            ', (nullrows, nullcols))
        print(f"% of columns contain null: {nullcolpercent:.2f}%")
        print(f"% of nulls in total:       {nullpercent:.2f}%")
        print('')

def nullcount(t):
    nullsums = []
    for i in range(t.shape[1]):
        nullsum = t.iloc[:,i].isna().sum()
        nullsums.append(nullsum)
    nullsums.sort()
    write(nullsums, 'nullsums.pickle')
    print(nullsums)

matrix = read('orig/gdsc_drug_cell_IC50_matrix.DataFrame.Pickle')
cnv    = read('orig/gdsc_cell_feature_CNV.DataFrame.Pickle')
crp    = read('orig/gdsc_cell_feature_CRP.DataFrame.Pickle')
exp    = read('orig/gdsc_cell_feature_Exp.DataFrame.Pickle')
prot   = read('orig/gdsc_cell_feature_Prot.DataFrame.Pickle')

tables = [matrix, cnv, crp, exp, prot]
tablenames = ['matrix', 'cnv', 'crp', 'exp', 'prot']
ntables = len(tables)

#nullstats()

joined = matrix\
    .join(cnv, how='inner', rsuffix='_cnv')\
    .join(crp, how='inner', rsuffix='_crp')\
    .join(exp, how='inner', rsuffix='_exp')\
    .join(prot, how='inner', rsuffix='_prot')

for i in range(ntables):
    ncols = tables[i].shape[1]
    tables[i] = joined.iloc[:, :ncols]
    joined = joined.iloc[:, ncols:]

#nullstats()
#nullcount(tables[0])

#print(cnv.sort_values(by='DDX11L1', ascending=False).iloc[:10,:5])
#print(exp.iloc[:5,:3])

for i in range(ntables):
    t = tables[i].dropna(axis=1).to_numpy()
    if i > 0:
        t = StandardScaler().fit_transform(t)
    write(t, f"{tablenames[i]}.pickle")
