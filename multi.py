#df1 = add_stats(df=df, bars1=21, bars2=63, bars3=252)
#df['daily_change'] = df['close'] — df['open']


import numpy as np
from tqdm import tqdm

myrange = np.arange(2000000)
i_2 = []

for i in tqdm(myrange, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}'):
    i_2.append(i**2)
