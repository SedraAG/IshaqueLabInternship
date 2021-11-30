import numpy as np
import pandas as pd
from Bio import SeqIO
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from Bio.SeqUtils import GC
plt.style.use('ggplot')

# calculate the index of the beginning of each chr in df
# df has to have columns named chr_ind and chr 

def index_of_chr(df):
    indices = df.groupby(["chr_ind"]).count()['chr']
    my_list = []
    for i in range (1,len(indices)+1):
        if i == 1:
            my_list.append(indices[1]-1)
        else:
            indices[i] = indices[i] + indices[i-1]
            my_list.append(indices[i] -1)    
    return my_list
    
 

# +1 to avoid dividing by zero
# s is the start of the bin and e is the end of the bin and a is the sequence
# of the chromosome wehre the bin is located


def countGC(s,e,a):
    content = a[s:e]
    na = content.count("A")
    nt = content.count("T")
    ng = content.count("G")
    nc = content.count("C")
    nat = na + nt
    ncg = ng + nc
    return (ncg/((ncg+nat)+1))


def countGC1(s,e,a):
    content = a[s:e]
    return GC(content)

# the difference between countGC and countGC1 is that countGC does not include the NNN..
# bases in the calculations, so we only calculate (C+G)/(A+G+C+T), while countGC1
# includes the N bease in the calculations. (G+C)/(A+G+C+T+N)calculate GC_content of
# each bin in each chromosom and add them all together in one list


def countCGall(chr_list,indices,df):
    chr_cg = []
    for i in range (22):
        chromosom = chr_list[i]
        index = indices[i] + 1         
        if i == 0 :
            for j in range (0,index):
                chr_cg.append(countGC(df.bin_start[j],df.bin_end[j],chromosom))   
        else:
            last_index = indices[i-1] + 1
            for j in range (last_index, index):
                chr_cg.append(countGC(df.bin_start[j], df.bin_end[j], chromosom))
    return chr_cg


def countCGall1(chr_list,indices,df):
    chr_cg = []
    for i in range (22):
        chromosom = chr_list[i]
        index = indices[i] + 1     
        if i == 0 :
            for j in range (0,index):
                chr_cg.append(countGC1(df.bin_start[j],df.bin_end[j],chromosom))   
        else:
            last_index = indices[i-1] + 1
            for j in range (last_index, index):
                chr_cg.append(countGC1(df.bin_start[j],df.bin_end[j],chromosom))
    return chr_cg


# make a function for plots

##
##
##  

# now we want to nomalise the counts to the mean, median and mode
# -> we have more than one mode, so we can take the mean of those
# we color different chromosomes with different colors


def normalise_to_fixed_value(counts,fixed_value):
    if fixed_value == "mean":
        mean_normalised_counts = counts * (2 / counts.mean())
        return mean_normalised_counts
    elif fixed_value == "median":
        median_normalised_counts = counts * (2 / counts.median())
        return median_normalised_counts
    elif fixed_value == "mode":
        mode_normalised_counts = counts * (2/counts.mode().mean())
        return mode_normalised_counts
    else: return




def estimate_copy_number(df):
    user_input = input("mean, median or mode? ")
    if (user_input == 'mean'): 
        copy_nr_mean = ((df.groupby(['chr_ind']).mean()).mean_normalised_counts).round(decimals = 0)
        return copy_nr_mean
    elif (user_input == 'median'):
        copy_nr_median=((df.groupby(['chr_ind']).median()).median_normalised_counts).round(decimals = 0)
        return copy_nr_median
    elif (user_input == 'mode'): 
        copy_nr_median = ((df.groupby(['chr_ind']).median()).mode_normalised_counts).round(decimals = 0)
        return copy_nr_median
    else: print('please try again! ')

# GC-content normalisation:

# for the correlation curve:
# we cut the big interval into 360 samller intervall (i variabel in the for  loop) 
# and calculate the GC-conten median in each interval and add the value to the list 
# cg_window.
# the size of the window is also variable (we have 0,025)
# we also add the median of the counts to the median_counts list



## plot function
def smoothplot(df,rs,re,steps):
    median_counts = []
    cg_window = []
    for i in range (rs,re,steps):
        median_counts.append(df.loc[(df.GC_content >= (i/10000)-0.01) &
                                     (df.GC_content <= (i/10000)+0.015)].counts.median())
        cg_window.append(df.loc[(df.GC_content >= (i/10000)-0.01) &
                                     (df.GC_content <= (i/10000)+0.015)].GC_content.median())

# normalization for GC content:
# the function gc_correction normalises the read counts to the GC-content and has the 
# parameters: 
# df:the data frame
# GC_content: the column on df which contains the gc_content
# counts: the column on df, which contains read counts 
# start: the samllest value in GC-content * decimal_place
# end: the greatest value in GC-content * decimal_place
# step: distance between the points that should be taken
# decimal_place: in our case is 10000, because we have values from 0.345 to 0.52 
# and we want the for loop to iterate through all the points with distance
# of 0.0005 = decimal_step
# window_size: the size of the window from where we calculate the median


def gc_correction(df,GC_content,counts,start,end,step,decimal_place,window_size):
    window_size = window_size / 2
    gc_normlised_counts = np.zeros(len(df))
    decimal_step = step / decimal_place
    df['gc_normlised_counts'] = gc_normlised_counts
    for i in range (start,end,step):
        df.loc[((GC_content >= (i/decimal_place)) & (GC_content < ((i/decimal_place) + decimal_step))),'gc_normlised_counts'] = df.loc[(GC_content >= (i/decimal_place)) & (GC_content < ((i/decimal_place) + decimal_step))].counts / (df.loc[(GC_content >= ((i/decimal_place) - window_size)) & (GC_content < ((i/decimal_place) + decimal_step + window_size))].counts.median())*2    

# if __name__== "__main__" :

