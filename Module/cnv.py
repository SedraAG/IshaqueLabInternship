from ast import arg
from curses import window
import numpy as np
import pandas as pd
import os
from Bio import SeqIO
from matplotlib import pyplot as plt
import seaborn as sns
from Bio.SeqUtils import GC
plt.style.use('ggplot')
from scipy.interpolate import interp1d
import statsmodels.api as sm
import argparse


# calculate the end index of each chr in df
# requires 2 different columns: chr_ind to group the df by it and any other column 

def add_column(df,name):
    if (not (name in df)):
        df[name] = np.nan


def index_of_chr(df,chr_ind,chr):
    indices = df.groupby([chr_ind]).count()[chr]
    my_list = []
    for i in range (1,len(indices)+1):
        if i == 1:
            my_list.append(indices[i])
        else:
            indices[i] = indices[i] + indices[i-1]
            my_list.append(indices[i])    
    return my_list
    


# s is the start of the bin and e is the end of the bin and a is the sequence
# of the chromosome wehre the bin is located

def countGC(start,end,sequence):
    """ calculate the GC-cotent  of a sequnce
    
    ignores N-bases in calculations"""
    content = sequence[start:end]
    na = content.count("A")
    nt = content.count("T")
    ng = content.count("G")
    nc = content.count("C")
    nat = na + nt
    ncg = ng + nc
    return (ncg/((ncg+nat)+1)) # +1 to avoid dividing by zero

def countGC1(start,end,sequence):
    """ calculate the GC-cotent  of a sequnce

    includes N-bases in calculations"""
    content = sequence[start:end]
    return GC(content)*100

# the difference between countGC and countGC1 is that countGC does not include the NNN..
# bases in the calculations, so we only calculate (C+G)/(A+G+C+T), while countGC1
# includes the N bease in the calculations. (G+C)/(A+G+C+T+N)calculate GC_content of
# each bin in each chromosom and add them all together in one list

def countCGall(chr_list,indices,df,number_of_chr):
    """ calculate the GC-cotent  of a sequnce
    
    ignores N-bases in calculations"""
    chr_cg = []
    for i in range (number_of_chr):  # for each chr
        chromosom = chr_list[i]
        index = indices[i]       
        if i == 0 :
            for j in range (0,index):
                chr_cg.append(countGC(df.bin_start[j],df.bin_end[j],chromosom))   
        else:
            last_index = indices[i-1]
            for j in range (last_index, index):
                chr_cg.append(countGC(df.bin_start[j], df.bin_end[j], chromosom))
    return chr_cg


def countCGall1(chr_list,indices,df):
    """ calculate the GC-cotent  of a sequnce
    
    includes N-bases in calculations"""
    chr_cg = []
    for i in range (22):
        chromosom = chr_list[i]
        index = indices[i]    
        if i == 0 :
            for j in range (0,index):
                chr_cg.append(countGC1(df.bin_start[j],df.bin_end[j],chromosom))   
        else:
            last_index = indices[i-1] 
            for j in range (last_index, index):
                chr_cg.append(countGC1(df.bin_start[j],df.bin_end[j],chromosom))
    return chr_cg


def normalise_to_fixed_value(df,counts,fixed_value,type):
    if fixed_value == "mean":
        mean_normalised_counts = counts * (2 / counts.mean())
        df["mean_normalised" + type +"_counts"] = mean_normalised_counts
    elif fixed_value == "median":
        median_normalised_counts = counts * (2 / counts.median())
        df["median_normalised" + type +"_counts"] = median_normalised_counts
    elif fixed_value == "mode":
        mode_normalised_counts = counts * (2/counts.mode().mean())
        df["mode_normalised" + type +"_counts"] = mode_normalised_counts

# requires colum 'chr_ind' 

def estimate_copy_number(df,fixed_value,column):
    if (fixed_value=='mean'): 
        copy_nr_mean=df.groupby(['chr_ind']).mean().round(decimals=0)
        return copy_nr_mean[column]
    elif (fixed_value=='median'):
        copy_nr_median=df.groupby(['chr_ind']).median().round(decimals=0)
        return copy_nr_median[column]
    elif (fixed_value=='mode'):
        copy_nr_median=df.groupby(['chr_ind']).median().round(decimals = 0)
        return copy_nr_median

# GC-content normalisation:
# for the correlation curve:
# we cut the big interval into 360 samller intervall (i variabel in the for  loop) 
# and calculate the GC-conten median in each interval and add the value to the list 
# cg_window.
# the size of the window is also variable (we have 0,025)
# we also add the median of the counts to the median_counts list



# rs: range start
# re: range end
# steps: ~how many points 
# pl
def smoothplot(df,rs,re,window_size):
    median_counts = []
    cg_window = []
    for i in range (rs,re,window_size):
        median_counts.append(df.loc[(df.GC_content >= (i/10000)-0.01) &
                                     (df.GC_content <= (i/10000)+0.015)].counts.median())
        cg_window.append(df.loc[(df.GC_content >= (i/10000)-0.01) &
                                     (df.GC_content <= (i/10000)+0.015)].GC_content.median())
    return median_counts, cg_window

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

def gc_ncorrection(df,GC_content,start,end,step,decimal_place,window_size):
    window_size = window_size / 2
    gc_normlised_counts = np.zeros(len(df))
    decimal_step = step / decimal_place
    df['gc_normalised_normal_counts'] = gc_normlised_counts
    for i in range (start,end,step):
        df.loc[((GC_content >= (i/decimal_place)) & 
        (GC_content < ((i/decimal_place) + decimal_step))),'gc_normalised_normal_counts'] = df.loc[(GC_content >= (i/decimal_place)) & (GC_content < ((i/decimal_place) + decimal_step))].normal_counts / (df.loc[(GC_content >= ((i/decimal_place) - window_size)) & (GC_content < ((i/decimal_place) + decimal_step + window_size))].normal_counts.median()) * 2551.4

def gc_ccorrection(df,GC_content,start,end,step,decimal_place,window_size):
    window_size = window_size / 2
    gc_normlised_counts = np.zeros(len(df))
    decimal_step = step / decimal_place
    df['gc_normalised_cancer_counts'] = gc_normlised_counts
    for i in range (start,end,step):
        df.loc[((GC_content >= (i/decimal_place)) & (GC_content < ((i/decimal_place) + decimal_step))),'gc_normalised_cancer_counts'] = df.loc[(GC_content >= (i/decimal_place)) & (GC_content < ((i/decimal_place) + decimal_step))].cancer_counts / (df.loc[(GC_content >= ((i/decimal_place) - window_size)) & (GC_content < ((i/decimal_place) + decimal_step + window_size))].cancer_counts.median())* 1804.3

def gc_correction(df,GC_content,start,end,step,decimal_place,window_size):
    window_size = window_size / 2
    gc_normlised_counts = np.zeros(len(df))
    decimal_step = step / decimal_place
    df['gc_normlised_mixed_counts'] = gc_normlised_counts
    for i in range (start,end,step):
        df.loc[((GC_content >= (i/decimal_place)) & (GC_content < ((i/decimal_place) + decimal_step))),'gc_normlised_mixed_counts'] = df.loc[(GC_content >= (i/decimal_place)) & (GC_content < ((i/decimal_place) + decimal_step))].mixed_counts / (df.loc[(GC_content >= ((i/decimal_place) - window_size)) & (GC_content < ((i/decimal_place) + decimal_step + window_size))].mixed_counts.median())*2   

def gc_correction_withh_lowess(df, counts, frac):
    lowess = sm.nonparametric.lowess(df[counts],df['GC_content'],frac=frac)
    copy = df.sort_values("GC_content", ignore_index = True)
    copy['lowess_gc_corrected_cancer_counts'] = copy[counts] / (list(zip(*lowess))[1])
    copy = copy.sort_values("bin_ind", ignore_index = True)
    return copy['lowess_gc_corrected_cancer_counts'] 
    
def comma_sep_data_processing(path):
    return(pd.read_csv(path))

def space_sep_data_processing(path):
    return(pd.read_csv(path,sep = '\s+'))

def simple_scatter_plot(df,x,y,x_label,y_label,titel): 
    plt.figure(figsize=(16, 4))
    plt.scatter(x, y, s = 25)
    if (x is df["bin_ind"]):
        indices = index_of_chr(df, "chr_ind", "bin_start")
        del indices[-1:]
        for i in indices:
            plt.axvline(x=df.bin_ind[i], ymin=0, ymax=1, c='black')
    plt.title(titel, fontdict=None,fontsize=30, loc='center', pad=None)
    plt.xlabel(x_label, size = 25)
    plt.ylabel(y_label, size = 25)
    plt.savefig(titel+'.png', dpi=300, bbox_inches='tight')
    plt.close()


params = {'legend.fontsize': 'x-large',
          'figure.titlesize': 20,
          "figure.autolayout": True,
         'scatter.marker': '.',
         'scatter.edgecolors': 'black'}
plt.rcParams.update(params)
    


def main():
    number_of_chr = len(list(filter(lambda f: f.endswith('.fna'), os.listdir('.')))) 
    parser  = argparse.ArgumentParser() 
    parser.add_argument('-NC','--normal_counts_file', type = argparse.FileType('r'), help = "counts per bin of normal tissue as csv",required=True)
    parser.add_argument('-CC','--cancer_counts_file', type = argparse.FileType('r'), help = "counts per bin of cancer tissue as csv",required= True)
    parser.add_argument('-B','--bins_file', type = argparse.FileType('r'), help = "bins file",required=True)
    parser.add_argument('-cm','--gc_correction_method', type = str, help = "GC-corrction method",default = "Lowess")
    parser.add_argument('-T','--outlier_threshold', type = int, help = "outlier threshold, default is 10000", default=10000)
    parser.add_argument('-dn','--desired_number', type = int, help = "desired_number of chromosomes to be normalised",required=True)
    parser.add_argument('-fr','--fraction', type = float, help = "fhe fraction of the data used when estimating each y-value with Lowess (between 0 and 1)", default=.3)

    args = parser.parse_args()
    desired_number = args.desired_number
    frac = args.fraction    
    normal_df = comma_sep_data_processing(args.normal_counts_file)
    normal_df = normal_df.set_axis(['bin_ind', 'normal_counts'], axis=1) 
    cancer_df = comma_sep_data_processing(args.cancer_counts_file)
    cancer_df = cancer_df.set_axis(['bin_ind', 'cancer_counts'], axis=1)
    bin_df = space_sep_data_processing(args.bins_file)
    df = pd.merge(bin_df,normal_df,on = 'bin_ind', how = 'inner')
    df = pd.merge(df,cancer_df,on = 'bin_ind', how = 'inner')
    threshold = args.outlier_threshold 
    df = df.drop(df[df.normal_counts > threshold].index)
    df = df.reset_index(drop=True)
    #primary plot with geosomes
    simple_scatter_plot(df,df['bin_ind'], df['cancer_counts'],'bins','cancer counts','Primary plot with genosomes')
    #remove genosomes
    while (number_of_chr > desired_number):
        df = df.drop(df[df.chr_ind == number_of_chr].index)
        df = df.reset_index(drop=True)
        number_of_chr = number_of_chr-1
    simple_scatter_plot(df,df['bin_ind'], df['cancer_counts'],'bins','cancer counts','Primary plot with user given chromosomes')
    indices = index_of_chr(df, "chr_ind", "bin_start")
    chr_list = [(SeqIO.read(f"chr{i}.fna", "fasta")).seq  for i in range (1,number_of_chr+1)]
    df['GC_content'] = countCGall(chr_list,indices,df,number_of_chr)
    df = df.drop(df[df.GC_content == 0].index)
    df = df.reset_index(drop=True)
    simple_scatter_plot(df,df["GC_content"],df["cancer_counts"],"GC-Content","cancer counts","GC-content in cancer counts")
    if (args.gc_correction_method) == "Lowess":
        add_column(df,"Lowess_GC_correctd_cancer_counts")
        add_column(df,"Lowess_GC_correctd_normal_counts")
        df["Lowess_GC_correctd_cancer_counts"] = gc_correction_withh_lowess(df, "cancer_counts", frac)
        df["Lowess_GC_correctd_normal_counts"] = gc_correction_withh_lowess(df, "normal_counts", frac)
        lowess_gc_cancer = sm.nonparametric.lowess(df['Lowess_GC_correctd_cancer_counts'],df['GC_content'], frac=.3)
        lowess_x_gc_cancer = list(zip(*lowess_gc_cancer))[0]
        lowess_y_gc_cancer = [element * df['cancer_counts'].median() for element in list(zip(*lowess_gc_cancer))[1]]
        lowess_gc_normal = sm.nonparametric.lowess(df['Lowess_GC_correctd_normal_counts'],df['GC_content'], frac=.3)
        lowess_x_gc_normal = list(zip(*lowess_gc_normal))[0]
        lowess_y_gc_normal = [element * df['normal_counts'].median() for element in list(zip(*lowess_gc_normal))[1]]
        lowess_cancer = sm.nonparametric.lowess(df['cancer_counts'],df['GC_content'], frac=.3)
        lowess_x_cancer = list(zip(*lowess_cancer))[0]
        lowess_y_cancer = list(zip(*lowess_cancer))[1]
        lowess_normal = sm.nonparametric.lowess(df['normal_counts'],df['GC_content'], frac=.3)
        lowess_x_normal = list(zip(*lowess_normal))[0]
        lowess_y_normal = list(zip(*lowess_normal))[1]
        fig1,axs1=plt.subplots(4,sharex = True)
        fig1.suptitle('Plot 1')
        axs1[0].scatter(df['GC_content'], df['cancer_counts'])
        axs1[0].plot(lowess_x_cancer, lowess_y_cancer, '-b', linewidth = 1)
        axs1[2].scatter(df['GC_content'],df['normal_counts'])
        axs1[2].plot(lowess_x_normal, lowess_y_normal, '-b', linewidth = 1)
        axs1[1].scatter(df['GC_content'], df['Lowess_GC_correctd_cancer_counts']*df['cancer_counts'].median())
        axs1[1].plot(lowess_x_gc_cancer,lowess_y_gc_cancer, '-b', linewidth = 1)
        axs1[3].scatter(df['GC_content'], df['Lowess_GC_correctd_normal_counts']*df['normal_counts'].median())
        axs1[3].plot(lowess_x_gc_normal,lowess_y_gc_normal, '-b', linewidth = 1)
        axs1[0].set_ylabel('cancer counts',fontsize=7)
        axs1[1].set_ylabel('corrected GC cancer',fontsize=7)
        axs1[2].set_ylabel('normal counts', fontsize=7)
        axs1[3].set_ylabel('corrected GC normal',fontsize=7)
        axs1[3].set_xlabel('GC_content')
        plt.savefig('plot1.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig2, axs2 = plt.subplots(5,sharex=True)
        fig2.suptitle('Plot 2')
        plt.rcParams["figure.figsize"] = [30,10]
        axs2[0].scatter(df['bin_ind'], df['cancer_counts'])
        axs2[1].scatter(df['bin_ind'], df['Lowess_GC_correctd_cancer_counts']*df['cancer_counts'].median())
        axs2[2].scatter(df['bin_ind'], df['normal_counts'])
        axs2[3].scatter(df['bin_ind'], df['Lowess_GC_correctd_normal_counts']*df['normal_counts'].median())
        axs2[4].scatter(df['bin_ind'], (df['Lowess_GC_correctd_cancer_counts']/df['Lowess_GC_correctd_normal_counts']))   
        indices = index_of_chr(df,"chr_ind","bin_start")
        del indices[-1:] 
        for j in range(5):
            for i in indices:
                axs2[j].axvline(x=df.bin_ind[i],ymin=0,ymax=1,c='black')
        axs2[0].set_ylabel('gc_uncorrected cancer',fontsize=7)
        axs2[1].set_ylabel('gc_corrected cancer',fontsize=7)
        axs2[2].set_ylabel('gc_uncorrected normal',fontsize=7)
        axs2[3].set_ylabel('gc_corrected normal',fontsize=7)
        axs2[4].set_ylabel('gc_corrected cancer / normal',fontsize=7)
        plt.xlabel('Chromosoms')
        plt.savefig('plot2.png', dpi=300, bbox_inches='tight')
        plt.close()
        df['GC_corrected_and_control_normalised'] = df["Lowess_GC_correctd_cancer_counts"] / df["Lowess_GC_correctd_normal_counts"]
        df["GC_corrected_and_control_normalised_diploid"] = (df['GC_corrected_and_control_normalised']*2)/df['GC_corrected_and_control_normalised'].median()
        df["GC_corrected_and_control_normalised_median"] = df.groupby('chr_ind').GC_corrected_and_control_normalised_diploid.transform('median')
        df["GC_corrected_and_control_normalised_median"] = df["GC_corrected_and_control_normalised_median"].round(decimals = 0)                           
        fig3, axs3 = plt.subplots()
        axs3.scatter(df['bin_ind'], df['GC_corrected_and_control_normalised_diploid']) 
        fig3.suptitle('Plot 3')
        indices = index_of_chr(df,"chr_ind","bin_start")
        del indices[-1:]
        plt.text(0, .9,'   chr 1',fontsize = 12,verticalalignment ='center') 
        j = 2
        for i in indices:
            axs3.axvline(x=df.bin_ind[i],ymin=0,ymax=1,c='black')
            axs3.text(df.bin_ind[i], .9, '  chr '+str(j),fontsize = 12,verticalalignment ='center') 
            j = j+1
        df[df.chr_ind.isin(list(range(1,20)))].groupby('chr').plot(c ='black',legend= 0,y='GC_corrected_and_control_normalised_median', x='bin_ind', kind='line', ax=axs3,linewidth=5.0)   
        plt.rcParams["figure.figsize"] = [20.50, 5.50]
        plt.xlabel('bins',fontsize = 20)
        plt.ylabel('cancer/control',fontsize = 20)
        
        plt.savefig('plot3.png', dpi=300, bbox_inches='tight')
        plt.close()

    if (args.gc_correction_method) == "median":
        gc_ncorrection(df,df.GC_content,int((df['GC_content'].min()*10000)),int((df['GC_content'].max()*10000)),5,10000,0.05)
        gc_ccorrection(df,df.GC_content,int((df['GC_content'].min()*10000)),int((df['GC_content'].max()*10000)),5,10000,0.05)
        cancer_uncorrected_counts = []
        cg_uncorrected_cancer = []
        s =df.sort_values("GC_content",ignore_index=True)
        for i in range (0,2550,30):
            cancer_uncorrected_counts.append(s.loc[(i-150):(i+150)].cancer_counts.median())
            cg_uncorrected_cancer.append(s.loc[(i-150):(i+150)].GC_content.mean())    
        normal_uncorrected_counts = []
        cg_uncorrected_normal = []
        for i in range (0,2550,30):
            normal_uncorrected_counts.append(s.loc[(i-150):(i+150)].normal_counts.median())
            cg_uncorrected_normal.append(s.loc[(i-150):(i+150)].GC_content.median())
        cancer_corrected_counts = []
        cg_corrected_cancer = []
        s  = df.sort_values("GC_content",ignore_index=True)
        for i in range (0,2550,30):
            cancer_corrected_counts.append(s.loc[(i-150):(i+150)].gc_normalised_cancer_counts.median())
            cg_corrected_cancer.append(s.loc[(i-150):(i+150)].GC_content.median())
        normal_corrected_counts = []
        cg_corrected_normal = []
        for i in range (0,30,100):
            normal_corrected_counts.append(s.loc[(i-150):(i+150)].gc_normalised_normal_counts.median())
            cg_corrected_normal.append(s.loc[(i-150):(i+150)].GC_content.median())
            
        fig1,axs1= plt.subplots(4,sharex=True)
        fig1.suptitle('Plot 1')
        axs1[0].scatter(df['GC_content'],df['cancer_counts'])
        axs1[0].plot(cg_uncorrected_cancer, cancer_uncorrected_counts,color="b",linewidth = 1.0)
        axs1[2].scatter(df['GC_content'],df['normal_counts'])
        axs1[2].plot(cg_uncorrected_normal, normal_uncorrected_counts,color="b",linewidth=1.0)
        axs1[1].scatter(df['GC_content'],df['gc_normalised_cancer_counts'])
        axs1[1].plot(cg_corrected_cancer, cancer_corrected_counts,color = "b",linewidth = 1.0)
        axs1[3].scatter(df['GC_content'],df['gc_normalised_normal_counts'])
        axs1[3].plot(cg_corrected_normal,normal_corrected_counts,color="b",linewidth=1.0)
        axs1[0].set_ylabel('cancer counts',fontsize=7)
        axs1[1].set_ylabel('corrected GC cancer',fontsize=7)
        axs1[2].set_ylabel('normal counts', fontsize=7)
        axs1[3].set_ylabel('corrected GC normal',fontsize=7)
        axs1[3].set_xlabel('GC_content')
        plt.savefig('plot1.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig2, axs2 = plt.subplots(5,sharex=True)
        fig2.suptitle('Plot 2')
        plt.rcParams["figure.figsize"] = [30,10]
        axs2[0].scatter(df['bin_ind'], df['cancer_counts'])
        axs2[1].scatter(df['bin_ind'], df['gc_normalised_cancer_counts']*df['cancer_counts'].median())
        axs2[2].scatter(df['bin_ind'], df['normal_counts'])
        axs2[3].scatter(df['bin_ind'], df['gc_normalised_normal_counts']*df['normal_counts'].median())
        axs2[4].scatter(df['bin_ind'], (df['gc_normalised_cancer_counts']/df['gc_normalised_normal_counts']))   
        indices = index_of_chr(df,"chr_ind","bin_start")
        del indices[-1:] 
        for j in range(5):
            for i in indices:
                axs2[j].axvline(x=df.bin_ind[i],ymin=0,ymax=1,c='black')
        axs2[0].set_ylabel('gc_uncorrected cancer',fontsize=7)
        axs2[1].set_ylabel('gc_corrected cancer',fontsize=7)
        axs2[2].set_ylabel('gc_uncorrected normal',fontsize=7)
        axs2[3].set_ylabel('gc_corrected normal',fontsize=7)
        axs2[4].set_ylabel('gc_corrected cancer / normal',fontsize=7)
        plt.xlabel('Chromosoms')
        plt.savefig('plot2.png', dpi=300, bbox_inches='tight')
        plt.close()

        df['GC_corrected_and_control_normalised'] = df["gc_normalised_cancer_counts"] / df["gc_normalised_normal_counts"]
        df["GC_corrected_and_control_normalised_diploid"] = (df['GC_corrected_and_control_normalised']*2)/df['GC_corrected_and_control_normalised'].median()
        df["GC_corrected_and_control_normalised_median"] = df.groupby('chr_ind').GC_corrected_and_control_normalised_diploid.transform('median')
        df["GC_corrected_and_control_normalised_median"] = df["GC_corrected_and_control_normalised_median"].round(decimals = 0)                           
        fig3, axs3 = plt.subplots()
        axs3.scatter(df['bin_ind'], df['GC_corrected_and_control_normalised_diploid']) 
        fig3.suptitle('Plot 3')
        indices = index_of_chr(df,"chr_ind","bin_start")
        del indices[-1:]
        plt.text(0, .9,'   chr 1',fontsize = 12,verticalalignment ='center') 
        j = 2
        for i in indices:
            axs3.axvline(x=df.bin_ind[i],ymin=0,ymax=1,c='black')
            axs3.text(df.bin_ind[i], .9, '  chr '+str(j),fontsize = 12,verticalalignment ='center') 
            j = j+1
        df[df.chr_ind.isin(list(range(1,20)))].groupby('chr').plot(c ='black',legend= 0,y='GC_corrected_and_control_normalised_median', x='bin_ind', kind='line', ax=axs3,linewidth=5.0)   
        plt.rcParams["figure.figsize"] = [20.50, 5.50]
        plt.xlabel('bins',fontsize = 20)
        plt.ylabel('cancer/control',fontsize = 20)
        plt.tight_layout()
        plt.savefig('plot3.png', dpi=300, bbox_inches='tight')
        plt.close()

    

if __name__== "__main__" :
    main()