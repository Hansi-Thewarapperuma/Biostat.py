'''
Biological hypothesis testing
Input : csv filles and via dataframe
Output : relevant plots, normality tests and other statistic tests results
Author: Hansi Thewarapperuma
Date: 31/01/2023
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

temperature = pd.read_csv('Temperature.csv')
# print(temperature)
print(temperature.describe())

# Testing the normality assumption for the temperature variable
# plt.hist(temperature)
# plt.show()
# plt.savefig('temperature histogram')

# sns.histplot(temperature, kde= True)
# plt.show()
# plt.savefig('temperature histogram using seaborn.jpg')
# plt.title('Histogram of temperature')

# from statsmodels.graphics.gofplots import qqplot
# qqplot(temperature, line= 's')
# plt.show()
# plt.savefig('QQplot for temperature.jpg')
# plt.title('QQplot for temperature')

# shapiro test
# from scipy import stats
stat,p = stats.shapiro(temperature)
print('shapiro test results: ','stat=%.3f p-value=%.3f'%(stat,p))

# one sample t test
known_temperature = 98.6
t,p = stats.ttest_1samp(temperature, known_temperature)
print('Q1 - one sample t test results: ')
print('t stat: ', t)
print('p value: ',p)

# Question 2

horn_length = pd.read_csv('HornedLizards.csv')
# print(horn_length)

# get rid of NA values in csv file
new_data = horn_length.dropna(axis = 0, how ='any')
# print(len(horn_length))
# print(len(new_data))

print(horn_length.describe())

# ******filter pandas data values by column value using df.loc*****

# extract horn lengths of survived
# step1- extract both values and annotation columns from mixed csv and assign it to 'survived'
survived = new_data.loc[new_data['Survive']=='survived']
# print(survived)
# step2 - extract the values from 'survived'
survived_val = survived.loc[:,"Squamosal horn length"]
# print(survived_val)
# print(len(survived_val))
# check = survived['Squamosal horn length']
# print(len(check))

# extract horn lengths of dead
# step1- extract both values and annotation columns from mixed csv and assign it to 'dead'
dead = new_data.loc[new_data['Survive']=='dead']
# step2 - extract the values from 'dead'
dead_val = dead.loc[:,"Squamosal horn length"]
# print(dead_values)

# histogram for the survived sample
# fig, axs = plt.subplots(1,2,figsize=(12,4))
# # ******mistake- replace survived by x
# sns.histplot(survived_val, kde= True, ax=axs[0])
# axs[0].set_title('Histogram of survived lizards')

# histogram for the dead sample
# sns.histplot(dead_val, kde=True, ax=axs[1])
# axs[1].set_title('Histogram of dead lizards')
# plt.show()
# plt.savefig('Histogram of 2 samples.jpg')

# QQplot for survived sample
# from statsmodels.graphics.gofplots import qqplot
# qqplot(survived_val, line= 's')
# plt.show()
# plt.title('QQplot for survived sample')
# plt.savefig('QQplot for survived sample.jpg')

# QQplot for dead sample
# qqplot(dead_val, line= 's')
# plt.show()
# plt.title('QQplot for dead sample')
# plt.savefig('QQplot for dead sample.jpg')

# shapiro tests for both groups
# from scipy import stats
stat,p = stats.shapiro(survived_val)
print('shapiro test for survived lizards: ','stat=%.3f p-value=%.3f'%(stat,p))

stat,p = stats.shapiro(dead_val)
print('shapiro test for dead lizards: ','stat=%.3f p-value=%.3f'%(stat,p))

# comparison of means using boxplot and violin plot
# boxplot for survived sample
# fig, axs = plt.subplots(1,2,figsize=(12,4))
# sns.boxplot(survived, ax=axs[0])
# axs[0].set_title('Boxplot of survived lizards')

# boxplot for the dead sample
# sns.boxplot(dead, ax=axs[1])
# axs[1].set_title('Boxplot of dead lizards')
# plt.show()
# plt.savefig('Histogram of 2 samples.jpg')

# violinplot for survived sample
# fig, axs = plt.subplots(1,2,figsize=(12,4))
# sns.violinplot(survived, ax=axs[0])
# axs[0].set_title('Violinplot of survived lizards')

# violinplot for the dead sample
# sns.violinplot(dead, ax=axs[1])
# axs[1].set_title('Violinplot of dead lizards')
# plt.show()
# plt.savefig('Violinplots of 2 samples.jpg')

# paired sample t test
# t,p = stats.ttest_ind(survived_val, dead_val, equal_var=True, alternative= 'greater')
# print('t stat: ',t)
# print('p value: ', p)

# check for equal variances
# print(np.var(survived_val),np.var(dead_val))

# Perform the two sample t-test with equal variances
# t,p = stats.ttest_ind(survived_val, dead_val, equal_var=True, alternative= 'greater')
# print('t stat: ',t)
# print('p value: ', p)

# perform MannWhitney test
t,p = stats.mannwhitneyu(survived_val, dead_val, alternative= 'greater')
print('Q2 - Mannwhitney test')
print('U statistics: ',t)
print('p value: ', p)

# question 3

# create dataframe
antibody_production = pd.read_csv('BlackbirdTestosterone.csv')

# select a subset of a dataframe
log_before = antibody_production['log before']
log_after = antibody_production['log after']
log_difference = antibody_production['dif in logs']
before_sample = antibody_production['Before']
after_sample = antibody_production['After']

# log_before.head()

print(log_before.describe())
print(log_after.describe())
print(log_difference.describe())

# plotting histogram for log difference using seaborn
# sns.histplot(log_difference, kde= True)
# plt.show()
# plt.savefig('histogram for log difference using seaborn.jpg')
# plt.title('Histogram of log difference')

# plotting qqplot for the log difference using statmodels
# qqplot(log_difference, line= 's')
# plt.show()
# plt.title('QQplot for log difference ')
# plt.savefig('QQplot for log difference.jpg')

stat,p = stats.shapiro(log_difference)
print('shapiro test for log difference','stat=%.3f p-value=%.3f'%(stat,p))

# comparison of means using boxplot and violin plot
# boxplot for before sample
# fig, axs = plt.subplots(1,2,figsize=(12,4))
# sns.boxplot(before_sample, ax=axs[0])
# axs[0].set_title('Boxplot of before sample')

# boxplot for the after sample
# sns.boxplot(after_sample, ax=axs[1])
# axs[1].set_title('Boxplot of after sample')
# plt.show()
# plt.savefig('Histogram of 2 before and after samples.jpg')

# violinplot for before sample
# fig, axs = plt.subplots(1,2,figsize=(12,4))
# sns.violinplot(before_sample, ax=axs[0])
# axs[0].set_title('Violinplot of before sample')

# violinplot for the after sample
# sns.violinplot(after_sample, ax=axs[1])
# axs[1].set_title('Violinplot of after sample')
# plt.show()
# plt.savefig('Violinplots of 2 samples.jpg')

# paired sample t test
# ********order of log_after anf log_before
t,p = stats.ttest_rel(log_before, log_after, alternative='less')
print('Q3 - Paired sample t test results')
print('t stat: ',t)
print('p value: ', p)

# t,p = stats.ttest_rel(after_sample, before_sample, alternative='less')
# print('t stat: ',t)
# print('p value: ', p)

# question 3
# create manual dataframe
df = pd.DataFrame([[1,10,37],[49,35,9]],index=['eaten','not_eaten'],columns=['uninfected','lightly_infected','highly_infected'])
print(df)
# myCrosstable = pd.crosstab(df['uninfected'], df['lightly_infected'], df['highly_infected'])

# df = pd.DataFrame([['Eaten by birds',1,10,37],['not eaten by birds',49,35,9]],columns=['','uninfected','lightly_infected','highly_infected'])
df2 = df.stack()
print(df2)

df3 = df2.to_dict()
print(df3)

# plotting the mosaic plot
# from statsmodels.graphics.mosaicplot import mosaic
# mosaic(df3, gap= 0.05)
# plt.show()
# plt.title('mosaic plot')
# plt.savefig('mosaic plot.jpg')

chi = stats.chi2_contingency(df)
print('chi square statistic: ',chi)

# output the expected values
# expected_values = stats.contingency.expected_freq(df)
# print(expected_values)
# create a dataframe using the above expected values
expected_values_table = pd.DataFrame(chi.expected_freq, index=['eaten','not_eaten'], columns=['uninfected','lightly_infected','highly_infected'])
print(expected_values_table)



















