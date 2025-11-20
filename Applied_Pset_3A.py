#Coding Sample Python
#Marine Fujisawa

#This project analyzes Medicare billing by home health agencies to identify potential over-billing while distinguishing between fraud and legitimately higher costs due to sicker patients. 
#The goal is to flag suspicious billing without penalizing agencies serving high-need populations.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%
#1. Loading the provider data first
path="/Users/marinefujisawa1/Documents/GitHub/Coding-Sample-Python/data/unformatted_medicare_post_acute_care_hospice_by_provider_and_service_2014_12_31.csv"
provider=pd.read_csv(path, na_values=["NA", "*"])

#I assume the NAs mean the data wasn't reported, or it couldn't be obtained.
#The asterik might be data that is redacted.
# %%
#1. Checking for errors
assert provider.shape==(31665, 122)
#No error, so the shape is correct
# %%
#1.2 Loading the Home Health Resource Group provider data.
pathhhrg="/Users/marinefujisawa1/Documents/Harris/DAP 1 Python/data/Provider_by_HHRG_PUF_2014.xlsx"
provider_hhrg=pd.read_excel(pathhhrg)
# %%
#1.2 Checking the column data types
provider_hhrg.dtypes
# %%
#1.2 Replacing the USD $ symbol so monteary amounts can be numeric
numeric_columns=provider_hhrg.iloc[:, 8:21].columns

for col in numeric_columns:
    provider_hhrg[col]=provider_hhrg[col].str.replace("[\$,]", "", regex=True)

provider_hhrg
# %%
#1.2 Changing the columns with numerical data to be numeric datatype in python, not object
provider_hhrg=provider_hhrg.apply(pd.to_numeric, errors="ignore")
# %%
#1.2 Checking it worked!
provider_hhrg.dtypes
# %%
#1.2 End
assert provider_hhrg.shape==(111904, 20)
# %%
#1.3 Loading the case mix dataset for the health agencies.  
pathweight="/Users/marinefujisawa1/Documents/Harris/DAP 1 Python/data/CY 2014 Final HH PPS Case-Mix Weights.xlsx"
case_mix_weight=pd.read_excel(pathweight)
# %%
#1.3 Dropping unnecessary columns, and renaming to make analysis easier
case_mix_weight=case_mix_weight.drop(columns=["2013 HH PPS Case-Mix Weights"])
case_mix_weight=case_mix_weight.rename(columns={"2014 Final HH PPS Case-Mix Weights": "casemix_2014"})
# %%
case_mix_weight.columns
# %%
#1.3 End
assert case_mix_weight.shape==(153, 4)
# %%
#2.1 Looking at the service categories in provider dataframe
provider['Srvc_Ctgry'].unique()
#From the above, we got: 'HH', 'HOS', 'IRF', 'LTC', 'SNF'.
#HH stands for Home Health, HOS stands for Hospice, IRF stands for Inpatient Rehabilitation Facility, LTC stands for Long Term Care Hospital and SNF stands for Skilled Nursing Facility.

# %%
#2.2 Looking at total number of beneficiaries in provider
print(provider[
    (provider["Srvc_Ctgry"]=="HH") & 
    (provider["Smry_Ctgry"]=="NATION")
]["Bene_Dstnct_Cnt"].sum())

#Got: 3,416,037. 
# %%
#2.3 Comparing total. number of episodes for home health care in provider and provider hhrg
print(provider[(provider["Smry_Ctgry"]=="NATION")&
                (provider["Srvc_Ctgry"]=="HH")]["Tot_Epsd_Stay_Cnt"].sum())
print(provider_hhrg[provider_hhrg["Smry_Ctgry"]=="NATION"]["Tot_Epsd_Stay_Cnt"].sum())
#There is a difference, 6558889 vs 5988839. This might be due to how they are counting each episode and if it shows up multiple times within the dataset. 
# %%
#2.4 Searching for which column(s) identify a unique row in provider_hhrg
#Prvdr_ID and Prvdr_Name
only_provider=provider_hhrg[provider_hhrg["Smry_Ctgry"]=="PROVIDER"]

group_provider=only_provider.groupby(["Grpng", "Prvdr_ID"])
# %%
#2.4
assert len(only_provider)==group_provider.ngroups
# %%
#2.4 End
assert all(group_provider.size()==1)
# %%
#3.1 Want to merge the datasets! 
#The Grpng_Disc column in provider_hhrg and the Description/Clinical, Functional, and Service Levels columns in case_mix_weight contain the 5 potential merge keys.
#This is:
#1. Category of episode state (though worded differently, like "early" in provider_hhrg, "1st or 2nd episode" in case_mix_weight)
#2. Number of therapy visits
#3-5: Clinical, Functional, and Service severity levels, though again coded differently. 
# %%
#3.1
#In provider_hhrg, the "Grpng_Desc" column has the necessary information to merge
provider_hhrg['Grpng_Desc'].nunique()
#From the above, there are 153 unique values
#Each entry holds 5 piecies of info separated by commas
# %%
#3.2 Split the column containing the unique information to get it ready for a merge
split_cols=provider_hhrg["Grpng_Desc"].str.split(",", expand=True)
split_cols
#did this to check what we get. 
#there is an extra column generated which all just has missing values
# %%
#3.2 continue splitting
provider_hhrg['num_commas']=provider_hhrg['Grpng_Desc'].str.count(',')

print(provider_hhrg['num_commas'].unique())
#From the above, we see that some cells contain 5 commas- so when we split the data, it generates another column.
# %%
provider_hhrg[provider_hhrg["num_commas"]==5]
#%%
print(split_cols.iloc[:, 5].unique())
# Since all the values in this 6th column are empty, it's just some extra commas at the end of the row. 
# We also saw this with the code above- filtering for rows with 5 commas.
#Best to remove the comma at the end so this error doesn't happen!
#%%
#3.2 Continued, remove comma at the end of a string in a cell
provider_hhrg['Grpng_Desc']=provider_hhrg['Grpng_Desc'].str.rstrip(',')
#%%
#3.2 End
provider_hhrg[["episode_state", "num_therapy", "clinical", "functional", "service"]]=provider_hhrg["Grpng_Desc"].str.split(",", expand=True) 
provider_hhrg
# %%
#3.3 On to merging case_mix_weight
#The columns "Description" and "Clinical, Functional, and Service Levels" contain our needed information.
print(case_mix_weight.groupby(["Description", "Clinical, Functional, and Service Levels"]).ngroups)
#From the above, we see there are 153 distinct groupings of the two.
# %%
#3.4 split the selected columns into five columns containing the same information as the five columns created in provider_hhrg
case_mix_weight[['episode_state', 'num_therapy']]=case_mix_weight['Description'].str.split(',', expand=True)
case_mix_weight
# %%
#3.4 Continued
case_mix_weight['clinical']=case_mix_weight['Clinical, Functional, and Service Levels'].str.slice(0, 2)
case_mix_weight['functional']=case_mix_weight['Clinical, Functional, and Service Levels'].str.slice(2, 4)
case_mix_weight['service']=case_mix_weight['Clinical, Functional, and Service Levels'].str.slice(4, 6)
case_mix_weight
# %%
#3.5: Adjust values in case_mix_weight so that they match those in provider_hhrg
#starting with a for loop
columns=['episode_state', 'num_therapy', 'clinical', 'functional', 'service']
for i in columns:
    print(f"{i}:")
    print("\nprovider_hhrg unique values:")
    print(sorted(provider_hhrg[i].unique()))
    print("\ncase_mix_weight unique values:")
    print(sorted(case_mix_weight[i].unique()))
# %%
#3.5 continue- matching the values to each other using a dictionary
case_mix_weight['episode_state']=case_mix_weight['episode_state'].replace({
    '1st and 2nd Episodes': 'Early Episode',
    'All Episodes': 'Early or Late Episode',
    '3rd+ Episodes': 'Late Episode'
})
case_mix_weight['episode_state'].unique()
# %%
#3.5 continue- same as above, but for num_therapy
case_mix_weight['num_therapy']=case_mix_weight['num_therapy'].replace({
    ' 0 to 5 Therapy Visits': ' 0-13 therapies',
    ' 10 Therapy Visits': ' 0-13 therapies',
    ' 11 to 13 Therapy Visits': ' 0-13 therapies',
    ' 14 to 15 Therapy Visits': ' 14-19 therapies',
    ' 16 to 17 Therapy Visits': ' 14-19 therapies',
    ' 18 to 19 Therapy Visits': ' 14-19 therapies',
    ' 20+ Therapy Visits ': ' 20+ therapies',
    ' 6 Therapy Visits': ' 0-13 therapies',
    ' 7 to 9 Therapy Visits': ' 0-13 therapies'
})

case_mix_weight['num_therapy'].unique()
# %%
#3.5 continued, finishing up num_therapy
case_mix_weight['num_therapy']=case_mix_weight['num_therapy'].str.strip()
provider_hhrg['num_therapy']=provider_hhrg['num_therapy'].str.strip()
print(case_mix_weight['num_therapy'].unique())
print(provider_hhrg['num_therapy'].unique())
# %%
#3.5 continued, aligning values in the clinical column
case_mix_weight['clinical']=case_mix_weight['clinical'].replace({
    'C1': 'Clinical Severity Level 1',
    'C2': 'Clinical Severity Level 2',
    'C3': 'Clinical Severity Level 3'
})

case_mix_weight['clinical'].unique()
# %%
#3.5 continued, checking functional column
provider_hhrg['functional']=provider_hhrg['functional'].str.strip()
provider_hhrg['functional'].unique()
# %%
#3.5 continued for functional column
case_mix_weight['functional']=case_mix_weight['functional'].replace({
    'F1': 'Functional Severity Level 1',
    'F2': 'Functional Severity Level 2',
    'F3': 'Functional Severity Level 3'
})

case_mix_weight['functional'].unique()
# %%
#3.5 Last
case_mix_weight['service']=case_mix_weight['service'].replace({
    'S1': 'Service Severity Level 1',
    'S2': 'Service Severity Level 2',
    'S3': 'Service Severity Level 3',
    'S4': 'Service Severity Level 4',
    'S5': 'Service Severity Level 5'
})

case_mix_weight['service'].unique()
# %%
#3.5: Creating a new DataFrame named provider_hhrg_wt by merging case_mix_weight with provider_hhrg
provider_hhrg_wt=pd.merge(provider_hhrg,case_mix_weight,
    on=['episode_state', 'num_therapy', 'clinical', 'functional', 'service'],
    how='left', indicator=True, validate='m:1'
)

provider_hhrg_wt.head()
provider_hhrg_wt['_merge'].unique()
# %%
#Checking the merge worked!
assert len(provider_hhrg_wt)==len(provider_hhrg)==111904

assert provider_hhrg_wt['casemix_2014'].isna().sum()==0
# %%
# 4. Billing Outlier Analysis

#4.1 Going to create a new dataframe, provider_sum, that contains information making our billing analysis easier
provider_sum_ctgry=provider_hhrg_wt[provider_hhrg_wt['Smry_Ctgry']=="PROVIDER"]
# %%
#4.1 Continued
provider_sum_grouped=provider_sum_ctgry.groupby(['Prvdr_ID', 'Prvdr_Name', 'State'])

# %%
#4.1 Continued
avg_cost=provider_sum_grouped.apply(
    lambda group: np.average(group["Avg_Chrg_Per_Epsd"], weights=group["Tot_Epsd_Stay_Cnt"])
)

avg_casemix=provider_sum_grouped.apply(
    lambda group: np.average(group["casemix_2014"], weights=group["Tot_Epsd_Stay_Cnt"])
)

total_episodes=provider_sum_grouped["Tot_Epsd_Stay_Cnt"].sum()

provider_sum=pd.concat([avg_cost, avg_casemix, total_episodes], axis=1)
provider_sum
# %%
#4.1 Cleaning up format of provider_sum
provider_sum=provider_sum.reset_index()

provider_sum=provider_sum.rename(columns={
    0: 'avg_cost',
    1: 'avg_casemix',
    'Tot_Epsd_Stay_Cnt': 'total_episodes'
})

provider_sum
# %%
#4.2 Visualizing data
sns.histplot(data=provider_sum, x='avg_cost', color="plum")
plt.title("Variation in Average Cost per Episode by Provider")
plt.xlabel("Average Cost per Episode")
plt.ylabel("Providers")
plt.show()
# %%
#A long right tails shows that there are some outliers of providers that have a much higher average cost per episode than the rest.
#However, this doesn't automatically mean they are defrauding the government. It is plausible that some providers/facilities are specialized in providing some high cost or exclusive care that other providers don't have.
#Like the introduction stated, these providers may just have unusually sick patients.
# %%
#4.3 Visualizing Data
sns.regplot(data=provider_sum, x="avg_casemix", y="avg_cost", line_kws={"color": "plum"})
plt.title("Relationship Betweeen Average Cost and Average Case-Mix")
plt.xlabel("Average Case Mix")
plt.ylabel("Average Cost per Episode")
plt.show()
# %%
#4.3 Continued
#From the graph, we see a positive correlation between the two variables- higher average case mix correlates to higher average cost per episode.
#According to Google, case mix weight is a representation of the complexity/severity of a given patient.
#Thus, it would make sense that a higher case mix weight would lead to a higher average cost per episode, since the patients are more in need/complex to care for.
#It is relevant to detct fraud because any outliers (high average cost but low average case mix) may be participating in fraud. 
# %%
#4.4 Creating column cost_normalized, equal to the ratio of average cost to the average case-mix weight for each provider
provider_sum['cost_normalized']=provider_sum["avg_cost"] / provider_sum["avg_casemix"]

# %%
#4.5 Visualizing data with normalized cost
sns.histplot(data=provider_sum, x='cost_normalized', color="indianred", label="Normalized Cost")
sns.histplot(data=provider_sum, x="avg_cost", color="lavender", label="Avg Cost")

plt.title("Variation in Nomalized and Average Cost per Episode by Provider")
plt.xlabel("Cost per Episode")
plt.ylabel("Providers")
plt.legend()
plt.show()
# %%
#4.5
#From the plot, we see that there isn't much of a difference between the average and normalized cost.
#The normalized cost is slightly more to the right, but it's not a very discernable difference.
#Cost normalized should have accounted for the case mix weight (so sicker patients), as opposed to the average cost.
#The fact that there is still a wide right tail, and with normalized cost even appearing to be shifted to the right slightly, it might mean there are some cases of overbilling.
# %%
#4.6 Finding top five home health care providers with the highest average billing per episode in Ilinois
illinois_providers=provider_sum[provider_sum['State']=="IL"]
print(illinois_providers.sort_values(by='avg_cost', ascending=False).head(5))
# %%
#4.6 Finding top 5 providers with the highest average costs adjusted for case-mix weight in Illinois
print(illinois_providers.sort_values(by='cost_normalized', ascending=False).head(5))

# %%
