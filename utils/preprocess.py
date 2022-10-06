import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def print_hello_world():
    print('Hello world!')


def fix_abnormal_geo_location(df, mask):
    property_lat_lng_dict = {
        '1953': (1.3164866999403357, 103.8574701820852),
        'pollen & bleu': (1.3135917327724542, 103.80674437513318),
        'ness': (1.3127625951710529, 103.8868400395438),
        'm5': (1.2956970241386963, 103.82891099257093)
    }
    for name, (lat, lng) in property_lat_lng_dict.items():
        lat_mask = mask & (df['property_name'] == name)
        lng_mask = mask & (df['property_name'] == name)
        df.loc[lat_mask, 'lat'] = lat
        df.loc[lng_mask, 'lng'] = lng


def save_geo_scatter_plot(lat_array, lng_array, filename):
    # Calculate the point density
    print('Stacking data...')
    geo_coordinates = np.vstack([lat_array, lng_array])
    print('Calculating kernel density...')
    density = gaussian_kde(geo_coordinates)(geo_coordinates)

    plt.scatter(lat_array, lng_array, c=density)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.savefig(filename, dpi=500)


# need to fix lat and lng first
def map_subzone_by_geo_location_knn(df):
    subzone_mask_na = df['subzone'].isna()
    df_na_subzone = df[subzone_mask_na]
    df_full_subzone = df[~subzone_mask_na]
    property_location = df_full_subzone[['lat', 'lng']].to_numpy()
    subzone_label = df_full_subzone['subzone'].tolist()
    subzone_knn_classifier = KNeighborsClassifier(n_neighbors=10)  # tried for 3,5,10, almost same
    subzone_knn_classifier.fit(property_location, subzone_label)
    property_location_pred = df_na_subzone[['lat', 'lng']].to_numpy()
    pred_subzones = subzone_knn_classifier.predict(property_location_pred)
    print(pred_subzones)
    df.loc[subzone_mask_na, 'subzone'] = pred_subzones

# num beds num bath
# better to run after section of tenure and year
def map_value_by_most_common(df, attr, group):
    na_mask = df[attr].isna()
    size_attr_by_type = df.groupby([group, attr]).size()
    attr_by_type_mapping = size_attr_by_type.reset_index(level=0).groupby(group).agg(most_common = (0, 'idxmax'))
    attr_mapping_dict = attr_by_type_mapping.reset_index().set_index(group).to_dict()['most_common']
    print(attr_mapping_dict)
    df[attr] = df.apply(lambda x: attr_mapping_dict[x[group]] if (pd.isna(x[attr]) and x[group] in attr_mapping_dict) else x[attr], axis=1)
    # print('Final na', attr, df[attr].isna().sum())
    # print(df[attr].value_counts())

def process_num_beds_baths(df, method = 1):
    # num_beds
    upper = np.percentile(df.loc[df.num_beds.isna(), 'size_sqft'], 75)
    df.loc[(df.num_beds.isna()) & (df.size_sqft <= upper), 'num_beds'] = 0

    pn = df['property_name'][df.num_beds.isna()].unique()
    check1 = df.loc[np.in1d(df['property_name'], pn)]
    df.loc[np.in1d(df['property_name'], check1['property_name'].drop_duplicates(keep=False).to_list()), 'num_beds'] = 0
    df.sort_values(by=['property_name', 'size_sqft', 'num_beds'], ascending=False, inplace=True)
    df['num_beds'].fillna(method='ffill', inplace=True)

    # num_baths
    if method == 1:
        df.sort_values(by=['property_name', 'size_sqft', 'num_baths'], ascending=False, inplace=True)
        df['num_baths'].fillna(method='ffill', inplace=True)
    else:
        map_value_by_most_common(df, 'num_beds', 'num_baths')


# tenure & year
# pre-condition: property
def normalize_tenure(df):
    df['tenure'] = df['tenure'].astype(str)
    tenure_type = []

    for i in range(len(df['tenure'])):
        if df['tenure'][i] != 'nan':
            if 'leasehold' in df['tenure'][i]:
                temp = int(df['tenure'][i].split('-')[0])
                tenure_type.append(temp)
#                 tenure_type.append(normalize_tenure(temp))
            else:
                tenure_type.append('freehold')

        else:
            if df['property_type'][i] == 'hdb':
                tenure_type.append(99)

            else:
                tenure_type.append(np.nan)
    df['tenure'] = pd.Series(tenure_type)


# def add_leaselast_time(df):   
#     # Handle 'tenure' and 'built_year' column
#     df['tenure'] = df['tenure'].astype(str)
#     df['built_year'] = df['built_year'].astype(str)

#     leasehold_time = []
#     tenure_type = []

#     for i in range(len(df['tenure'])):

#         if df['tenure'][i] != 'nan':
#             if 'leasehold' in df['tenure'][i]:
#                 temp = int(df['tenure'][i].split('-')[0])
                
#                 if len(df['built_year'][i]) != 0:
#                     hold_time = temp + float(df['built_year'][i]) - 2022
#                     leasehold_time.append(hold_time)
                    
#                 else:
#                     leasehold_time.append('nan')
                
#                 tenure_type.append(temp)
# #                 tenure_type.append(normalize_tenure(temp))


#             else:
#                 leasehold_time.append('infinite')
#                 tenure_type.append('freehold')

#         else:
#             if df['property_type'][i] == 'hdb':
#                 if len(df['built_year'][i]) != 0:
#                     hold_time = 99 + float(df['built_year'][i]) - 2022
#                     leasehold_time.append(hold_time)
                    
#                     tenure_type.append(99)

#             else:
#                 leasehold_time.append('nan')
#                 tenure_type.append('nan')

#     df['tenure_type'] = pd.Series(tenure_type)
#     df['leaselast_time'] = pd.Series(leasehold_time)


# 99.co/singapore/insider/big-leasehold-debate
def normalize_tenure_to_three_cat(a):
    if abs(a - 99) < abs(a - 999):
        return 99
    return 999




# size price
def df_process_size_sqft(df):
    # preprocess obvious outliers
    df["size_sqft"] = np.where(df["size_sqft"] > 1e6, df["size_sqft"]/1000, df["size_sqft"])
    df["size_sqft"] = np.where(df["size_sqft"] > 60000, df["size_sqft"]/10, df["size_sqft"])
    df["size_sqft"] = np.where(df["size_sqft"] == 0, 1690, df["size_sqft"])

    # find outliers by rules and process these outliers separately
    min_extre_list = []
    max_extre_list = []

    for uni_name in uni_property_name:
        data_desc = df[df['property_name']==uni_name]['size_sqft'].describe()
        mid_dt = data_desc['50%']    
        min_dt = data_desc['min']
        max_dt = data_desc['max']

        # based on assumption that size_sqft properties with same property_name cannot range outside [mid/10, mid*10]
        # and data input may confuse size_sqft and size in square meters
        if min_dt < mid_dt / 10:
            min_extre_list += df[(df['size_sqft'] < mid_dt/10) & (df['property_name'] == uni_name)]['listing_id'].values.tolist()

        if max_dt > mid_dt * 10:
            max_extre_list += df[(df['size_sqft'] > mid_dt*10) & (df['property_name'] == uni_name)]['listing_id'].values.tolist()


        prt_type = df[df['property_name']==uni_name]['property_type'].unique()
        for p_type in prt_type:
            prt_desc = df[(df['property_name']==uni_name) & (df['property_type']==p_type)]['size_sqft'].describe()
            mid_dt = data_desc['50%']    
            min_dt = data_desc['min']
            max_dt = data_desc['max']
            # based on assumption that size_sqft properties with same property_name and same property_type cannot range outside [mid/3, mid*3]
            # and assign value with similar size_sqft to this data point
            if min_dt < mid_dt / 5:
                df[(df['size_sqft'] < mid_dt/5) & (df['property_name'] == uni_name) & (df['property_type'] == p_type)] = mid_dt

            if max_dt > mid_dt * 5:
                df[(df['size_sqft'] > mid_dt*5) & (df['property_name'] == uni_name) & (df['property_type'] == p_type)] = mid_dt

    for lst_id in min_extre_list:
        df["size_sqft"] = np.where(df["listing_id"] == lst_id, df["size_sqft"] * 10, df["size_sqft"])

    for lst_id in max_extre_list:
        df["size_sqft"] = np.where(df["listing_id"] == lst_id, df["size_sqft"] / 10, df["size_sqft"])


def df_process_price(df):
    df = df.drop(df[df['price']==0].index)

    df['price_sqft'] = df['price'] / df['size_sqft']
    min_price_extre_list = []
    max_price_extre_list = []

    for uni_name in uni_property_name:
        price_describe = df[df['property_name']==uni_name]['price_sqft'].describe()
        mid_dt = price_describe['50%']
        min_dt = price_describe['min']
        max_dt = price_describe['max']
        if min_dt < mid_dt / 10:
            min_price_extre_list += df[(df['price_sqft'] < mid_dt/10) & (df['property_name'] == uni_name)]['listing_id'].values.tolist()

        if max_dt > mid_dt * 10:
            max_price_extre_list += df[(df['price_sqft'] > mid_dt*10) & (df['property_name'] == uni_name)]['listing_id'].values.tolist()


    for i in max_price_extre_list:
        idx = df[df['listing_id']==i].index
        property_name = df[df['listing_id']==i]['property_name'].values[0]
        price = df[df['listing_id']==i]['price'].values[0]
        price_describe = df[df['property_name']==property_name]['price'].describe()
        quarter3_dt = price_describe['75%']

        while price > quarter3_dt:
            df.loc[idx, ['price']] = price/10
            price /= 10












