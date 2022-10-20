import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def print_hello_world():
    print('Hello world!')


def fix_abnormal_geo_location(df):
    lat_mask = df['lat'] > 1.5
    lng_mask = (df['lng'] > 105) | (df['lng'] < 103)
    mask = lat_mask & lng_mask
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
    geo_coordinates = np.vstack([lng_array, lat_array])
    print('Calculating kernel density...')
    density = gaussian_kde(geo_coordinates)(geo_coordinates)

    plt.scatter(lng_array, lat_array, c=density)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(filename, dpi=500)


# need to fix lat and lng first
def map_subzone_by_geo_location_knn(df):
    subzone_mask_na = df['subzone'].isna()
    df_na_subzone = df[subzone_mask_na]
    df_full_subzone = df[~subzone_mask_na]
    property_location = df_full_subzone[['lat', 'lng']].to_numpy()
    subzone_label = df_full_subzone['subzone'].tolist()
    subzone_knn_classifier = KNeighborsClassifier(n_neighbors=9)
    subzone_knn_classifier.fit(property_location, subzone_label)
    property_location_pred = df_na_subzone[['lat', 'lng']].to_numpy()
    pred_subzones = subzone_knn_classifier.predict(property_location_pred)
    # print(pred_subzones)
    df.loc[subzone_mask_na, 'subzone'] = pred_subzones


def universalize_tenure(df):
    tenure_mapping_dict = {
        '99-year leasehold': ['99-year leasehold', '103-year leasehold', '110-year leasehold', '102-year leasehold',
                              '100-year leasehold'],
        '999-year leasehold': ['999-year leasehold', '956-year leasehold', '946-year leasehold', '929-year leasehold',
                               '947-year leasehold'],
        'freehold': ['freehold']
    }

    for tenure_key, tenure_values in tenure_mapping_dict.items():
        for tenure_value in tenure_values:
            df.loc[df['tenure'] == tenure_value, ('tenure')] = tenure_key


def fillna_by_property_name(df, attr):
    na_mask = df[attr].isna()
    ds_mapping = df[~na_mask].groupby('property_name')[attr].first()
    lambda_mapping = lambda x: ds_mapping[x['property_name']] if (pd.isna(x[attr]) and x['property_name'] in ds_mapping) else x[attr]
    df[attr] = df.apply(lambda_mapping, axis=1)
    # print(attr, 'still na', df[attr].isna().sum())


def fillna_by_grouping(df, na_attr, grouping_attr):
    # get count of each na_attr grouped by grouping_attr
    na_mask = df[na_attr].isna()
    ds_size_grouping = df[~na_mask].groupby([grouping_attr, na_attr]).size()
    size_grouping_dict = ds_size_grouping.to_dict()

    # transform dict
    grouping_key_list, grouping_size_list = zip(*size_grouping_dict.items())
    grouping_dict = dict()
    for i, (grouping, attr) in enumerate(grouping_key_list):
        size = grouping_size_list[i]
        grouping_info = grouping_dict.setdefault(grouping, {'value': [], 'count': []})
        grouping_info['value'].append(attr)
        grouping_info['count'].append(size)
    # print(grouping_dict)

    # utility
    def randomize_value_wrt_count(grouping_dict, grouping):
        count_array = np.array(grouping_dict[grouping]['count'])
        prob_array = count_array / count_array.sum()
        return np.random.choice(grouping_dict[grouping]['value'], p=prob_array)

    # utility
    def fill_value_by_grouping(row, na_attr, grouping_attr, grouping_dict):
        if row[grouping_attr] in grouping_dict:
            return randomize_value_wrt_count(grouping_dict, row[grouping_attr])
        else:
            return row[na_attr]

    # fill na value
    df.loc[na_mask, na_attr] = df[na_mask].apply(lambda x: fill_value_by_grouping(x, na_attr, grouping_attr, grouping_dict), axis=1)
    # print('still na', df[na_attr].isna().sum())


# require size_sqft to be fixed first
def fill_na_num_beds(df):
    na_mask = df['num_beds'].isna()
    sqft_mask = df['size_sqft'] < 1000.
    # all studios, 0 would be a meaningful value
    df.loc[na_mask & sqft_mask, 'num_beds'] = 0.

    na_mask = df['num_beds'].isna()
    # super big studio, often configured into multiple bedrooms, just follow num_baths here
    df.loc[na_mask, 'num_beds'] = df.loc[na_mask, 'num_baths']

    # manually lookup, for num_baths also
    df.loc[3713, 'num_beds'] = 5.
    df.loc[3713, 'num_baths'] = 5.
    df.loc[5002, 'num_beds'] = 3.
    df.loc[5002, 'num_baths'] = 2.


def discretize_built_year(df, bins, labels):
    ds_interval = pd.cut(df['built_year'], bins=bins)
    ds_labels = pd.cut(ds_interval, bins).map(labels)
    df['built_year'] = ds_labels


def fill_conservation_house_built_year(df, value):
    df.loc[df['property_type'] == 'conservation house', 'built_year'] = value


def fix_abnormal_beds_baths_number(df):
    mask = (df['num_beds'] == 1.0) & (df['num_baths'] == 10.0)
    df.loc[mask, 'num_baths'] = 1.0


def fix_super_high_price(df):
    # tune for each super high price
    df.loc[635, 'price'] = df.loc[635, 'price'] / 100  # land only, previously 1e8 plus
    df.loc[5976, 'price'] = df.loc[5976, 'price'] / 1e5  # hdb, previously 3.9e10
    df.loc[16264, 'price'] = df.loc[16264, 'price'] / 1e4  # hdb, previously 4.9e9

    # super expensive hdb
    df.loc[3066, 'price'] = df.loc[3066, 'price'] / 5  # previously 2,100,000
    df.loc[3430, 'price'] = df.loc[3430, 'price'] / 10  # previously 8,400,000
    df.loc[9744, 'price'] = df.loc[9744, 'price'] / 5  # previously 1,155,000

    # condo / ec
    df.loc[663, 'price'] = df.loc[663, 'price'] / 10  # previously 30,880,500
    df.loc[19587, 'price'] = df.loc[19587, 'price'] / 10  # previously 16,884,000


def remove_price_zero_records(df):
    df.drop(df[df['price'] == 0.].index, inplace=True)


def fill_zero_sqft(df):
    mask = (df['size_sqft'] == 0.) & (df['property_name'] == 'clavon') & (df['num_beds'] == 5.)
    df.loc[mask, 'size_sqft'] = 1690.


def fix_abnormal_sqft(df):
    # obtain the insights from EDA
    sqft_mask_1 = (df['size_sqft'] > 35000) & (df['size_sqft'] < 1e5)
    sqft_mask_2 = (df['size_sqft'] > 1e6) & (df['size_sqft'] < 1e7)
    df.loc[sqft_mask_1, 'size_sqft'] = df.loc[sqft_mask_1, 'size_sqft'] / 10
    df.loc[sqft_mask_2, 'size_sqft'] = df.loc[sqft_mask_2, 'size_sqft'] / 1000

    # super large hdb, ranging from 6,000 to 15,000
    sqft_mask_3 = (df['size_sqft'] > 6000) & (df['property_type'] == 'hdb')
    df.loc[sqft_mask_3, 'size_sqft'] = df.loc[sqft_mask_3, 'size_sqft'] / 10

    # super large condo
    df.loc[16370, 'size_sqft'] = df.loc[16370, 'size_sqft'] / 10  # previously 13,000

    # abnormal landed
    df.loc[13461, 'size_sqft'] = df.loc[13461, 'size_sqft'] / 10  # previously 25,000

    # specific case
    df.loc[5976, 'size_sqft'] = 1313  # find from other same property records

def convert_sqm_to_sqft(df):
    sqm_to_sqft_multiplier = 10.764
    sqft_mask_sqm = (df['size_sqft'] < 200) & (df['size_sqft'] > 0)
    df.loc[sqft_mask_sqm, 'size_sqft'] = df.loc[sqft_mask_sqm, 'size_sqft'] * sqm_to_sqft_multiplier

    # cluster house
    df.loc[14218, 'size_sqft'] = df.loc[14218, 'size_sqft'] * sqm_to_sqft_multiplier
    df.loc[15027, 'size_sqft'] = df.loc[15027, 'size_sqft'] * sqm_to_sqft_multiplier


def target_encode_property_type_subzone(df):
    df_price_sqft = df.groupby(['subzone', 'property_type'])[['price', 'size_sqft']].sum().reset_index()
    df_price_sqft['sqft_price'] = df_price_sqft['price'] / df_price_sqft['size_sqft']
    df_price_sqft.drop(columns=['price', 'size_sqft'], inplace=True)
    encoding_dict = df_price_sqft.set_index(['subzone', 'property_type']).to_dict()['sqft_price']
    df['subzone_property_type_encoding'] = df.apply(lambda x: encoding_dict[(x['subzone'], x['property_type'])], axis=1)
    df.drop(columns=['subzone', 'property_type'], inplace=True)

    cols = list(df.columns)
    a, b = cols.index('price'), cols.index('subzone_property_type_encoding')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]
    return df, encoding_dict


def encode_built_year(df, map_dict):
    df['built_year'] = df['built_year'].map(map_dict)


def encode_tenure(df):
    tenure_dict = {
        '99-year leasehold': 0.,
        '999-year leasehold': 1.,
        'freehold': 2.
    }
    df['tenure'] = df['tenure'].map(tenure_dict)


def encode_planning_area(df, pa_list):
    df_pa_list = df['planning_area'].to_list()
    pa_dict = {k: v for v, k in enumerate(pa_list)}
    ont_hot_length = len(pa_list)
    one_hot_array = np.zeros((len(df_pa_list), ont_hot_length))
    for i, pa in enumerate(df_pa_list):
        one_hot_index = pa_dict[pa]
        one_hot_array[i][one_hot_index] = 1.
    pa_columns = ['pa_' + pa.replace(' ', '_') for pa in pa_list]
    df_pa = pd.DataFrame(one_hot_array, columns=pa_columns)
    df.reset_index(drop=True, inplace=True)
    df_pa.reset_index(drop=True, inplace=True)
    df = df.join(df_pa)
    df.drop(columns=['planning_area'], inplace=True)
    return df


# num beds num bath
# better to run after section of tenure and year
def map_value_by_most_common(df, attr, group):
    na_mask = df[attr].isna()
    size_attr_by_type = df.groupby([group, attr]).size()
    attr_by_type_mapping = size_attr_by_type.reset_index(level=0).groupby(group).agg(most_common = (0, 'idxmax'))
    attr_mapping_dict = attr_by_type_mapping.reset_index().set_index(group).to_dict()['most_common']
    # print(attr_mapping_dict)
    df[attr] = df.apply(lambda x: attr_mapping_dict[x[group]] if (pd.isna(x[attr]) and x[group] in attr_mapping_dict) else x[attr], axis=1)
    # print('Final na', attr, df[attr].isna().sum())
    # print(df[attr].value_counts())

def process_num_beds_and_baths(df, method = 1):
    # num_beds
    upper = np.percentile(df.loc[df.num_beds.isna(), 'size_sqft'], 75)

    df.loc[(df.num_beds.isna() & (df.size_sqft <= upper)), 'num_beds'] = 0

    pn = df['property_name'][df.num_beds.isna()].unique()
    check1 = df.loc[np.in1d(df['property_name'], pn)]
    df.loc[np.in1d(df['property_name'], check1['property_name'].drop_duplicates(keep=False).to_list()), 'num_beds'] = 0
    df.sort_values(by=['property_name', 'size_sqft', 'num_beds'], ascending=False, inplace=True)
    df['num_beds'].fillna(method='ffill', inplace=True)

    correct = np.round(df.loc[(df['property_type'] == 'hdb'), 'num_beds'].mean())
    df.loc[(df['property_type'] == 'hdb') & (df['num_beds'] == 0), 'num_beds'] = correct

    # num_baths
    if method == 1:
        df.sort_values(by=['property_name', 'size_sqft', 'num_baths'], ascending=False, inplace=True)
        df['num_baths'].fillna(method='ffill', inplace=True)
    else:
        map_value_by_most_common(df, 'num_baths', 'num_beds')

# property_type
def process_property_type(df):
    # Handle 'property_type' column (for test)
    df['property_type'] = df['property_type'].str.lower()
    df['property_type'] = df['property_type'].apply(lambda x: 'hdb' if 'hdb' in x else x)

# encoding with average price or price per sqft
def encode_with_avg_price(df, attr):
    ser = df.groupby(attr)['price'].mean()
    df.loc[:, attr].apply(lambda x: ser[x])

def encode_with_avg_price_sqft(df, attr):
    df['pps'] = df['price'] / df['size_sqft']
    ser = df.groupby(attr)['pps'].mean()
    df.loc[:, attr].apply(lambda x: ser[x])
    df.drop('pps', axis=1, inplace=True)

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

def handle_years_and_tenure_nan(df_no_nan, attr):
    #ffill (choose one)
    df_no_nan[attr].fillna(method='ffill', inplace=True)
    print(df_no_nan[attr].value_counts())
    print("na number of", attr, "column is", df_no_nan[attr].isna().sum())

#     df_no_nan[attr].fillna(method='bfill', inplace=True)
#     print(df_no_nan[attr].value_counts())
#     print("na number of", attr, " column is", df_no_nan[attr].isna().sum())

# 99.co/singapore/insider/big-leasehold-debate
def normalize_tenure_to_three_cat(a):
    if abs(a - 99) < abs(a - 999):
        return 99
    return 999




# size price
def df_process_size_sqft(df):
    uni_property_name = df['property_name'].unique()
    
    # preprocess obvious outliers
    df["size_sqft"] = np.where(df["size_sqft"] > 1e6, df["size_sqft"]/1000, df["size_sqft"])
    df["size_sqft"] = np.where(df["size_sqft"] > 60000, df["size_sqft"]/10, df["size_sqft"])
    df["size_sqft"] = np.where(df["size_sqft"] == 0, 1690, df["size_sqft"])

    # find outliers by rules and process these outliers separately
    min_extre_list = []
    max_extre_list = []

    # Jiechen add
    uni_property_name = df['property_name'].unique()

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
            # based on assumption that size_sqft properties with same property_name and same property_type cannot range outside [mid/5, mid*5]
            # and assign value with median size_sqft to this data point
            property_mean = df[(df['property_name'] == uni_name) & (df['property_type'] == p_type)]['size_sqft'].mean()
            if min_dt < mid_dt / 5:
                df['size_sqft'] = np.where((df['size_sqft'] < mid_dt/5) & (df['property_name'] == uni_name) & (df['property_type'] == p_type), property_mean, df['size_sqft'])
            if max_dt > mid_dt * 5:
                df['size_sqft'] = np.where((df['size_sqft'] > mid_dt*5) & (df['property_name'] == uni_name) & (df['property_type'] == p_type), property_mean, df['size_sqft'])

    for lst_id in min_extre_list:
        df["size_sqft"] = np.where(df["listing_id"] == lst_id, df["size_sqft"] * 10, df["size_sqft"])

    for lst_id in max_extre_list:
        df["size_sqft"] = np.where(df["listing_id"] == lst_id, df["size_sqft"] / 10, df["size_sqft"])


def df_process_price(df):
    uni_property_name = df['property_name'].unique()

    df.drop(index = df[df['price']==0].index, inplace=True)

    df['price_sqft'] = df['price'] / df['size_sqft']
    min_price_extre_list = []
    max_price_extre_list = []

    # Jiechen add
    uni_property_name = df['property_name'].unique()

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

    df.drop(columns=['price_sqft'], inplace=True)










