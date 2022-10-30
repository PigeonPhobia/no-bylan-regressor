import numpy as np
import pandas as pd
from geopy import distance
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

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


def auto_fill_na_num_beds(df):
    na_mask = df['num_beds'].isna()
    sqft_mask = df['size_sqft'] < 1000.
    # all studios, 0 would be a meaningful value
    df.loc[na_mask & sqft_mask, 'num_beds'] = 0.


# require size_sqft to be fixed first
def fill_na_num_beds(df):
    auto_fill_na_num_beds(df)

    na_mask = df['num_beds'].isna()
    # super big studio, often configured into multiple bedrooms, just follow num_baths here
    df.loc[na_mask, 'num_beds'] = df.loc[na_mask, 'num_baths']

    # manually lookup, for num_baths also
    df.loc[3713, 'num_beds'] = 5.
    df.loc[3713, 'num_baths'] = 5.
    df.loc[5002, 'num_beds'] = 3.
    df.loc[5002, 'num_baths'] = 2.


# require size_sqft to be fixed first
def fill_na_num_beds_for_test(df):
    auto_fill_na_num_beds(df)

    na_mask = df['num_beds'].isna()
    # super big studio, often configured into multiple bedrooms, just follow num_baths here
    df.loc[na_mask, 'num_beds'] = df.loc[na_mask, 'num_baths']

    # manually lookup, for num_baths also
    df.loc[505, 'num_beds'] = 3.
    df.loc[505, 'num_baths'] = 2.
    df.loc[1086, 'num_beds'] = 6.
    df.loc[1086, 'num_baths'] = 6.
    df.loc[1558, 'num_beds'] = 3.
    df.loc[1558, 'num_baths'] = 2.
    df.loc[4855, 'num_beds'] = 3.
    df.loc[4855, 'num_baths'] = 2.


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


def fix_abnormal_sqft_for_test(df):
    # super large hdb, ranging from 5,000 to 12,000
    sqft_mask = (df['size_sqft'] > 5000) & (df['property_type'] == 'hdb')
    df.loc[sqft_mask, 'size_sqft'] = df.loc[sqft_mask, 'size_sqft'] / 10

    # special case
    df.loc[3340, 'size_sqft'] = 6674  # find from other same property records


def convert_sqm_to_sqft(df):
    sqm_to_sqft_multiplier = 10.764
    sqft_mask_sqm = (df['size_sqft'] < 200) & (df['size_sqft'] > 0)
    df.loc[sqft_mask_sqm, 'size_sqft'] = df.loc[sqft_mask_sqm, 'size_sqft'] * sqm_to_sqft_multiplier

    # cluster house
    df.loc[14218, 'size_sqft'] = df.loc[14218, 'size_sqft'] * sqm_to_sqft_multiplier
    df.loc[15027, 'size_sqft'] = df.loc[15027, 'size_sqft'] * sqm_to_sqft_multiplier


def convert_sqm_to_sqft_for_test(df):
    sqm_to_sqft_multiplier = 10.764
    sqft_mask_sqm = (df['size_sqft'] < 200) & (df['size_sqft'] > 0)
    df.loc[sqft_mask_sqm, 'size_sqft'] = df.loc[sqft_mask_sqm, 'size_sqft'] * sqm_to_sqft_multiplier


def rearrange_columns(df):
    cols = list(df.columns)
    lng_index = cols.index('lng')
    cols.insert(lng_index + 1, 'subzone_property_type_encoding')
    cols = cols[:-1]
    df = df[cols]
    return df


def move_price_to_last_column(df):
    cols = list(df.columns)
    cols.insert(len(cols), 'price')
    cols.remove('price')
    df = df[cols]
    return df


def target_encode_property_type_subzone(df):
    df_price_sqft = df.groupby(['subzone', 'property_type'])[['price', 'size_sqft']].sum().reset_index()
    df_price_sqft['sqft_price'] = df_price_sqft['price'] / df_price_sqft['size_sqft']
    df_price_sqft.drop(columns=['price', 'size_sqft'], inplace=True)
    encoding_dict = df_price_sqft.set_index(['subzone', 'property_type']).to_dict()['sqft_price']
    df['subzone_property_type_encoding'] = df.apply(lambda x: encoding_dict[(x['subzone'], x['property_type'])], axis=1)
    df.drop(columns=['subzone', 'property_type'], inplace=True)
    return encoding_dict


def generate_subzone_encoding_map(df):
    df_price_sqft = df.groupby('subzone')[['price', 'size_sqft']].sum().reset_index()
    df_price_sqft['sqft_price'] = df_price_sqft['price'] / df_price_sqft['size_sqft']
    df_price_sqft.drop(columns=['price', 'size_sqft'], inplace=True)
    encoding_dict = df_price_sqft.set_index('subzone').to_dict()['sqft_price']
    return encoding_dict


def generate_property_type_encoding_map(df):
    df_price_sqft = df.groupby('property_type')[['price', 'size_sqft']].sum().reset_index()
    df_price_sqft['sqft_price'] = df_price_sqft['price'] / df_price_sqft['size_sqft']
    df_price_sqft.drop(columns=['price', 'size_sqft'], inplace=True)
    encoding_dict = df_price_sqft.set_index('property_type').to_dict()['sqft_price']
    return encoding_dict


def target_encode_property_type_subzone_for_test(df, subzone_property_type_encoding, subzone_encoding, property_type_encoding):
    df['subzone_property_type'] = list(zip(df['subzone'], df['property_type']))
    df['subzone_property_type_encoding'] = np.nan

    # map by combination of subzone and property_type
    df['subzone_property_type_encoding'] = df['subzone_property_type'].map(subzone_property_type_encoding)

    # map by subzone only
    mask = df['subzone_property_type_encoding'].isna()
    df.loc[mask, 'subzone_property_type_encoding'] = df[mask]['subzone'].map(subzone_encoding)

    # map by property_type only
    mask = df['subzone_property_type_encoding'].isna()
    df.loc[mask, 'subzone_property_type_encoding'] = df[mask]['property_type'].map(property_type_encoding)

    df.drop(columns=['subzone', 'property_type', 'subzone_property_type'], inplace=True)


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


def encode_property_type(df):
    df_type_list = df['property_type'].to_list()
    type_list = sorted(df['property_type'].unique())
    type_dict = {k: v for v, k in enumerate(type_list)}
    ont_hot_length = len(type_list)
    one_hot_array = np.zeros((len(df_type_list), ont_hot_length))
    for i, type in enumerate(df_type_list):
        one_hot_index = type_dict[type]
        one_hot_array[i][one_hot_index] = 1.
    type_columns = ['pt_' + type.replace(' ', '_') for type in type_list]
    df_type = pd.DataFrame(one_hot_array, columns=type_columns)
    df.reset_index(drop=True, inplace=True)
    df_type.reset_index(drop=True, inplace=True)
    df = df.join(df_type)
    return df


def calculate_distance_km(df_property, df_target):
    len_property = len(df_property)
    len_target = len(df_target)
    distance_array = np.zeros([len_property, len_target])
    for idx_property in tqdm(range(len_property)):
        property_loc = df_property.iloc[idx_property, :]
        for idx_target in range(len_target):
            target_loc = df_target.iloc[idx_target, :]
            coord_property = (property_loc['lat'], property_loc['lng'])
            coord_target = (target_loc['lat'], target_loc['lng'])
            distance_km = distance.distance(coord_property, coord_target).km
            distance_array[idx_property][idx_target] = distance_km
    return distance_array


def populate_num_targets_within_range(df, df_property, distance_array, lower, upper, key):
    num_targets = ((distance_array < upper) & (distance_array >= lower)).sum(axis=1)
    df_property[key] = num_targets
    mapping_dict = df_property.to_dict()[key]
    df[key] = df['property_name'].map(mapping_dict)


def populate_distance_to_nearest_target(df, df_property, distance_array, key):
    distances = distance_array.min(axis=1)
    df_property[key] = distances
    mapping_dict = df_property.to_dict()[key]
    df[key] = df['property_name'].map(mapping_dict)


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

# property_type
def process_property_type(df):
    # Handle 'property_type' column (for test)
    df['property_type'] = df['property_type'].str.lower()
    df['property_type'] = df['property_type'].apply(lambda x: 'hdb' if 'hdb' in x else x)










