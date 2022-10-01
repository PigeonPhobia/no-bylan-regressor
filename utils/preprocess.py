import numpy as np

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
