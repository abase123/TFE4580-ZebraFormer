import numpy as np
from scipy.signal import correlate
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import re
import itertools

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster,leaves_list
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import ks_2samp
from clusim.clustering import Clustering, remap2match

def fix_sign(df,cross_corr_df):
    # Iterate over rows and columns
    for i, row in df.iterrows():
        for col in df.columns:
            if cross_corr_df.at[i, int(col)] < 0:  # Checking the sign of corresponding value in df2
                df.at[i, col] = -df.at[i, col]  # Changing sign in df1

    return df

def filter_scaler(df):
    df = df[df.any(axis=1)]
    return df 





def compare_distributions(df1, df2):
    results = {}
    for column in df1.columns:
        if column in df2.columns:  # Ensure the column exists in both dataframes
            stat, p_value = ks_2samp(df1[column].dropna(), df2[column].dropna())
            results[column] = {'KS Statistic': stat, 'P-Value': p_value}
    return results
    
def get_corr_df(atten_scores,cross_corr_df):
    # Calculate correlations and store them in a dictionary
    correlation_dict = {}
    for item in atten_scores.columns:
        # np.corrcoef returns a matrix, we're interested in the [0, 1] element which is the correlation coefficient
        correlation = np.corrcoef(cross_corr_df[item], atten_scores[item])[0,1]
        correlation_dict[item] = correlation

    # Convert the dictionary to a DataFrame for better presentation
    correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['Pair', 'Correlation'])
    
    return correlation_df

def scale_attns(df,scaler):
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def preprocess_signals(signal1, signal2):
    """Subtract the mean from each signal."""
    signal1_centered = signal1 - np.mean(signal1)
    signal2_centered = signal2 - np.mean(signal2)
    return signal1_centered, signal2_centered

def standardize_cross_correlation(cross_correlations):
    """Standardize the cross-correlation values using Z-score."""
    standardized = zscore(cross_correlations)
    return standardized

def calculate_cross_correlation(signal1, signal2, num_lags):
    """Calculate the cross-correlation between two signals with given lags."""
    cross_corr = correlate(signal1, signal2, mode='full')
    center = len(cross_corr) // 2
    lags = np.arange(-center, center + 1)
    relevant_corrs = cross_corr[(center-num_lags):(center+num_lags+1)]
    return relevant_corrs, lags



# Function to extract cluster numbers and sort columns based on that
def sorted_columns_by_cluster(df, color):
    pattern = re.compile(rf"cluster_(\d+)_({color})")
    sorted_columns = sorted(df.columns, key=lambda x: int(pattern.search(x).group(1)))
    return sorted_columns


import numpy as np

def compute_cross_corr(pair):
    cluster_1_red_s1_paired = pair.iloc[:, 0]  # Single column as 1D array
    cluster_1_green_s1_paired = pair.iloc[:, 1]  # Single column as 1D array

    # Raw cross-correlation using 'full' mode
    cross_correlation = np.correlate(cluster_1_red_s1_paired - np.mean(cluster_1_red_s1_paired),
                                     cluster_1_green_s1_paired - np.mean(cluster_1_green_s1_paired),
                                     mode='full')
    
    
    
    # Normalizing factor
    norm_factor = np.sqrt(np.sum((cluster_1_red_s1_paired - np.mean(cluster_1_red_s1_paired))**2) *
                          np.sum((cluster_1_green_s1_paired - np.mean(cluster_1_green_s1_paired))**2))
    
    # Normalized cross-correlation
    cross_correlation = cross_correlation / norm_factor

    num_points = len(cluster_1_green_s1_paired)
    lags = np.arange(-num_points + 1, num_points)
    
    # Extract up to 50 lags
    max_lags = 50
    central_index = len(cross_correlation) // 2
    cc_limited = cross_correlation[(central_index - max_lags):(central_index + max_lags + 1)]
    lags_limited = lags[(central_index - max_lags):(central_index + max_lags + 1)]
    
    #Max lag
    max_correlation_value = np.max(np.abs(cc_limited))
    max_correlation_lag = lags_limited[np.argmax(np.abs(cc_limited))]
    
    return lags_limited, cc_limited, max_correlation_value


"""def compute_cross_corr(pair):
    # Using the functions
    cluster_1_red_s1_paired = pair.iloc[:, 0] # Single column as 1D array
    cluster_1_green_s1_paired = pair.iloc[:, 1] # Single column as 1D array


    cross_correlation = np.correlate(cluster_1_red_s1_paired.values, cluster_1_green_s1_paired.values, mode='full')
    num_points = len(cluster_1_green_s1_paired)
    lags = np.arange(-num_points + 1, num_points)
    # Extract up to 50 lags
    max_lags = 50
    central_index = len(cross_correlation) // 2
    cc_limited = cross_correlation[(central_index - max_lags):(central_index + max_lags + 1)]
    lags_limited = lags[(central_index - max_lags):(central_index + max_lags + 1)]
    
    #Max lag
    max_correlation_value = np.max(np.abs(cc_limited))
    max_correlation_value
    max_correlation_lag = lags_limited[np.argmax(np.abs(cc_limited))]
    

    return lags_limited,cc_limited"""


def get_column_pair(df, pair_number):
    # Calculate indices for the desired pair directly from pair_number using zero-based indexing
    index1 = pair_number * 2
    index2 = index1 + 1
    
    # Check if indices are within the column range
    if index2 < len(df.columns):
        # Return the selected columns
        return df.iloc[:, index1:index2 + 1]
    else:
        # Return an empty DataFrame or raise an error if the pair number is invalid
        return pd.DataFrame()

def create_z_score_df(paired_df,bin):
    PATH = "data/DatasetClusters/fishes/fish02/z_scores.csv"
    NUMBERS_OF_PAIR=384
    data_dict = {}
    data_dict_2 = {}
    
    for i in range(NUMBERS_OF_PAIR):
        pair = get_column_pair(df=paired_df,pair_number=i)
        pair_name = pair.columns[0] + "," +  pair.columns[1]
        l,c ,_ = compute_cross_corr(pair)
        binned_c,_ = bin_and_average(c, bin,l[0],l[-1])
        z = standardize_cross_correlation(binned_c)
        data_dict[i] = z
        data_dict_2[i] = binned_c
    
    z_scores_df = pd.DataFrame(data_dict)
    cross_corr_df = pd.DataFrame(data_dict_2)
    
    return z_scores_df,cross_corr_df



def standardize_attention(df):
    """Standardize the cross-correlation values in a DataFrame using Z-score."""
    # Applying zscore to each column. The `axis=0` argument standardizes along columns.
    standardized_df = df.apply(zscore, axis=0)
    
    # Optionally, you can handle NaN values if your data contains missing values
    # which can result from columns having the same value and thus a standard deviation of zero.
    standardized_df = standardized_df.fillna(0)
    
    return standardized_df



def bin_and_average(values, bin_size, min_lag, max_lag):
    # Create an array of lag indices from min_lag to max_lag
    lags = np.arange(min_lag, max_lag + 1)
    # Calculate the number of bins
    num_bins = len(values) // bin_size
    binned_values = []
    midpoints = []
    for i in range(num_bins):
        # Compute the start and end indices for the current bin
        start_index = i * bin_size
        end_index = start_index + bin_size
        # Calculate the average of the current bin
        bin_avg = np.mean(values[start_index:end_index])
        binned_values.append(bin_avg)
        # Calculate the midpoint of the lags for the current bin
        bin_midpoint = np.mean(lags[start_index:end_index])
        midpoints.append(bin_midpoint)
    return binned_values, midpoints


def perform_hierarchical_clustering(z_scores_df,name_image_dendo):
    # Transpose the DataFrame to make columns correspond to features
    data_transposed = z_scores_df.transpose()

    # Calculate the pairwise correlation distance matrix
    # This step translates correlation into a distance measure for clustering
    Y = pdist(data_transposed, 'correlation')

    # Perform hierarchical clustering using centroid linkage
    Z = linkage(Y, method='centroid')

    c2 = Clustering().from_scipy_linkage(Z, dist_rescaled=True)
    # Create a larger dendrogram to visualize the clustering
    plt.figure(figsize=(100, 50))  # Adjust the figsize parameter as needed for clarity
    dendrogram(
        Z,
        orientation='top',  # Change orientation to 'left'
        labels=data_transposed.index,
        leaf_font_size=15,  # Adjust font size to make labels readable
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Signal Pairs')
    plt.savefig(f"images/{name_image_dendo}.png")
    #plt.show()
    leaf_order = leaves_list(Z)
    ordered_columns = data_transposed.index[leaf_order]
    return Z,ordered_columns,c2




def generate_patch_list(n):
    # Create the negative patches
    neg_patches = [f"-{i}-patch" for i in range(n, 0, -1)]
    # The 'synch' entry
    synch = ['synch']
    # Create the positive patches
    pos_patches = [f"{i}-patch" for i in range(1, n+1)]
    # Combine all parts into the final list
    return neg_patches + synch + pos_patches







def perform_hierarchical_clusterin_attn(attn_score,name):
    # Transpose the DataFrame to make columns correspond to features
    data_transposed = attn_score.transpose()

    # Calculate the pairwise correlation distance matrix
    # This step translates correlation into a distance measure for clustering
    Y = pdist(data_transposed, 'correlation')

    # Perform hierarchical clustering using centroid linkage
    Z = linkage(Y, method='centroid')
    c2 = Clustering().from_scipy_linkage(Z, dist_rescaled=True)
    # Create a larger dendrogram to visualize the clustering
    plt.figure(figsize=(20, 180))  # Adjust the figsize parameter as needed for clarity
    dendrogram(
        Z,
        orientation='left',  # Change orientation to 'left'
        labels=data_transposed.index,
        leaf_font_size=15,  # Adjust font size to make labels readable
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title('Hierarchical Clustering Dendrogram From CrossFormer')
    plt.xlabel('Distance')
    plt.ylabel('Signal Pairs')
    plt.savefig(f"images/{name}.png")
    #plt.show()
    
    leaf_order = leaves_list(Z)
    ordered_columns = data_transposed.index[leaf_order]
    return Z,ordered_columns,c2





def perform_hierarchical_clusterin_heat(attn_score,name):
    # Transpose the DataFrame to make columns correspond to features
    data_transposed = attn_score.transpose()

    # Calculate the pairwise correlation distance matrix
    # This step translates correlation into a distance measure for clustering
    Y = pdist(data_transposed, 'correlation')

    # Perform hierarchical clustering using centroid linkage
    Z = linkage(Y, method='complete')

    # Create a larger dendrogram to visualize the clustering
    plt.figure(figsize=(20, 180))  # Adjust the figsize parameter as needed for clarity
    dendrogram(
        Z,
        orientation='left',  # Change orientation to 'left'
        labels=data_transposed.index,
        leaf_font_size=15,  # Adjust font size to make labels readable
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title('Hierarchical Clustering Dendrogram From CrossFormer')
    plt.xlabel('Distance')
    plt.ylabel('Signal Pairs')
    plt.savefig(f"images/{name}.png")
    #plt.show()
    
    leaf_order = leaves_list(Z)
    ordered_columns = data_transposed.index[leaf_order]
    return Z,ordered_columns




def standardize_attention(df):
    """Standardize the cross-correlation values in a DataFrame using Z-score."""
    # Applying zscore to each column. The `axis=0` argument standardizes along columns.
    standardized_df = df.apply(zscore, axis=0)
    
    # Optionally, you can handle NaN values if your data contains missing values
    # which can result from columns having the same value and thus a standard deviation of zero.
    standardized_df = standardized_df.fillna(0)
    
    return standardized_df




def quantile_transform_data(df, output_dist='normal'):
    transformer = QuantileTransformer(output_distribution=output_dist, random_state=0)
    quantile_transformed_df = pd.DataFrame(transformer.fit_transform(df), columns=df.columns)
    return quantile_transformed_df

def rolling_quantile_transform(df, window_size=10, output_dist='normal', n_quantiles=19):
    # Initialize the transformer with a specified number of quantiles
    transformer = QuantileTransformer(output_distribution=output_dist, n_quantiles=n_quantiles, random_state=0)

    # Placeholder for transformed data
    transformed_data = pd.DataFrame(index=df.index, columns=df.columns)

    # Apply transformation in rolling windows
    for start in range(0, len(df), window_size):
        end = start + window_size
        # Slice the DataFrame to the current window
        current_data = df[start:end]
        # Ensure the transformer's n_quantiles does not exceed the number of samples
        effective_quantiles = min(n_quantiles, len(current_data))
        transformer.n_quantiles = effective_quantiles
        # Transform the data in the current window and replace in the placeholder DataFrame
        transformed_data.iloc[start:end] = transformer.fit_transform(current_data)
    
    return transformed_data


def combined_quantile_moving_average_same_window(df, window_size=10, output_dist='normal', n_quantiles=10):
    # Initialize the Quantile Transformer with a specified number of quantiles
    transformer = QuantileTransformer(output_distribution=output_dist, n_quantiles=n_quantiles, random_state=0)

    # Placeholder DataFrame for the final transformed data
    transformed_data = pd.DataFrame(index=df.index, columns=df.columns)

    # Apply transformation and moving average within the same window
    for start in range(0, len(df), window_size):
        end = start + window_size
        # Slice the DataFrame to the current window
        current_data = df[start:end]

        # Adjust n_quantiles to match the number of samples in the window if necessary
        effective_quantiles = min(n_quantiles, len(current_data))
        transformer.n_quantiles = effective_quantiles

        # Fit and transform the data in the current window
        quantile_transformed = transformer.fit_transform(current_data)
        quantile_transformed_df = pd.DataFrame(quantile_transformed, index=current_data.index, columns=df.columns)

        # Apply the moving average filter to the quantile-transformed data in the same window
        moving_average_df = quantile_transformed_df.rolling(window=window_size, min_periods=1, center=True).mean()

        # Assign the moving averaged data back to the placeholder DataFrame
        transformed_data.iloc[start:end] = moving_average_df

    return transformed_data



from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler

def process_data_window(df, window_size=10, winsor_limits=(0.05, 0.05)):
    scaler = RobustScaler()
    processed_data = pd.DataFrame(index=df.index, columns=df.columns)

    for start in range(0, len(df), window_size):
        end = start + window_size
        window_data = df[start:end]

        # Winsorize data to limit extreme values
        winsorized_data = window_data.apply(lambda x: winsorize(x, limits=winsor_limits), axis=0)
        
        # Apply robust scaling
        scaled_data = scaler.fit_transform(winsorized_data)
        processed_data.iloc[start:end] = scaled_data

    return processed_data





def redistribute_outliers(row, threshold=0.6):
    # Iterate through each cell in the row
    for idx, value in enumerate(row):
        if value > threshold:
            # Calculate the amount to redistribute
            redistribute_amount = value * 0.3  # 10% of the outlier value
            # Dampen the outlier
            row[idx] -= redistribute_amount

            # Find indices of non-NaN neighbors
            non_nan_indices = [i for i in range(len(row)) if not np.isnan(row[i]) and i != idx]
            # Calculate weights decreasing with distance from the outlier
            weights = np.exp(-0.5 * np.abs(np.array(non_nan_indices) - idx))
            weights /= weights.sum()  # Normalize weights

            # Redistribute the amount to non-NaN neighbors
            for neighbor_idx, weight in zip(non_nan_indices, weights):
                if (row[neighbor_idx] < 0.3):
                    row[neighbor_idx] += redistribute_amount * weight

    return row




# Function to apply custom robust scaling
def custom_robust_scale(column):
    median = column.median()
    q3 = column.quantile(0.80)
    modified_iqr = q3 - median  # Custom IQR focusing only on the upper half
    return (column - median) / modified_iqr



def create_cluster_dic(clusters):
    cluster_group = {}
    labels = np.arange(384)

    for i in range(len(clusters)):
        cluster_group[clusters[i]] = []
        
    for key in cluster_group.keys():
        for ind in range(len(labels)):
            if(clusters[ind] == key):
                cluster_group[key].append(ind)
                
    return cluster_group 
            