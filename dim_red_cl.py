'''
--------------------------------------------------------------------------------
Comparison of dimensionality reduction and clustering on the handwritten digits dataset
--------------------------------------------------------------------------------

The cluster quality metrics evaluated for K-means are as follows:
________________________________________________________________________________________________________
Shorthand    full name                      Score ranges       Indicators of similarity between clusters
=========    ===========================    ============       =============================================================================
homo         homogeneity score              0.0 to 1.0         1.0 stands for perfectly homogeneous labeling
compl        completeness score             0.0 to 1.0         1.0 stands for perfectly complete labeling
v-meas       V measure                      0.0 to 1.0         1.0 stands for perfectly complete labeling
ARI          adjusted Rand index            -1.0 to 1.0.       Random labelings have an ARI close to 0.0. And 1.0 stands for a perfect match.
AMI          adjusted mutual information    0 to 1             1 when the partitions are identical (i.e., perfectly matched). Random partitions (independent labellings) have an expected AMI of ~0 on average and can be negative.
F-M          Fowlkes-Mallows index          0 to 1             A high value indicates a good similarity between two clusters.
silhouette   silhouette coefficient         -1 to 1            The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster.
_________________________________________________________________________________________________________

'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation

import figures
import statistics

show_plots = False



# Load the digits dataset
digits = load_digits()

# Standardize a dataset along any axis 
data = scale(digits.data) 

# This data from scikit-learn is stored in the .data member, which is an (n_samples, n_features) array
n_samples, n_features = data.shape
# print ('Shape: ', digits.data.shape) # Number of samples (1797) and number of features (64)

# The class of each observation is stored in the .target attribute of the dataset 
n_digits = len(np.unique(digits.target))
labels = digits.target

dig = 1 # Digit number. Change this digit number (to a digit between 0 and 9) in order to get a list printed of the sample numbers that correspond to this specified digit
# print ('Samples that are {}:'.format(dig), np.where(labels == dig)[0]) # Prints a list of sample numbers that are the specified digit

# ####################################################################################
# Visualize the dataset 

sample_n = 138 # Input number for which sample to visualize
if show_plots:
    if 0: # Plot of a single image from the dataset. Type 1 instead of 0 in this if statement to display the figure
        plt.figure(0, figsize=(3, 3))
        plt.imshow(digits.images[sample_n], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
        # print ('Digit number {} is sample number {}'.format(labels[sample_n], sample_n)) # Print digit number and sample number
        plt.show()
        # plt.savefig('example_digit_{}.png'.format(labels[sample_n]), dpi=150) 

# Plot of four examples from each of the 10 digits (0 through 9)
sample_lst = [] # A list of lists. Each list within contains four sample numbers of a single digit, and it goes from digit 0 (the first list within the list of lists) to digit 9 (the last list)
for i in range(10):
    sample_nums = np.where(labels == i)[0][0:4] # Indexes (0:4) into an array of sample numbers that are a specified digit (the digit is written as i in the loop)
    sample_nums = sample_nums.tolist() # Convert the array called sample_nums to a list with the same items
    sample_lst.append(sample_nums) 
if show_plots:
    if 0:
        fig = plt.figure(figsize=(10,6))
        plt_index = 0
        for i in range(0,10):
            for j in sample_lst[i]:
                plt_index = plt_index + 1
                ax = fig.add_subplot(5, 8, plt_index)
                ax.imshow(digits.images[j], cmap=plt.cm.gray_r, interpolation='nearest')
                ax.set_yticklabels([]) # Turn off y tick labels
                ax.set_xticklabels([]) # Turn off x tick labels
                ax.set_yticks([]) # Turn off y ticks
                ax.set_xticks([]) # Turn off x ticks
        plt.tight_layout()
        # plt.show()  
        plt.savefig('digits_each_4.png', dpi=150) 

def plot_ten_images(image_data, data, model, shape=None):
    """
    Plot ten images along one row
    image_data: the image data, for example digits.images
    data: the data, for example, digits.data
    model: fit the model in order to obtain the inverse transform, for example, the input could be pca(n_components=2).fit(data) 
    shape: Input the size of the image (separated by a comma), for example, if the image is 8x8, you would input as a parameter: (8, 8)
    """
    fig = plt.figure(figsize=(10,1.5))
    plt_index = 0 
    if model:
        model_result = model.transform(data)
        # Transform the PCA reduced data back to its original space. This will help visualize how the image data appears after PCA reduction
        model_inversed = model.inverse_transform(model_result) 
    for i in range(10):
        plt_index = plt_index + 1
        ax = fig.add_subplot(1, 10, plt_index)
        if shape:
            ax.imshow(model_inversed[i].reshape(shape), cmap=plt.cm.gray_r, interpolation='nearest')
        else:
            ax.imshow(image_data[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    plt.show()  

def plot_dim_red(dim_red, data, title=None, xlabel=None, ylabel=None):
    """
    Scatter plot of dimensionality reduced data.
    dim_red: dimensionality reduced data
    data: data
    title: string of the title
    xlabel: string of the x label
    ylabel: string of the y label
    """
    plt.scatter(dim_red[:, 0], dim_red[:, 1], c=digits.target, cmap=plt.cm.get_cmap('tab10', 10), marker='o', edgecolor='none', alpha=0.5) 
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    if xlabel:
        plt.xlabel('Component 1')
    if ylabel:
        plt.ylabel('Component 2')
    if title:
        plt.title(title)
    plt.show()

def plot_dim_red_clust(dim_red, cluster_type, data, dim_red_name=None, cluster_type_name=None, centroids=None):
    """
    Visualize the results of clustering on dimensionality reduced data.
    Plots the dimensionality reduced data vs. the clustered dimensionality reduced data
    Parameters:
        dim_red: dimensionality reduction method
        cluster_type: the type of clustering algorithm, for example, from the Scikit-learn library, one can use the k-means implementatation as sklearn.cluster.KMeans()
        dim_red_name: name of dimesnionality reduction to be in the title of the corresponding plot
        cluster_type_name: name of clustering method to be in the title of the corresponding plot
        centroids: plots the coordinates of cluster centers 
    """
    color = cluster_type.labels_
    fig, axarr = plt.subplots(1, 2, figsize=(9,4))  
    # Plot of dimensionality reduced data 
    ax1 = axarr[0]
    ax1.scatter(dim_red[:,0], dim_red[:,1], c='k', marker='.')
    if dim_red_name:
        ax1.set_title('{} reduced data'.format(dim_red_name))
    # Plot of clustering on dimensionality reduced data
    ax2 = axarr[1]
    ax2.scatter(dim_red[:,0], dim_red[:,1], c=color, marker='.')
    if centroids:
        centroids = cluster_type.cluster_centers_
        ax2.scatter(centroids[:, 0], centroids[:, 1],
                    marker='o', edgecolors='k', 
                c=np.arange(len(np.unique(digits.target))), zorder=10)
    if cluster_type_name:
        ax2.set_title('{} clustering on {} reduced data'.format(cluster_type_name, dim_red_name))
    plt.tight_layout()
    plt.show()

# ###########################################################################################
# PCA-reduced data

# The first two principal components are generated below (on the standardized data) 
pca = PCA(n_components=2) # The original 64 dimensions are reduced to 2
pca_result = pca.fit_transform(data) # Now we have the reduced feature set
# print ('PCA reduced shape: ', pca_result.shape) # Number of samples (1797) in the dataset and PCA reduced number of features
# print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)) # Percentage of variance explained by each dimension. Here, the n_components=2, as designated in the pca above, so two values are expected
# print ('The first two components account for {:.0f}% of the variation in the entire dataset'.format((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100))

# Principal components are assigned to the percnt variable below and PCA is done on the non-standardized data
percnt = 2 # Input the percentage of variance (e.g., 0.50 would preserve 50% of the variance)
pca3 = PCA(n_components=percnt).fit(digits.data) 
pca3_result = pca3.transform(digits.data)
# Transform the PCA reduced data back to its original space. This will help visualize how the image data appears after PCA reduction
pca_inversed = pca3.inverse_transform(pca3_result)  # Use the inverse of the transform to reconstruct the reduced digits

if percnt < 1:
    percnt_var = round(int(percnt*100))
    # print ('{:0d}% of the variance is contained within {} principal components'.format(percnt_var, pca3.n_components_))

# One can estimate how many components are needed to describe the data by assessing the 
# cumulative explained variance ratio as a function of the number of components, see the plot below
if show_plots:
    if 0:
        pca4 = PCA().fit(digits.data) # Conducting PCA without standardizing the data and without providing a number of components
        plt.plot(np.cumsum(pca4.explained_variance_ratio_)) # The cumulative sum of the percentage of variance explained by all of the components
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('Number of components vs. retained variance')
        plt.show() # Note, about 90% of the variance is retained by using 20 components, but 75% is retained within 10 components
        # plt.savefig('n_comp_var.png', dpi=150)  

# #####################
# Visualize results of PCA

# Plot of PCA reduced data with a colorbar. This uses the standardized data
if show_plots:
    plot_dim_red(pca_result, data, 'PCA reduced handwritten digits', 'Component 1', 'Component 2')

    plot_ten_images(digits.images, digits.data, model=None) # Figure of ten original images
    plot_ten_images(digits.images, digits.data, pca3, shape=(8, 8)) # Figure of images that have undergone PCA reduction. The inverse transform is shown here

# #############################################################################
# t-SNE reduced data

n_iter = 1000
n_perplexity = 40

t_sne = TSNE(n_components=2, perplexity=n_perplexity, n_iter=n_iter, early_exaggeration=4)
tsne_result = t_sne.fit_transform(data) # This is t-SNE reduced data

if show_plots:   
    fig, axarr = plt.subplots(1, 2, figsize=(8,4))  
    # Plot of unlabeled of t-SNE dimensionality reduced data
    ax1 = axarr[0]
    ax1.scatter(tsne_result[:,0], tsne_result[:,1], c='k', marker='.')
    # Plot of labeled of t-SNE dimensionality reduced data
    ax2 = axarr[1]
    ax2.scatter(tsne_result[:,0], tsne_result[:,1], c=digits.target, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('tab10', 10), marker='.')
    plt.tight_layout()
    plt.show()



# Figure of multiple plots of different perplexities
if show_plots:
    # Plot of t-SNE reduced data 
    plot_dim_red(tsne_result, data, 't-SNE reduced handwritten digits')

    if 0:
        perplex_lst = (2,5,30,50,100)
        fig = plt.figure(figsize=(10,2.2))
        plt_index = 0
        for i in perplex_lst:
            plt_index = plt_index + 1
            t_sne = TSNE(n_components=2, perplexity=i, n_iter=n_iter)
            tsne_result = t_sne.fit_transform(data) # This is t-SNE reduced data
            ax = fig.add_subplot(1, 5, plt_index)
            ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                        c=digits.target, edgecolor='none', alpha=0.5,
                        cmap=plt.cm.get_cmap('tab10', 10), marker='.') 
            ax.set_title('perplexity = {}'.format(i))
            ax.set_yticklabels([]) # Turn off y tick labels
            ax.set_xticklabels([]) # Turn off x tick labels
            ax.set_yticks([]) # Turn off y ticks
            ax.set_xticks([]) # Turn off x ticks
        plt.tight_layout()
        plt.show()

# #############################################################################
# Truncated SVD reduced data

svd = TruncatedSVD(n_components=2, n_iter=10)
svd_result = svd.fit_transform(data)  

# Plot of Truncated SVD reduced data 
if show_plots:
    plot_dim_red(svd_result, data, 'Truncated SVD reduced handwritten digits')

# #############################################################################
# Isomap reduced data

iso = Isomap(n_components=2)
iso_result = iso.fit_transform(data)

# Plot of isomap reduced data 
# plot_dim_red(iso_result, data, 'Isomap reduced handwritten digits')

# #############################################################################
# Metrics for K-means clustering

n_clusters = n_digits
n_init = 10
sample_size = 300

# Metrics to evaluate the model
print('init\t\t homo\t compl\t v-meas\t ARI\t AMI\t F-M\t silhouette')

def kmeans_metrics(estimator, name, data):
    """
    Compare different initializations of K-means to assess the quality of the clustering.

    Parameters:
        estimator: K-means algorithm with parameters to pass (init, n_clusters, and n_init)
        name: name of the method for initialization
        data: data
    """
    estimator.fit(data)
    print ('{:<9}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}'
    .format(name, metrics.homogeneity_score(labels, estimator.labels_),
    metrics.completeness_score(labels, estimator.labels_),
    metrics.v_measure_score(labels, estimator.labels_),
    metrics.adjusted_rand_score(labels, estimator.labels_),
    metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
    metrics.fowlkes_mallows_score(labels, estimator.labels_),
    metrics.silhouette_score(data, estimator.labels_,
                                metric='euclidean',
                                sample_size=sample_size)))

kmeans_metrics(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
              name="k-means++", data=data)

kmeans_metrics(KMeans(init='random', n_clusters=n_clusters, n_init=10),
              name="random", data=data)

# In this case the seeding of the centers is deterministic, thus the kmeans algorithm is run only once (i.e., n_init=1)
pca = PCA(n_components=n_digits).fit(data)
kmeans_metrics(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
              name="PCA-based",
              data=data)

# ###################
# K-means clustering

def k_means_reduced(reduced_data, initialization, n_clusters, n_init):
    """
    This returns K-means clustering on data that has undergone dimensionality reduction.
    Parameters:
        reduced_data: The data that has undergone dimensionality reduction
        initialization: Method for initialization, defaults to ‘k-means++’:
        n_clusters: The number of clusters to form as well as the number of centroids to generate.
        n_init: Number of times the k-means algorithm will run with different centroid seeds.
    """
    k_means = KMeans(init=initialization, n_clusters=n_clusters, n_init=n_init) 
    k_means_model = k_means.fit(reduced_data)
    return k_means_model

# K-means clustering on PCA reduced data
k_pca = k_means_reduced(pca_result, 'k-means++', n_clusters, n_init)
# K-means clustering on t-SNE reduced data
k_t_sne = k_means_reduced(tsne_result, 'k-means++', n_clusters, n_init)
# K-means clustering on Truncated SVD reduced data
k_SVD = k_means_reduced(svd_result, 'k-means++', n_clusters, n_init)
# K-means clustering on isomap reduced data
k_iso = k_means_reduced(iso_result, 'k-means++', n_clusters, n_init)

# #####################
# Visualize results of K-means clustering

# Figure of PCA vs t-SNE reduced data and K-means clustering on both
if show_plots: 
    fig, axarr = plt.subplots(2, 2, figsize=(10,8)) 

    # Plot of PCA reduced data 
    axarr[0, 0].scatter(pca_result[:,0],pca_result[:,1],c='k')
    axarr[0, 0].set_title('PCA reduced data')
    
    # Plot of t-SNE reduced data 
    axarr[1, 0].scatter(tsne_result[:,0], tsne_result[:,1],c='k')
    axarr[1, 0].set_title('t-SNE reduced data')

    # Plot of K-means clustering on PCA reduced data
    cluster_ax = axarr[0, 1]
    cluster_ax.scatter(pca_result[:,0],pca_result[:,1],c=k_pca.labels_, marker='o', edgecolor='none', alpha=0.5)
    centroids = k_pca.cluster_centers_
    centroid_x = cluster_ax.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', edgecolors='k', 
                c=np.arange(n_clusters), zorder=10)
    cluster_ax.set_title('K-means clustering on PCA reduced data')

    # Plot of K-means clustering on t-SNE reduced data
    cluster_ax2 = axarr[1, 1]
    cluster_ax2.scatter(tsne_result[:,0],tsne_result[:,1],c=k_t_sne.labels_, marker='o', edgecolor='none', alpha=0.5)
    centroids = k_t_sne.cluster_centers_
    centroid_x = cluster_ax2.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', edgecolors='k', 
                c=np.arange(n_clusters), zorder=10) # If you want the cluster center marker as an 'x', use: marker='x', s=100, linewidths=5, color='k', zorder=10
    cluster_ax2.set_title('K-means clustering on t-SNE reduced data')
    fig.suptitle('K-means clustering on the handwritten digits dataset')
    # plt.show()
    plt.savefig('pca_tsne_k_means.png', dpi=150)

if show_plots:
    # Plot of k-means clustering on t-SNE reduced data
    plot_dim_red_clust(tsne_result, k_t_sne, data, 't-SNE', 'K-means', centroids=True)
    # Plot of k-means clustering on pca reduced data
    plot_dim_red_clust(pca_result, k_pca, data, 'PCA', 'K-means', centroids=True)
    # Plot of K-means clustering on isomap reduced data
    plot_dim_red_clust(iso_result, k_iso, data, 'isomap', 'K-means')
    # Plot of K-means clustering on truncated SVD reduced data
    plot_dim_red_clust(svd_result, k_SVD, data, 'truncated SVD', 'K-means')

def elbow_curve(dim_red_data):
    """
    Plots an elbow curve to select the optimal number of clusters (k) for k-means clustering.
    Fit KMeans and calculate sum of squared errors (SSE) for each cluster (k), which is 
      defined as the sum of the squared distance between each member of the cluster and its centroid.
    Parameter:
        dim_red_data: Dimensionality reduced data
    """
    sse = {}
    for k in range(1, 40): 
        # Initialize KMeans with k clusters and fit it 
        kmeans = KMeans(n_clusters=k, random_state=0).fit(dim_red_data)  
        # Assign sum of squared errors to k element of the sse dictionary
        sse[k] = kmeans.inertia_ 
    # Add the plot title, x and y axis labels
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    # Plot SSE values for each k stored as keys in the dictionary
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()

if show_plots:
    elbow_curve(tsne_result)

# When the plot is made, we can approximately see that the line levels off approximately after about 10 clusters, implying that 
# the addition of more clusters may not explain much more of the variance 
# To determine the bend in the knee, see https://github.com/arvkevi/kneed and https://raghavan.usc.edu/papers/kneedle-simplex11.pdf

# #########
# Visualize random digits from a select k-means cluster using t-SNE reduced data

clust_num = 3 # The cluster number to select
indexes = 8 # The number of plots to generate on the figure. 
fig_size = (7,8)

def cluster_indices(clust_num, labels_array): #numpy 
    """
    This takes parameters such as the cluster number (clust_num) and the labels of each data point.
    This returns the indices of the cluster_num you provide.
    """
    return np.where(labels_array == clust_num)[0]

cluster_data = cluster_indices(clust_num, k_t_sne.labels_) # A list of sample numbers from the chosen cluster number
cl_mode = statistics.mode(labels[cluster_data])
dig_lst = np.where(digits.target == cl_mode)[0]
# print (("where:"), np.where(k_t_sne.labels_ == 6)[0]) # This is telling us the sample number where the digit 6 is

print ('Samples from cluster {}:'.format(clust_num), cluster_data)
print ('Digits from cluster {}:'.format(clust_num), labels[cluster_data])

cluster_data_random = []
for i in range(indexes):
    rand_num = np.random.choice(cluster_data)
    cluster_data_random.append(rand_num)
# Plot of random digits from a chosen cluster
if show_plots:
    fig = plt.figure(figsize=fig_size)
    plt_index = 0
    for i in cluster_data_random:
        plt_index = plt_index + 1
        ax = fig.add_subplot((indexes-(indexes/2)), 2, plt_index)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        fig.suptitle('Random digits from k-means cluster number {}'.format(clust_num), fontsize=14)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def digit_matches(cluster_data):
    """
    This checks whether or not the mode of a cluster matches each digit within that cluster.
    It returns whether or not these two match. If all digits of a particular cluster are the same, this function prints: 'Match!'
    """
    cl_mode = statistics.mode(labels[cluster_data])
    dig_lst = np.where(digits.target == cl_mode)[0]
    if dig_lst.shape == cluster_data.shape:
        matches = (dig_lst == cluster_data).all()
    else:
        matches = dig_lst == cluster_data
    if matches:
        print ('All cluster digits are the same digit')
    else:
        print ('The cluster digits are not all the same digit')

# Print whether or not the chosen cluster has all one digit 
digit_matches(cluster_data)

def where_not_equal(cluster_data, labels):
    """
    cluster_data are indices (of samples)
    This takes a parameter of another function, the digit_matches function defined above. 
    This returns the indices of the digits that do not match the mode digit within the chosen cluster."""
    cl_mode = statistics.mode(labels[cluster_data])
    return np.where(labels[cluster_data] != cl_mode)[0]

print ('Where the digits are not equal list: ', len(where_not_equal(cluster_data, labels)))
print ('Where the digits are not equal: ', where_not_equal(cluster_data, labels))

sampl_lst = []
digit_lst = []
wher_lst = []
for i in range(10):
    cl_num = i
    cl_data = cluster_indices(cl_num, k_t_sne.labels_) 
    sampl_lst.append(cl_data)
    digit_lst.append(labels[cl_data])
    not_equal = where_not_equal(cl_data, labels).tolist()
    wher_lst.append(not_equal)
print (wher_lst)
# #############################################################################
# Mean-shift clustering
# We can use the estimate_bandwidth function to estimate a good bandwidth for the data
bandwidth = round(estimate_bandwidth(data))

mean_s = MeanShift(bandwidth=bandwidth)
mean_s.fit(pca_result)

ms = MeanShift(bandwidth=bandwidth)
ms_tsne = ms.fit(tsne_result)
ms_labels = ms_tsne.labels_
ms_cluster_centers = ms_tsne.cluster_centers_

ms_labels_unique = np.unique(ms_labels)
ms_n_clusters = len(ms_labels_unique)

print ('The number of estimated clusters from mean-shift clustering is: {}'.format(ms_n_clusters))

# #########
# Visualize the results of Mean-shift clustering
if show_plots: 
    color = ms_labels
    fig, axarr = plt.subplots(1, 2, figsize=(9,4))  
    # Plot of t-SNE reduced data 
    ax1 = axarr[0]
    ax1.scatter(tsne_result[:,0], tsne_result[:,1], c='k', marker='.')
    ax1.set_title('t-SNE reduced data')

    # Plot of mean shift clustering on t-SNE reduced data
    ax2 = axarr[1]
    ax2.scatter(tsne_result[:,0], tsne_result[:,1], c=color, marker='.')
    cluster_center = ms_cluster_centers[ms_labels]
    ax2.scatter(cluster_center[:,0], cluster_center[:,1],
                marker='o', edgecolors='k', 
                c=color, zorder=10) 
    ax2.set_title('Mean-shift clustering on t-SNE reduced data')
    plt.tight_layout()
    plt.show()

if show_plots:
    # Plot of mean-shift clustering on t-SNE reduced data
    plot_dim_red_clust(tsne_result, ms_tsne, data, 't-SNE', 'Mean-shift')

# #############################################################################
# Spectral clustering
# spectral clustering can ignore sparse interconnections between arbitrarily shaped clusters of data

# sc_result = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize").fit(data)
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                           assign_labels='kmeans')  
sc_tsne = sc.fit(tsne_result)                
sc_labels = sc.fit_predict(tsne_result)

if show_plots:
    # Visualize the results of spectral clustering
    plot_dim_red_clust(tsne_result, sc_tsne, data, 't-SNE', 'Spectral')

# #############################################################################
# DBSCAN clustering
# For some more information, see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# See: http://abhijitannaldas.com/ml/dbscan-clustering-in-machine-learning.html
# See: https://www.dummies.com/programming/big-data/data-science/how-to-create-an-unsupervised-learning-model-with-dbscan/
# See: https://towardsdatascience.com/dbscan-clustering-for-data-shapes-k-means-cant-handle-well-in-python-6be89af4e6ea

db = DBSCAN(eps=3, min_samples=2)
db_tsne = db.fit(tsne_result)

if show_plots:
    # Visualize the results of DBSCAN clustering
    plot_dim_red_clust(tsne_result, db_tsne, data, 't-SNE', 'DBSCAN')

# #############################################################################
# Affinity propagation clustering

ap = AffinityPropagation().fit(tsne_result)
cluster_centers = ap.cluster_centers_

if show_plots:
    # Visualize the results of Affinity propogation clustering
    plot_dim_red_clust(tsne_result, ap, data, 't-SNE', 'Affinity propogation')