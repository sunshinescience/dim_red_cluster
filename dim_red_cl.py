'''
--------------------------------------------------------------------------------
Comparison of dimensionality reduction and clustering on the handwritten digits dataset
--------------------------------------------------------------------------------

The cluster quality metrics evaluated are as follows:
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

# Load the digits dataset
digits = load_digits()

# Standardize a dataset along any axis 
data = scale(digits.data) 

# This data from scikit-learn is stored in the .data member, which is a (n_samples, n_features) array
n_samples, n_features = data.shape
print ('Shape: ', digits.data.shape) # Number of samples (1797) in the dataset and number of features (64)

# The class of each observation is stored in the .target attribute of the dataset 
n_digits = len(np.unique(digits.target))
labels = digits.target

# print (digits.images[0].shape) # Checking if an example image (0) is grayscale. If it is, the tuple returned contains only the number of rows and columns

# Note: to display a figure, type 1 instead of 0 in the corresponding if statement in the relevant figures below

# ##############
# Visualize images from the dataset. 

# Visualize a single image from the dataset. 
sample_n = 32 # Input number for which sample to visualize
if 0: 
    plt.figure(0, figsize=(3, 3))
    plt.imshow(digits.images[sample_n], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    print ('Digit number:', labels[sample_n]) # Digit number
    print ('Sample number: ', sample_n) # Sample number
    plt.show()
    # plt.savefig('example_digit_{}.png'.format(labels[sample_n]), dpi=150) 

# List to use in the below code that provides four example digits for each digit in a figure
sample_lst = [] # A list of lists. Each list within contains four sample numbers of a single digit, and it goes from digit 0 (the first list within the list of lists) to digit 9 (the last list)
for i in range(10):
    sample_nums = np.where(labels == i)[0][0:4] # indexes (0:4) into an array of sample numbers that are a specified digit (the digit is written as i in the loop)
    sample_nums = sample_nums.tolist() # Convert the array called sample_nums to a list with the same items
    sample_lst.append(sample_nums) 
# print (sample_lst)

dig = 0 # Digit number. Change this digit number in order to get a list printed of the sample numbers that correspond to this specified digit
# print ('Samples that are {}:'.format(dig), np.where(labels == dig)[0]) # Prints a list of sample numbers that are the specified digit

# Plot of four examples of each specific digit, from each of the 10 digits (0 through 9)
if 1:
    fig = plt.figure(figsize=(10,6))
    plt_index = 0
    for i in range(0,10):
        for j in sample_lst[i]:
            plt_index = plt_index + 1
            ax = fig.add_subplot(5, 8, plt_index)
            ax.imshow(digits.images[j], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace = 0.8)
    plt.tight_layout()
    plt.show()  
    # plt.savefig('digits_each_4.png', dpi=150) 

index = 8 # The number of plots to generate on the figures below. 
fig_sz = (7,8) # The corresponding figure size. 

n_rand = []
for i in range(index):
    n_rand.append(np.random.randint(0, n_samples))
print (n_rand)

# Plot of eight random digits from the dataset
if 0:
    fig = plt.figure(figsize=fig_sz)
    plt_index = 0
    for i in n_rand:
        plt_index = plt_index + 1
        ax = fig.add_subplot((index-(index/2)), 2, plt_index)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    # plt.show()

n_rand_large = []
n = 36 # Number of plots of the digits to make
for i in range(n):
    n_rand_large.append(np.random.randint(0, n_samples))
print (n_rand_large)

# Plot of many random digits from the dataset
if 0:
    fig = plt.figure(figsize=(8,9))
    plt_index = 0
    for i in n_rand_large:
        plt_index = plt_index + 1
        ax = fig.add_subplot(6, 6, plt_index)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    # plt.show()
    # plt.savefig('digits_36random.png', dpi=150)

# Plot of ten example digits (from 0 to 9)
if 1:
    fig = plt.figure(figsize=(10,5))
    plt_index = 0
    for i in range(10):
        plt_index = plt_index + 1
        ax = fig.add_subplot(2, 5, plt_index)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace = 0.8)
    plt.tight_layout()
    plt.show()  
    # plt.savefig('digits_10example.png', dpi=150) 

# #############################################################################
# PCA-reduced data

# From the original 64 dimensions, the first two principal components are generated below on the standardized data
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data) # now we have the reduced feature set
print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)) # Percentage of variance explained by each dimension
print ('The first two components account for {:.0f}% of the variation in the entire dataset'.format((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100))
# print ('PCA reduced shape: ', pca_result.shape) # Number of samples (1797) in the dataset and PCA reduced number of features

# The first two principal components are generated below on the non-standardized data
pca2 = PCA(n_components=2)  # reduce from 64 to 2 dimensions
pca_result2 = pca2.fit_transform(digits.data)
# print ('Non-standardized PCA reduced data shape: ', pca_result2.shape)

# Plot of PCA reduced data with a colorbar. This uses the standardized data
if 0:
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=digits.target, cmap=plt.cm.get_cmap('tab10', 10), marker='.') 
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show() 
    # plt.savefig('pca_standardized_digits.png', dpi=150) 

# Plot of PCA reduced data with a colorbar. This uses the original data, (i.e., this data is not standardized)
if 1:
    plt.scatter(pca_result2[:, 0], pca_result2[:, 1],
                c=digits.target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show() 
    # plt.savefig('pca_digits.png', dpi=150) 

# The principal components are assigned to the percnt variable below and PCA is done on the non-standardized data
percnt = 0.50 # Input the percentage of variance (e.g., 0.50 would be 50% of the variance)
pca3 = PCA(n_components=percnt).fit(digits.data) # Preserving 50% of the variance by setting the number of components to 0.50 and using the unstandardized data (digits.data)
pca3_result = pca3.transform(digits.data)
# Transform the PCA reduced data back to its original space.
inversed_pca = pca3.inverse_transform(pca3_result)  # Use the inverse of the transform to reconstruct the reduced digits

if percnt < 1:
    percnt_var = round(int(percnt*100))
    print ('{:0d}% of the variance is contained within {} principal components'.format(percnt_var, pca3.n_components_))

# Figure of original images, as assigned to the variable called inversed_lst
inversed_lst = range(0, 10)
if 0:
    fig = plt.figure(figsize=(10,2))
    plt_index = 0
    for i in inversed_lst:
        plt_index = plt_index + 1
        ax = fig.add_subplot(1, 10, plt_index)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    plt.show() 
    # plt.savefig('digits0_9_original.png', dpi=150) 

# Figure of images that have undergone PCA reduction, as assigned to the variable called inversed_lst. The inverse transform is plotted here
if 0:
    fig = plt.figure(figsize=(10,2))
    plt_index = 0
    for i in inversed_lst:
        plt_index = plt_index + 1
        ax = fig.add_subplot(1, 10, plt_index)
        ax.imshow(inversed_pca[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    plt.show() 
    # plt.savefig('digits0_9_pca.png', dpi=150) 

# Figure of the above two plots combined into one in order to compare the original images vs corresponding PCA reduced images


 # One can estimate how many components are needed to describe the data by assesing the 
 # cumulative explained variance ratio as a function of the number of components
if 0:
    pca4 = PCA().fit(digits.data)
    plt.plot(np.cumsum(pca4.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Number of components vs. retained variance')
    plt.show() # About 90% of the variance is retained by using 20 components, but 75% is retained with 10 components
    # plt.savefig('n_comp_var.png', dpi=150)

# #############################################################################
# t-SNE reduced data

n_iter = 1000
n_perplexity = 40

t_sne = TSNE(n_components=2, perplexity=40, n_iter=n_iter)
if 1:
    tsne_result = t_sne.fit_transform(data) # This is t-SNE reduced data
else:
    tsne_result = t_sne.fit_transform(pca_result_for_t_sne) # This is pca reduced data that is now reduced via t-SNE 

print ('t-SNE reduced shape: ', tsne_result.shape)

# #############################################################################
# Truncated SVD reduced data

svd = TruncatedSVD(n_components=2, n_iter=10)
svd_result = svd.fit_transform(data)  

# #############################################################################
# Isomap reduced data

iso = Isomap(n_components=2)
iso_result = iso.fit_transform(data)

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

# In this case the seeding of the centers is deterministic, thus the kmeans algorithm is run only once (n_init=1)
pca = PCA(n_components=n_digits).fit(data)
kmeans_metrics(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
              name="PCA-based",
              data=data)
# #############################################################################
# K-means clustering
# assumption that clusters fall in convex globular clusters 

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
k_trunc_SVD = k_means_reduced(svd_result, 'k-means++', n_clusters, n_init)
# K-means clustering on isomap reduced data
k_iso = k_means_reduced(iso_result, 'k-means++', n_clusters, n_init)

# #####################
# Visualize the results of K-means clustering

# Figure of PCA vs t-SNE reduced data and K-means clustering on both
if 0: 
    fig, axarr = plt.subplots(2, 2, figsize=(12,8)) 

    # Plot of PCA reduced data 
    axarr[0, 0].scatter(pca_result[:,0],pca_result[:,1],c='k')
    axarr[0, 0].set_title('PCA reduced data')
    
    # Plot of t-SNE reduced data 
    axarr[1, 0].scatter(tsne_result[:,0], tsne_result[:,1],c='k')
    axarr[1, 0].set_title('t-SNE reduced data')

    # Plot of K-means clustering on PCA reduced data
    cluster_ax = axarr[0, 1]
    cluster_ax.scatter(pca_result[:,0],pca_result[:,1],c=k_pca.labels_)
    centroids = k_pca.cluster_centers_
    centroid_x = cluster_ax.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=100, linewidths=5, color='k', zorder=10)
    cluster_ax.set_title('K-means clustering on PCA reduced data')

    # Plot of K-means clustering on t-SNE reduced data
    cluster_ax2 = axarr[1, 1]
    cluster_ax2.scatter(tsne_result[:,0],tsne_result[:,1],c=k_t_sne.labels_)
    centroids = k_t_sne.cluster_centers_
    centroid_x = cluster_ax2.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', edgecolors='k', 
                c=np.arange(n_clusters), zorder=10) # If you want the cluster center marker as an 'x', use: marker='x', s=100, linewidths=5, color='k', zorder=10
    cluster_ax2.set_title('K-means clustering on t-SNE reduced data')
    fig.suptitle('K-means clustering on the handwritten digits dataset')
    plt.show()

# Figure of k-means clustering on pca reduced data
if 1: 
    labels = k_pca.labels_
    color = labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  

    # Plot of PCA reduced data 
    ax1 = axarr[0]
    ax1.scatter(pca_result[:,0], pca_result[:,1], c=digits.target, marker='.')
    ax1.set_title('pca reduced data')
    # ax1.set_xlabel('Component 1')
    # ax1.set_ylabel('Component 2')

    # Plot of K-means clustering on PCA reduced data
    ax2 = axarr[1]
    ax2.scatter(pca_result[:,0], pca_result[:,1], c=color, marker='.')
    centroids = k_pca.cluster_centers_
    centroid_o = ax2.scatter(centroids[:, 0], centroids[:, 1],
            marker='o', edgecolors='k', 
            c=np.arange(n_clusters), zorder=10)
    ax2.set_title('K-means clustering on pca reduced data')
    plt.tight_layout()
    plt.show()

# Figure of K-means clustering on t-SNE reduced data
if 1: 
    labels = k_t_sne.labels_
    color = labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  

    # Plot of t-SNE reduced data 
    ax1 = axarr[0]
    ax1.scatter(tsne_result[:,0], tsne_result[:,1], c='k', marker='.')
    ax1.set_title('t-SNE reduced data')

    # Plot of K-means clustering on t-SNE reduced data
    ax2 = axarr[1]
    ax2.scatter(tsne_result[:,0], tsne_result[:,1], c=color, marker='.')
    centroids = k_t_sne.cluster_centers_
    centroid_o = ax2.scatter(centroids[:, 0], centroids[:, 1],
            marker='o', edgecolors='k', 
            c=np.arange(n_clusters), zorder=10)
    ax2.set_title('K-means clustering on t-SNE reduced data')
    plt.tight_layout()
    plt.show()
    # plt.savefig('k_means_clusters{}_t_sne{}_per{}.png'.format(n_clusters, n_iter, n_perplexity), dpi=150)

# Figure of K-means clustering on Truncated SVD reduced data
if 1:
    labels = k_trunc_SVD.labels_
    color = labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  

    # Plot of truncated SVD reduced data 
    ax1 = axarr[0]
    ax1.scatter(svd_result[:,0], svd_result[:,1], c='k', marker='.')
    ax1.set_title('truncated SVD reduced data')

    # Plot of K-means clustering on truncated SVD reduced data
    ax2 = axarr[1]
    ax2.scatter(svd_result[:,0], svd_result[:,1], c=color, marker='.')
    ax2.set_title('K-means clustering on truncated SVD reduced data')
    plt.tight_layout()
    plt.show()

# Figure of K-means clustering on isomap reduced data
if 1:
    labels = k_iso.labels_
    color = labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  

    # Plot of isomap reduced data 
    ax1 = axarr[0]
    ax1.scatter(iso_result[:,0], iso_result[:,1], c='k', marker='.')
    ax1.set_title('isomap reduced data')

    # Plot of K-means clustering on isomap reduced data
    ax2 = axarr[1]
    ax2.scatter(iso_result[:,0], iso_result[:,1], c=color, marker='.')
    ax2.set_title('K-means clustering on isomap reduced data')
    plt.tight_layout()
    plt.show()

# Plot of isomap reduced data with a colorbar 
if 1:
    target = digits.target[::]
    plt.scatter(iso_result[:, 0], iso_result[:, 1], c=target, cmap=plt.cm.get_cmap('tab10', 10), marker='.') # cmap=plt.cm.get_cmap('jet', 10)
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show() 


'''# # # # 
# Trying to plot thumbnails at the locations of the projections
from matplotlib import offsetbox
def plot_components(data, model, images=None, ax=None, thumb_frac=0.5, cmap='gray'):
    ax = ax or plt.gca()
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 0], '.k')
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
            ax.add_artist(imagebox)

# Choose 1/4 of the "1" digits to project
data_iso = digits.data[digits.target == 1][::4]

fig, ax = plt.subplots(figsize=(10, 10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
plot_components(data, model, images=data.reshape(-1, 8, 8),
                ax=ax, thumb_frac=0.05, cmap='gray_r')  

# # # # 
'''

# Plot an elbow curve to select the optimal number of clusters for k-means clustering
# Fit KMeans and calculate the sum of squared errors (SSE) for each k, which is defined as the sum of the 
# squared distance between each member of the cluster and its centroid.
sse = {}
for k in range(1, 20): 
    # Initialize KMeans with k clusters and fit 
    kmeans = KMeans(n_clusters=k, random_state=0).fit(tsne_result)  
    # Assign sum of squared errors to k element of the sse dictionary
    sse[k] = kmeans.inertia_ 
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
# Plot SSE values for each k stored as keys in the sse dictionary 
plt.plot(list(sse.keys()), list(sse.values()))
plt.show()

# When we make the plot, we see that the line levels off approximately after about ___ clusters, implying that 
# the addition of more clusters may not explain much more of the variance 
# To determine the bend in the knee, see https://github.com/arvkevi/kneed and https://raghavan.usc.edu/papers/kneedle-simplex11.pdf

# #########
# Visualize random digits from a select cluster

clust_num = 1 # The cluster number to select
indexes = 8 # The number of plots to generate on the figure. 
fig_size = (7,8)

def cluster_indices(clust_num, labels_array): #numpy 
    """
    This takes parameters such as the cluster number (clust_num) and the labels of each data point.
    This returns the indices of the cluster_num you provide.
    """
    return np.where(labels_array == clust_num)[0]

cluster_data = cluster_indices(clust_num, k_t_sne.labels_)
print ('Samples from cluster {}:'.format(clust_num), cluster_data)

cluster_data_random = []
for i in range(indexes):
    rand_num = np.random.choice(cluster_data)
    cluster_data_random.append(rand_num)

# Plot of random digits from a chosen cluster
if 1:
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
    # plt.savefig('handwritten_digits_from_select_cluster.png', dpi=150)

# #############################################################################
# Mean-shift clustering
# We can use the estimate_bandwidth function to estimate a good bandwidth for the data
bandwidth = round(estimate_bandwidth(data))

mean_s = MeanShift(bandwidth=bandwidth)
mean_s.fit(pca_result)

ms = MeanShift(bandwidth=bandwidth)
ms_model = ms.fit(tsne_result)
ms_labels = ms_model.labels_
ms_cluster_centers = ms_model.cluster_centers_

ms_labels_unique = np.unique(ms_labels)
ms_n_clusters = len(ms_labels_unique)

print ('The number of estimated clusters from mean-shift clustering is: {}'.format(ms_n_clusters))

# #########
# Visualize the results of Mean-shift clustering
if 1: 
    color = ms_labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  
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

# #############################################################################
# Spectral clustering
# spectral clustering can ignore sparse interconnections between arbitrarily shaped clusters of data

# sc_result = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize").fit(data)
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                           assign_labels='kmeans')                         
sc_labels = sc.fit_predict(tsne_result)

# #########
# Visualize the results of spectral clustering

if 1: 
    color = sc_labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  
    # Plot of t-SNE reduced data 
    ax1 = axarr[0]
    ax1.scatter(tsne_result[:,0], tsne_result[:,1], c='k', marker='.')
    ax1.set_title('t-SNE reduced data')

    # Plot of spectral clustering on t-SNE reduced data
    ax2 = axarr[1]
    ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=color, marker='.')
    ax2.set_title('Spectral clustering on t-SNE reduced data')
    plt.tight_layout()
    plt.show()

# #############################################################################
# DBSCAN clustering
# For some more information, see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# See: http://abhijitannaldas.com/ml/dbscan-clustering-in-machine-learning.html
# See: https://www.dummies.com/programming/big-data/data-science/how-to-create-an-unsupervised-learning-model-with-dbscan/
# See: https://towardsdatascience.com/dbscan-clustering-for-data-shapes-k-means-cant-handle-well-in-python-6be89af4e6ea

db = DBSCAN(eps=3, min_samples=2)
db_model = db.fit(tsne_result)

# #########
# Visualize the results of DBSCAN clustering

if 1: 
    db_labels = db_model.labels_
    color = db_labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  
    # Plot of t-SNE reduced data 
    ax1 = axarr[0]
    ax1.scatter(tsne_result[:,0], tsne_result[:,1], c='k', marker='.')
    ax1.set_title('t-SNE reduced data')

    # Plot of DBSCAN clustering on t-SNE reduced data
    ax2 = axarr[1]
    ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=color, marker='.')
    ax2.set_title('DBSCAN clustering on t-SNE reduced data')
    plt.tight_layout()
    plt.show()

# #############################################################################
# Affinity propogation clustering

ap = AffinityPropagation().fit(tsne_result)
cluster_centers = ap.cluster_centers_

# #########
# Visualize the results of Affinity propogation clustering on t-SNE reduced data
if 0: 
    labels = ap.labels_
    color = labels
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  
    # Plot of t-SNE reduced data 
    ax1 = axarr[0]
    ax1.scatter(tsne_result[:,0], tsne_result[:,1], c='k', marker='.')
    ax1.set_title('t-SNE reduced data')

    # Plot of Affinity propogation clustering on t-SNE reduced data
    ax2 = axarr[1]
    ax2.scatter(tsne_result[:,0], tsne_result[:,1], c=color, marker='.')
    centroids = cluster_centers
    centroid_x = ax2.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=100, linewidths=5,
                c='k', zorder=10)
    ax2.set_title('Affinity propogation clustering on t-SNE reduced data')
    plt.tight_layout()
    plt.show()

