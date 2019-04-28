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
# Visualize images from the dataset 
# Note: to display a figure, type 1 instead of 0 in the corresponding if statement in the relevant figures below

# Plot of a single image from the dataset. 
sample_n = 138 # Input number for which sample to visualize
if 0: 
    plt.figure(0, figsize=(3, 3))
    plt.imshow(digits.images[sample_n], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    # print ('Digit number {} is sample number {}'.format(labels[sample_n], sample_n)) # Print digit number and sample number
    plt.show()
    # plt.savefig('example_digit_{}.png'.format(labels[sample_n]), dpi=150) 

'''# Plot of ten example digits (from 0 to 9)
def plot_ten_images(image_data, data, shape=None):
    """
    Plot ten images along one row
    image_data: the image data, for example digits.images
    data: the data, for example, digits.data
    shape: Input the size of the image (separated by a comma), for example, if the image is 8x8, you would input as a parameter: 8, 8
    """
    fig = plt.figure(figsize=(10,5))
    plt_index = 0
    pca3 = PCA(n_components=0.50).fit(digits.data) 
    pca3_result = pca3.transform(digits.data)
    # Transform the PCA reduced data back to its original space. This will help visualize how the image data appears after PCA reduction
    pca_inversed = pca3.inverse_transform(pca3_result)  
    for i in range(10):
        plt_index = plt_index + 1
        ax = fig.add_subplot(2, 5, plt_index)
        if shape:
            ax.imshow(pca_inversed[i].reshape(shape), cmap=plt.cm.gray_r, interpolation='nearest')
        else:
            ax.imshow(image_data[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    plt.show()  
    # plt.savefig('digits_10example.png', dpi=150) 

plot_ten_digits(digits.images, digits.data, inversed=True, shape=8, 8)'''

# Plot of four examples from each of the 10 digits (0 through 9)
sample_lst = [] # A list of lists. Each list within contains four sample numbers of a single digit, and it goes from digit 0 (the first list within the list of lists) to digit 9 (the last list)
for i in range(10):
    sample_nums = np.where(labels == i)[0][0:4] # Indexes (0:4) into an array of sample numbers that are a specified digit (the digit is written as i in the loop)
    sample_nums = sample_nums.tolist() # Convert the array called sample_nums to a list with the same items
    sample_lst.append(sample_nums) 
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

# ###########################################################################################
# PCA-reduced data

# The first two principal components are generated below (on the standardized data) 
pca = PCA(n_components=2) # The original 64 dimensions are reduced to 2
pca_result = pca.fit_transform(data) # Now we have the reduced feature set
# print ('PCA reduced shape: ', pca_result.shape) # Number of samples (1797) in the dataset and PCA reduced number of features
# print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)) # Percentage of variance explained by each dimension. Here, the n_components=2, as designated in the pca above, so two values are expected
# print ('The first two components account for {:.0f}% of the variation in the entire dataset'.format((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100))

# Principal components are assigned to the percnt variable below and PCA is done on the non-standardized data
percnt = 0.50 # Input the percentage of variance (e.g., 0.50 would preserve 50% of the variance)
pca3 = PCA(n_components=percnt).fit(digits.data) 
pca3_result = pca3.transform(digits.data)
# Transform the PCA reduced data back to its original space. This will help visualize how the image data appears after PCA reduction
pca_inversed = pca3.inverse_transform(pca3_result)  # Use the inverse of the transform to reconstruct the reduced digits

if percnt < 1:
    percnt_var = round(int(percnt*100))
    # print ('{:0d}% of the variance is contained within {} principal components'.format(percnt_var, pca3.n_components_))

# One can estimate how many components are needed to describe the data by assessing the 
# cumulative explained variance ratio as a function of the number of components, see the plot below
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
if 1:
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=digits.target, cmap=plt.cm.get_cmap('tab10', 10), marker='.') 
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show() 
    # plt.savefig('pca_standardized_digits.png', dpi=150) 

# Figure of original images, as assigned to the variable called inversed_lst
inversed_lst = range(0, 10)
if 1:
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
if 1:
    fig = plt.figure(figsize=(10,2))
    plt_index = 0
    for i in inversed_lst:
        plt_index = plt_index + 1
        ax = fig.add_subplot(1, 10, plt_index)
        ax.imshow(pca_inversed[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    # plt.show() 
    plt.savefig('digits0_9_pca.png', dpi=150) 

# Figure of the above two plots combined into one in order to compare the original images vs corresponding PCA reduced images
'''
show_images_figs = False
# Figure of original images, as assigned to the variable called inversed_lst
if 1:
    figures.plot_images(digits.images, save_fname='digits0_9_original.png', show=show_images_figs)
    figures.plot_images(pca_inversed, reshape=(8, 8), save_fname='digits0_9_pca_inversed.png.png', show=show_images_figs)
'''

# #############################################################################
# t-SNE reduced data

n_iter = 1000
n_perplexity = 40

t_sne = TSNE(n_components=2, perplexity=n_perplexity, n_iter=n_iter)
tsne_result = t_sne.fit_transform(data) # This is t-SNE reduced data

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

# In this case the seeding of the centers is deterministic, thus the kmeans algorithm is run only once (i.e., n_init=1)
pca = PCA(n_components=n_digits).fit(data)
kmeans_metrics(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
              name="PCA-based",
              data=data)

# #############################################################################
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

def plot_dim_red_clust(dim_red, cluster_type, dim_red_name=None, cluster_type_name=None, centroids=None):
    """
    Visualize the results of clustering on dimesnionality reduced data.
    Plots the dimensionality reduced data vs. the clustered dimensionality reduced data
    Parameters:
        dim_red: dimensionality reduction method
        cluster_type: the type of clustering algorithm, for example, from the Scikit-learn library, one can use the k-means implementatation as sklearn.cluster.KMeans()
        dim_red_name: name of dimesnionality reduction to be in the title of the corresponding plot
        cluster_type_name: name of clustering method to be in the title of the corresponding plot
        centroids: plots the coordinates of cluster centers 
    """
    color = cluster_type.labels_
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  
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
                c='k', zorder=10)
    if cluster_type_name:
        ax2.set_title('{} clustering on {} reduced data'.format(cluster_type_name, dim_red_name))
    plt.tight_layout()
    plt.show()

# Plot of k-means clustering on t-SNE reduced data
plot_dim_red_clust(tsne_result, k_t_sne, 't-SNE', 'K-means', centroids=True)
# Plot of k-means clustering on pca reduced data
plot_dim_red_clust(pca_result, k_pca, 'PCA', 'K-means')
# Plot of K-means clustering on isomap reduced data
plot_dim_red_clust(iso_result, k_iso, 'isomap', 'K-means')
# Plot of K-means clustering on truncated SVD reduced data
plot_dim_red_clust(svd_result, k_SVD, 'truncated SVD', 'K-means')

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

'''

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
print ('Samples from cluster {}:'.format(clust_num), cluster_data)
print ('Digits from cluster {}:'.format(clust_num), labels[cluster_data])

print (("where:"), np.where(k_t_sne.labels_ == 6)[0]) # This is telling us the sample number where the digit 5 is

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

# cluster_data should be equal to one specific digit (within dig_lst)
cl_dig = cluster_data[0] # The example digit, in which the whole cluster should be
dig_lst = np.where(labels == cl_dig)[0] # A list of the sample numbers of the digit (variable called cl_dig)

'''for i in range(10):
    clust_n == i
    cluster_data = cluster_indices(clust_n, k_t_sne.labels_) '''

'''
def digit_matches(cluster_data, data):
    """
    This checks whether or not the ________
    It returns whether or not these two match. 
    """
    cl_dig = cluster_data[0]
    dig_lst = np.where(digits.target == cl_dig)[0]
    matches = dig_lst == cluster_data
    if matches == True:
        print ('Match!')
    else:
        print ('No match')

dig_match_ans = digit_matches(cluster_data, data)
print (dig_match_ans)
'''

'''def digit_matches_answer(dig_match_ans):
    if dig_match_ans == True:
        print ('The cluster digits are all the same digit', cluster_data)
    else: 
        print ('The cluster digits are not all the same digit', np.where(dig_lst != cluster_data)[0])
digit_matches_answer(dig_match_ans)'''

print ('label for cluster_data 0: ', labels[cluster_data[0]])

dig_num = []
for i in cluster_data:
    dig_num.append(labels[i])
print ('Digit numbers from cluster {}'.format(clust_num), dig_num)
print ('Example digit from cluster{}'.format(clust_num), cluster_data[0]) # The example digit, in which the whole cluster should be
cl_dig_no_match = []
for i in dig_num:
    n = i + 1
    if i == n:
        continue
    else:
        cl_dig_no_match.append(n)
# print (cl_dig_no_match) 

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
