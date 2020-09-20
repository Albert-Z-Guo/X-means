import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class XMeans:
    def __init__(self, K_init=2, K_max=20, identical_spherical_normal_distributions=False, split_centroids_mannually=False, random_split=False, **KMeans_args):
        self.K_init = K_init
        self.K_max = K_max
        self.identical_spherical_normal_distributions = identical_spherical_normal_distributions
        self.split_centroids_mannually = split_centroids_mannually
        self.random_split = random_split
        self.KMeans_args = KMeans_args
        
    def centroids_subcluster(self, cluster_points, mu):
        '''subclusters' centers are picked by moving a distance proportional to the size of the parent cluster region 
        in opposite directions along a vector that can reach the furthest point from the parent cluster center or along a random vector'''
        norms = np.linalg.norm(cluster_points - mu, axis=1)
        direction = np.random.random_sample(size=len(mu)) if self.random_split else (cluster_points - mu)[norms.argmax()]
        vector = direction/np.sqrt(np.dot(direction, direction))*np.quantile(norms, 0.9)
        return np.array([mu + vector, mu - vector])
        
    def variance_identical_spherical(self, cluster_points_list, mu_list):
        '''maximum likelihood estimate of cluster variance assuming all clusters are in 'identical' spherical normal distribution'''
        R = np.sum([len(cluster_points) for cluster_points in cluster_points_list])
        K = len(mu_list)
        if R - K == 0: return 0
        return 1/(self.M*(R - K))*np.sum([np.sum(np.square(cluster_points - mu)) for (cluster_points, mu) in zip(cluster_points_list, mu_list)])

    def p(self, K):
        '''number of free parameters in a model'''
        return K*(1 + self.M) # equivalent to (K-1) + M*K + 1
            
    def BIC_identical_spherical(self, R_n_list, variance):
        '''Bayesian information criterion for a model assuming all clusters are in 'identical' spherical normal distribution;
        spherical normal distribution entails covariance matrix = variance * identity matrix in the standard multivariate normal distribution formula,
        which can in turn be simplified to the form in paper'''
        K = len(R_n_list)
        R = np.sum(R_n_list)
        l = np.sum([R_n*np.log(R_n) for R_n in R_n_list]) - R*np.log(R) - (R*self.M)/2*np.log(2*np.pi*variance) - self.M/2*(R - K) if variance != 0 else -R*np.log(R)
        return l - self.p(K)/2*np.log(R)
    
    def log_likelihood(self, R, cluster_points, mu, sigma):
        '''log likelihood of each cluster'''
        R_n = len(cluster_points)
        l = 0
        for point in cluster_points: # avoid using singular covariance matrices in logpdf computations
            l += np.log(R_n/R) + multivariate_normal.logpdf(point, mu, sigma) if np.abs(np.linalg.det(sigma)) > 1e-5 else np.log(1/R)
        return l
        
    def BIC(self, cluster_points_list, mu_list):
        '''Bayesian information criterion for a model from direct log likelihood computation using estimated means and estimated covariance matrices'''
        K = len(mu_list)
        R = np.sum([len(cluster_points) for cluster_points in cluster_points_list])
        # maximum likelihood estimate of covariance matrices; note that maximum likelihood estimate of means are just the sample means given by K-means algorithm
        sigma_list = [np.dot((cluster_points - mu).T, (cluster_points - mu))/len(cluster_points) for (cluster_points, mu) in zip(cluster_points_list, mu_list)]
        l = np.sum([self.log_likelihood(R, cluster_points, mu, sigma) for (cluster_points, mu, sigma) in zip(cluster_points_list, mu_list, sigma_list)])
        return l - self.p(K)/2*np.log(R)
    
    def fit(self, data):
        assert len(data) > 0 # data must not be empty
        data = np.array(data)
        self.R = data.shape[0]
        self.M = data.shape[1]
        K = self.K_init
        init = 'k-means++'

        while(1):            
            K_before = K
            
            # improve parameters (run K-means)
            model = KMeans(n_clusters=K, init=init, n_init=1, **self.KMeans_args).fit(data)
            labels = model.labels_
            centroids = model.cluster_centers_
            
            # improve structure            
            centroids_improved = list()
            for n in range(len(centroids)):
                points_n = data[labels == n]
                R_n = len(points_n)
                mu_n = centroids[n]
                
                # split each cluster into 2 subclusters if possible
                if R_n > 1:
                    # pick subclusters' centers
                    if self.split_centroids_mannually:
                        model_subcluster = KMeans(n_clusters=2, init=self.centroids_subcluster(points_n, mu_n), n_init=1, **self.KMeans_args).fit(points_n)
                    else:
                        model_subcluster = KMeans(n_clusters=2, **self.KMeans_args).fit(points_n)
                    labels_subcluster = model_subcluster.labels_
                    centroids_subcluster = model_subcluster.cluster_centers_
                    
                    # calculate parent model's BIC score and children model's BIC score
                    subcluster_points_list = [points_n[labels_subcluster == label] for label in range(len(centroids_subcluster))]
                    if self.identical_spherical_normal_distributions:
                        BIC_parent_model = self.BIC_identical_spherical([R_n], self.variance_identical_spherical([points_n], [mu_n]))
                        variance_spherical_identical = self.variance_identical_spherical(subcluster_points_list, centroids_subcluster)
                        BIC_children_model = self.BIC_identical_spherical([len(subcluster_points) for subcluster_points in subcluster_points_list], variance_spherical_identical)
                    else:
                        BIC_parent_model = self.BIC([points_n], [mu_n])
                        BIC_children_model = self.BIC(subcluster_points_list, centroids_subcluster)
                        
                    if BIC_children_model > BIC_parent_model:
                        K += 1
                        centroids_improved.extend(centroids_subcluster)
                    else:
                        centroids_improved.append(mu_n)
                else:
                    centroids_improved.append(mu_n)
                                        
            # update clusters' centers for next improvement iteration
            init = np.array(centroids_improved)

            # stop improvement iteration
            if K == K_before or K > self.K_max:
                break
        
        if K == self.K_init: print('\033[93mWARNING: No split made. Please check data distribution or reduce K_init if necessary.\033[0m')
        model_final = KMeans(n_clusters=K, **self.KMeans_args).fit(data)
        self.labels = model_final.labels_
        self.centroids = model_final.cluster_centers_
        self.K = K
        return self