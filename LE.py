from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import networkx as nx

import math
from scipy import spatial
import pandas as pd
class LE:
    
    def __init__(self, X:np.ndarray, dim:int, k:int = 2, eps = None, graph:str = 'k-nearest', weights:str = 'heat kernel', 
                 sigma:float = 0.1, laplacian:str = 'unnormalized', opt_eps_jumps:float = 1.5):
        """
        LE object
        Parameters
        ----------
        
        X: nxd matrix
        
        dim: number of coordinates
        
        k: number of neighbours. Only used if graph = 'k-nearest'
        
        eps: epsilon hyperparameter. Only used if graph = 'eps'. 
        If is set to None, then epsilon is computed to be the 
        minimum one which guarantees G to be connected
        
        graph: if set to 'k-nearest', two points are neighbours 
        if one is the k nearest point of the other. 
        If set to 'eps', two points are neighbours if their 
        distance is less than epsilon
        
        weights: if set to 'heat kernel', the similarity between 
        two points is computed using the heat kernel approach.
        If set to 'simple', the weight between two points is 1
        if they are connected and 0 otherwise. If set to 'rbf'
        the similarity between two points is computed using the 
        gaussian kernel approach.
        
        sigma: coefficient for gaussian kernel or heat kernel
        
        laplacian: if set to 'unnormalized', eigenvectors are 
        obtained by solving the generalized eigenvalue problem 
        Ly = λDy where L is the unnormalized laplacian matrix.
        If set to 'random', eigenvectors are obtained by decomposing
        the Random Walk Normalized Laplacian matrix. If set to 
        'symmetrized', eigenvectors are obtained by decomposing
        the Symmetrized Normalized Laplacian
        
        opt_eps_jumps: increasing factor for epsilon
        """
        
        self.X = X
        self.dim = dim
        self.k = k
        self.eps = eps
        if graph not in ['k-nearest', 'eps']:
            raise ValueError("graph is expected to be a graph name; 'eps' or 'k-nearest', got {} instead".format(graph))
        self.graph = graph
        if weights not in ['simple', 'heat kernel', 'rbf']:
            raise ValueError("weights is expected to be a weight name; 'simple' or 'heat kernel', got {} instead".format(weights))
        self.weights = weights
        self.sigma = sigma
        self.n = self.X.shape[0]
        if laplacian not in ['unnormalized', 'random', 'symmetrized']:
            raise ValueError("laplacian is expected to be a laplacian name; 'unnormalized', 'random' or 'symmetrized', got {} instead".format(laplacian))
        self.laplacian = laplacian
        self.opt_eps_jumps = opt_eps_jumps
        if self.eps is None and self.graph == 'eps':
            self.__optimum_epsilon()
    
    def __optimum_epsilon(self):
        """
        Compute epsilon
        
        To chose the minimum epsilon which guarantees G to be 
        connected, first, epsilon is set to be equal to the distance 
        from observation i = 0 to its nearest neighbour. Then
        we check if the Graph is connected, if it's not, epsilon
        is increased and the process is repeated until the Graph
        is connected
        """
        dist_matrix = pairwise_distances(self.X)
        self.eps = min(dist_matrix[0,1:])
        con = False
        while not con:
            self.eps = self.opt_eps_jumps * self.eps
            self.__construct_nearest_graph()
            con = self.cc == 1
            print('[INFO] Epsilon: {}'.format(self.eps))
        self.eps = np.round(self.eps, 3)
    
    def __heat_kernel(self, dist):
        """
        k(x, y) = exp(- ||x-y|| / sigma )
        """
        return np.exp(- dist/self.sigma)
    
    def __rbf(self, dist):
        """
        k(x, y) = exp(- (1/2*sigma^2) * ||x-y||^2)
        """
        return np.exp(- dist**2/ (2* (self.sigma**2) ) )
    
    def __simple(self, *args):
        return 1
    
    def __construct_nearest_graph(self):
        """
        Compute weighted graph G
        """
        similarities_dic = {'heat kernel': self.__heat_kernel,
                            'simple':self.__simple,
                            'rbf':self.__rbf}
        
        dist_matrix = pairwise_distances(self.X)
        if self.graph == 'k-nearest':
            nn_matrix = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k + 1]
        elif self.graph == 'eps':
            nn_matrix = np.array([ [index for index, d in enumerate(dist_matrix[i,:]) if d < self.eps and index != i] for i in range(self.n) ], dtype=object)
        # Weight matrix
        self._W = []
        for i in range(self.n):
            w_aux = np.zeros((1, self.n))
            similarities = np.array([ similarities_dic[self.weights](dist_matrix[i,v]) for v in nn_matrix[i]] )
            np.put(w_aux, nn_matrix[i], similarities)
            self._W.append(w_aux[0])
        self._W = np.array(self._W)
        # D matrix
        self._D = np.diag(self._W.sum(axis=1))
        # Check for connectivity
        self._G = self._W.copy() # Adjacency matrix
        self._G[self._G > 0] = 1
        self.G = nx.from_numpy_matrix(self._G)
        self.cc = nx.number_connected_components(self.G) # Multiplicity of lambda = 0
        if self.cc != 1:
            warnings.warn("Graph is not fully connected, Laplacian Eigenmaps may not work as expected")
            
    def __compute_unnormalized_laplacian(self):
        self.__construct_nearest_graph()
        self._L = self._D - self._W
        return self._L
    
    def __compute_normalized_random_laplacian(self):
        self.__construct_nearest_graph()
        self._Lr = np.eye(*self._W.shape) - (np.diag(1/self._D.diagonal())@self._W)
        return self._Lr
    
    def __compute_normalized_symmetrized_laplacian(self):
        self.__construct_nearest_graph()
        self.__compute_unnormalized_laplacian()
        d_tilde = np.diag(1/np.sqrt(self._D.diagonal()))
        self._Ls = d_tilde @ ( self._L @ d_tilde )
        return self._Ls
    
    def transform(self, eig=False):
        """
        Compute embedding
        """
        
        m_options = {
            'unnormalized':self.__compute_unnormalized_laplacian,
            'random':self.__compute_normalized_random_laplacian,
            'symmetrized':self.__compute_normalized_symmetrized_laplacian
        }
        
        L = m_options[self.laplacian]()
        
        if self.laplacian == 'unnormalized':
            eigval, eigvec = eigh(L, self._D) # Generalized eigenvalue problem
        else:
            eigval, eigvec = np.linalg.eig(L)
            
        order = np.argsort(eigval)
        self.Y = eigvec[:, order[self.cc:self.cc+self.dim + 1]]

        if eig == True:
            return eigval, eigvec
        else:
            return self.Y
    
    def plot_embedding_2d(self, colors='b', grid = True, dim_1 = 1, dim_2 = 2, cmap = None, size = (15, 10)):
        if self.dim < 2 and dim_2 <= self.dim and dim_1 <= self.dim:
            raise ValueError("There's not enough coordinates")
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        plt.axhline(c = 'black', alpha = 0.2)
        plt.axvline(c = 'black', alpha = 0.2)
        if cmap is None:
            plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors)
        else:    
            plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors, cmap=cmap)
        plt.grid(grid)
        if self.graph == 'k-nearest':
            title = 'LE with k = {} and weights = {}'.format(self.k, self.weights)
        else:
            title = 'LE with $\epsilon$ = {} and weights = {}'.format(self.eps, self.weights)
        plt.title(title)
        plt.xlabel('Coordinate {}'.format(dim_1))
        plt.ylabel('Coordinate {}'.format(dim_2))
        plt.savefig('plot2D.png')
    
    def plot_embedding_3d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, dim_3 = 3, cmap = None, size = (15, 10)):
        if self.dim < 3 and dim_2 <= self.dim and dim_1 <= self.dim and dim_3 <= self.dim:
            raise ValueError("There's not enough coordinates")
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection="3d")
        if cmap is None:
            ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors)
        ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors, cmap = cmap)
        plt.grid(grid)
        ax.axis('on')
        if self.graph == 'k-nearest':
            title = 'LE with k = {} and weights = {}'.format(self.k, self.weights)
        else:
            title = 'LE with $\epsilon$ = {} and weights = {}'.format(self.eps, self.weights)
        plt.title(title)
        ax.set_xlabel('Coordinate {}'.format(dim_1))
        ax.set_ylabel('Coordinate {}'.format(dim_2))
        ax.set_zlabel('Coordinate {}'.format(dim_3))
        plt.show()

    def scalingParameter(self):
        '''
        Determine scaling parameter m of scale-free network.
        '''
        k = []
        Pk = []

        for node in list(self.G.nodes()):
            degree = self.G.degree(nbunch=node)
            try:        
                pos = k.index(degree)
            except ValueError as e:     
                k.append(degree)
                Pk.append(1)
            else:
                Pk[pos] += 1

        # get a double log representation
        logk = []
        logPk = []
        for i in range(len(k)):
            logk.append(math.log10(k[i]))
            logPk.append(math.log10(Pk[i]))

        order = np.argsort(logk)
        logk_array = np.array(logk)[order]
        logPk_array = np.array(logPk)[order]
        #plt.plot(logk_array, logPk_array, ".")
        m, c = np.polyfit(logk_array, logPk_array, 1)
        #plt.plot(logk_array, m*logk_array + c, "-")
        print('Scaling Parameter', m)
        return m

    def sortedKeys(self):
        '''
        Nodes sorted decreasingly by degree.
        '''
        dictNodes = self.G.degree()
        sortedNodes = {k: v for k, v in sorted(dict(dictNodes).items(), key=lambda item: item[1])}
        sortedKeys = list(reversed(list(sortedNodes.keys())))
        return sortedKeys
    
    def pol2cart(self, rho, phi):
        x = [a*b for a,b in zip(rho, np.cos(phi))]
        y = [a*b for a,b in zip(rho,np.sin(phi))]
        return x, y

    def hyperEmbedding(self):
        '''
        Convert Cartesian to polar coordinates.
        '''
        # angular position
        thetas = 2*np.arctan(self.Y[:,0]/abs(self.Y[:,1])) 

        # radial distance from origin (hierarchy based on node degree)
        gamma = self.scalingParameter()
        beta = 1/(gamma-1)
        N = self.G.number_of_nodes()
        radii = np.zeros(N)
        n = 1
        sortedKeys = self.sortedKeys()
        for i in sortedKeys:
            radii[i-1] = 2*beta*np.log(n) + 2*(1-beta)*np.log(N)
            n += 1

        return thetas, radii
    
    def plotHyper(self, colors='b', annotate=False):
        print('Plotting...')
        self.colors = colors #colors
        thetas, radii = self.hyperEmbedding()
        x, y = self.pol2cart(radii,thetas)

        pd.DataFrame(zip(x,y)).to_csv("hyperCoord.csv", header=None, index=None) 

        fig = plt.figure() #Here is your error

        ax1 = fig.add_subplot(1,2,1)
        ax1.scatter(x, y, c=self.colors, cmap='jet') #colors
        if annotate==True:
            for i in range(len(x)):
                ax1.annotate(i, (x[i], y[i]))
        
        ax2 = fig.add_subplot(1,2,2,projection='polar')
        #colors = thetas
        ax2.scatter(thetas, radii, c=self.colors, cmap='jet', alpha=0.75)
        ax2.set_yticklabels([])
        if annotate==True:
            for i in range(len(radii)):
                ax2.annotate(i, xy=(thetas[i], radii[i]))

        plt.savefig('plotHyper.png')

class Align:

    def __init__(self, A, B, labels, mu=1000, graph = 'eps', weights = 'heat kernel', sigma = 5, laplacian = 'symmetrized'):
        self.A = A
        self.B = B
        self.labels = labels

        self.graph = graph
        self.weights = weights
        self.sigma = sigma
        self.laplacian = laplacian
        leX = LE(A, dim = 1, graph = self.graph, weights = self.weights, sigma = self.sigma, laplacian = self.laplacian) 
        leX_eigval, leX_eigvec = leX.transform(eig=True)
        leY = LE(B, dim = 1, graph = self.graph, weights = self.weights, sigma = self.sigma, laplacian = self.laplacian) 
        leY_eigval, leY_eigvec = leY.transform(eig=True)
        self.Gx = leX.G
        self.Gy = leY.G

        self.Lx = leX_eigvec
        self.Ly = leY_eigvec
        self.Ux = self.U_x(self.Gx,self.labels,mu)
        self.Uy = self.U_x(self.Gy,self.labels,mu)
        self.Uxy = self.U_xy(self.Gx,self.Gy,self.labels,mu)
        self.Uyx = self.U_xy(self.Gy,self.Gx,self.labels,mu)

    def U_x(self,g,l,mu):
        u = [v for v in list(g.nodes()) if v not in l]
        Ux = np.zeros((len(list(g.nodes())),len(list(g.nodes()))))
        for i in range(len(u)):
            for j in range(len(l)):
                if (list(g.nodes)[i] == list(g.nodes)[j]) & (list(g.nodes)[i] in l):
                    Ux[i,j] = mu
        return Ux

    def U_xy(self,gx,gy,l,mu):
        u = [v for v in list(gx.nodes()) if v not in l]
        Uxy = np.zeros((len(list(gx.nodes())),len(list(gy.nodes()))))
        for i in range(len(list(gx.nodes()))):
            for j in range(len(list(gy.nodes()))):
                if (list(gx.nodes)[i] == list(gy.nodes)[j]) & (list(gx.nodes)[i] in l):
                    Uxy[i,j] = mu
        return Uxy

    def jointLaplacian(self):
        '''
        Joint Laplacian.
        '''
        Lz = np.concatenate((np.concatenate((self.Lx+self.Ux,-self.Uxy),axis=1),
                            np.concatenate((-self.Uyx,self.Ly+self.Uy),axis=1)),axis=0)
        return Lz

    def embedJointLaplacian(self):
        Lz = self.jointLaplacian()
        eigval, eigvec = np.linalg.eig(Lz)
        
        order = np.argsort(eigval)
        orderEigvec = eigvec[:, order]
        Y = orderEigvec[:,1:len(order)]

        return Y

    def findCorrespondences(self, hyper=False, dims=None, gamma=5):
        '''
        dims: dimensionality of the low-dimensional embedding of Lz
        '''
        Lz = self.jointLaplacian()

        if hyper==True:
            #print('Hyperbolic embedding')
            Y = self.hyperEmbedding(gamma=gamma)
            norm=1
        else:
            #print('Euclidean embedding')
            Y = self.embedJointLaplacian()
            norm=2

        if dims==None:
            dims=len(Y[0,:]-1)

        Y1 = Y[0:len(self.A[:,0]),0:dims]
        Y2 = Y[len(self.A[:,0]):,0:dims]

        treeY1 = spatial.KDTree(Y1, metric='minkowski',p=norm)
        pairs=[]
        for i in range(len(Y2[:,0])):
            pairs.append([i, treeY1.query(Y2[i,:])[1]]) # Such a query takes a vector and returns the closest neighbor in Y1 for it
        
        count=0
        for i in range(len(np.array(pairs)[:,0])):
	        if np.array(pairs)[i,0] == np.array(pairs)[i,1]:
		        count += 1
        #print(count,'/',len(np.array(pairs)[:,0]),' correct.')
        percentage = count/len(np.array(pairs)[:,0])
        return  percentage #pairs
    
    def hyperEmbedding(self,gamma=5):
        '''
        Convert Cartesian to polar coordinates.
        '''
        Lz = self.jointLaplacian()
        Y = self.embedJointLaplacian()
        # angular position
        #thetas = 2*np.arctan(Y[:,0]/abs(Y[:,1])) 
        thetas = np.arctan(Y[:,0]/Y[:,1]) 

        # radial distance from origin (hierarchy based on node degree)
        gamma = 5 #TO BE IMPLEMENTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! self.scalingParameter()
        beta = 1/(gamma-1)
        N = len(Y[:,0])
        radii = np.zeros(N)
        n = 1
        sortedKeys = self.sortedKeys(Lz)
        for i in sortedKeys:
            radii[i-1] = 2*beta*np.log(n) + 2*(1-beta)*np.log(N)
            n += 1

        x, y = self.pol2cart(radii,thetas)
        hyperY = np.array([list(a) for a in zip(x,y)])
        return hyperY

    def pol2cart(self, rho, phi):
        x = [a*b for a,b in zip(rho, np.cos(phi))]
        y = [a*b for a,b in zip(rho,np.sin(phi))]
        return x, y

    def sortedKeys(self,Lz):
        '''
        Nodes sorted decreasingly by degree.
        '''
        diag = np.diagonal(Lz)
        dictNodes = {k:v for k,v in enumerate(diag)}
        sortedNodes = {k: v for k, v in sorted(dict(dictNodes).items(), key=lambda item: item[1])}
        sortedKeys = list(reversed(list(sortedNodes.keys())))
        return sortedKeys