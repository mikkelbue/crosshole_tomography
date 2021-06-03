import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

class TravelTime:
    def __init__(self, mesh, sensor_depths, slowness_model):
        
        # internalise the mesh parameters and compute the aspect.
        self.mesh = mesh
        self.left_boundary, self.right_boundary, self.bottom_boundary, self.top_boundary = self.mesh['extent']
        self.nx, self.ny = self.mesh['resolution']
        self.aspect = (self.top_boundary - self.bottom_boundary)/(self.right_boundary - self.left_boundary)
        
        # internalise sensor depths and get the number of sensor depths.
        self.sensor_depths = sensor_depths
        self.n_sensor_depths = self.sensor_depths.shape[0]
        
        # compute the grid size.
        self.dx = (self.right_boundary - self.left_boundary)/self.nx
        self.dy = (self.top_boundary - self.bottom_boundary)/self.ny
        
        # set up the grid.
        self.x_midpoints = np.linspace(self.left_boundary+self.dx/2, self.right_boundary-self.dx/2, self.nx)
        self.y_midpoints = np.linspace(self.bottom_boundary+self.dy/2, self.top_boundary-self.dy/2, self.ny)
        self.grid_midpoints = np.meshgrid(self.x_midpoints, self.y_midpoints)
        self.midpoints = np.vstack((self.grid_midpoints[0].flatten(), self.grid_midpoints[1].flatten())).T
        self.n_cells = self.midpoints.shape[0]
        
        # set up the sensors.
        self.left_sensors = np.vstack((self.left_boundary*np.ones(self.n_sensor_depths), self.sensor_depths)).T
        self.right_sensors = np.vstack((self.right_boundary*np.ones(self.n_sensor_depths), self.sensor_depths)).T
        
        # set up the slowness model.
        self.slowness_model = slowness_model
        self.slowness = self.slowness_model['kernel'](self.midpoints, self.slowness_model['lambda'])
        self.slowness.compute_eigenpairs(self.slowness_model['mkl'])
        
    def assemble(self):
        
        # create an empty matrix of the right dimensions.
        self.D = np.zeros((self.n_sensor_depths**2, self.n_cells))
        
        # iterate through all the possible rays.
        for i in range(self.n_sensor_depths**2):   
            
            # get the ray equation and its inverse (shortest path)
            ray, ray_inverse = self.get_ray(self.left_sensors[int(i/self.n_sensor_depths),:], 
                                            self.right_sensors[i%self.n_sensor_depths,:], 
                                            get_inverse=True)
            
            # iterate through all the cells to see if the ray intersects.
            for j in range(self.n_cells):
                midpoint = self.midpoints[j, :]
                
                # get the current cell vertices.
                cell_vertices = np.array([[midpoint[0]-self.dx/2, midpoint[1]-self.dy/2],
                                          [midpoint[0]+self.dx/2, midpoint[1]-self.dy/2],
                                          [midpoint[0]+self.dx/2, midpoint[1]+self.dy/2],
                                          [midpoint[0]-self.dx/2, midpoint[1]+self.dy/2]])
                
                # insert vertices in ray equation.
                ray_to_vertice = np.zeros(4)
                for k in range(4):
                    ray_to_vertice[k] = np.round(cell_vertices[k,1] - ray(cell_vertices[k,0]), decimals = 9)
                    
                # check if the line intersects with the cell.
                if ray_to_vertice[0]*ray_to_vertice[2] <= 0 or ray_to_vertice[1]*ray_to_vertice[3] <= 0:
                    
                    # if one corner has different sign than three others.
                    if np.prod(ray_to_vertice) < 0:
                        
                        # if the odd one out is negative, choose that as v_star
                        if ray_to_vertice[ray_to_vertice < 0].shape[0] == 1:
                            v_star = cell_vertices[np.argmin(ray_to_vertice), :]
                        
                        # if the odd one out is positive, choose that as v_star
                        elif ray_to_vertice[ray_to_vertice > 0].shape[0] == 1:
                            v_star = cell_vertices[np.argmax(ray_to_vertice), :]
                        
                        # get the intersection points.
                        intersection = np.array([[v_star[0], ray(v_star[0])],
                                                 [ray_inverse(v_star[1]), v_star[1]]])
                        
                        # compute the distance traversed in that cell.
                        d = np.sqrt((intersection[0,0]-intersection[1,0])**2 + (intersection[0,1]-intersection[1,1])**2)
                    
                    # if two corners have different sign than two others.
                    elif np.prod(ray_to_vertice) > 0:
                        
                        # if the ray crosses horizontally, get the intersection points.
                        if ray_to_vertice[0]*ray_to_vertice[1] > 0:
                            intersection = np.array([[cell_vertices[0,0], ray(cell_vertices[0,0])],
                                                     [cell_vertices[1,0], ray(cell_vertices[1,0])]])
                        
                        # if the ray crosses vertically, get the intersection points.
                        else: 
                            intersection = np.array([[ray_inverse(cell_vertices[0,1]), cell_vertices[0,1]],
                                                     [ray_inverse(cell_vertices[3,1]), cell_vertices[3,1]]])
                        
                        # compute the distance traversed in that cell.
                        d = np.sqrt((intersection[0,0]-intersection[1,0])**2 + (intersection[0,1]-intersection[1,1])**2)
                        
                    # if one or more of the elements is zero, the ray intersects exactly one or more corners.
                    else:
                        # if the ray intersects one corner, it doesn't actually travel through the cell.
                        if np.count_nonzero(ray_to_vertice == 0) == 1:
                            d = 0
                            
                        # if the ray intersects two corners, it travels along an edge.
                        elif np.count_nonzero(ray_to_vertice == 0) == 2:
                            
                            # due to the geometry of the problem, this will only occur at horizontal edges.
                            # assume that the ray distance is split equally between the cells sharing that edge..
                            d = self.dy/2
                        
                        # the ray cannot insect more that two corners...
                        else:
                            raise ValueError('Something is not right...')
                    
                    # set the matrix element to the computed distance.
                    self.D[i,j] = d
                    
    def get_ray(self, sensor_a, sensor_b, get_inverse=False):
        
        # compute the slope.
        m = (sensor_b[1]-sensor_a[1])/(sensor_b[0]-sensor_a[0])
        
        # compute the intersect.
        b = sensor_a[1] - m*sensor_a[0]
        
        # return the line and its inverse, if asked.
        if get_inverse:
            return lambda x: m*x + b, lambda y: (y-b) / m
        # otherwise, just return the line.
        else:
            return lambda x: m*x + b
    
    def solve(self, parameters):
        # generate a slowness field from the parameters.
        self.slowness.generate(parameters, self.slowness_model['mean'], self.slowness_model['stdev'])
        
        # get the exponential to ensure positivity.
        self.s = np.exp(self.slowness.random_field)
        
        # compute the arrival timels.
        self.t = np.dot(self.D, self.s)
        
    def invert(self, lamb):
            
        # get the (regularised) least squares solution.
        self.s_inv = np.linalg.multi_dot((np.linalg.inv(np.dot(self.D.T, self.D) + lamb*np.eye(self.D.shape[1])), self.D.T, self.t))
        
    def plot_slowness(self, width=6, plot_rays=False, inverse=False):
        
        plt.figure(figsize=(width,width*self.aspect))
        
        # get the reference slowness.
        if not inverse:
            slowness = self.s.reshape(self.grid_midpoints[0].shape)
        
        # get the inverse slowness
        else:
            slowness = self.s_inv.reshape(self.grid_midpoints[0].shape)
        
        # plot slowness
        im = plt.imshow(slowness, cmap='plasma', origin='lower', extent=self.mesh['extent'])
        plt.colorbar(im, fraction=0.08)
        
        # if rays are to be plotted, call the subroutine.
        if plot_rays:
            self.plot_rays(is_child=True)
        
        plt.show()
        
    def plot_coverage(self, width=6, plot_rays=False):
        
        plt.figure(figsize=(width, width*self.aspect))
        
        # get the coverage (normalised sum of distance travelled in each cell)
        coverage = self.D.sum(axis=0); coverage = coverage/coverage.max()
        im = plt.imshow(coverage.reshape(self.grid_midpoints[0].shape), cmap='plasma', origin='lower', extent=self.mesh['extent'])
        plt.colorbar(im, fraction=0.08)
        
        # if rays are to be plotted, call the subroutine.
        if plot_rays:    
            self.plot_rays(is_child=True)
        
        plt.show()
        
    def plot_rays(self, width=6, is_child=False):
        
        # set the default colors.
        colorize = False
        ray_color = 'k'
        
        # if the travel time has already been computed, colorize the rays according to that.
        if hasattr(self, 't'):
            colorize = True
            my_cmap = cm.get_cmap('Greens', 100)
            colors = self.t/self.t.max()
        
        # if this is not called from either of the other plotting functions,
        # set up a figure.
        if not is_child:
            plt.figure(figsize=(width,width*self.aspect))
        
        # iterate through all the possible rays.
        for i in range(self.n_sensor_depths**2):   
            
            # get the ray.
            ray = self.get_ray(self.left_sensors[int(i/self.n_sensor_depths),:], 
                               self.right_sensors[i%self.n_sensor_depths,:])
            
            # compute the line.
            x = np.linspace(self.left_boundary, self.right_boundary)
            y = ray(x)
            
            # color the lines, if the switch is on.
            if colorize:
                ray_color = my_cmap(colors[i])
            
            plt.plot(x, y, c=ray_color)
        
        # if this is not called from either of the other plotting functions,
        #  show the figure.
        if not is_child:
            plt.show()
