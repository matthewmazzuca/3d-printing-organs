import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as im
from skimage import filters
import mahotas as mh
from PIL import Image
import pylab
from scipy import random
from pybrain.structure.modules import KohonenMap
from mpl_toolkits.mplot3d import Axes3D


def countcells(stg):
# 	dna = im.imread(stg)
	dna = Image.open(stg)
	dna = dna.convert('L')
	dna = np.array(dna)
	# plt.imshow(dna)
	# plt.show()
	# T = filters.threshold_otsu(dna)
	# plt.imshow(dna > T)
	# plt.show()
	dnaf = im.gaussian_filter(dna, 1)
	T = filters.threshold_otsu(dnaf)
	plt.imshow(dna > T)
	plt.show()
	
	labeled, nr_objects = mh.label(dnaf > T)
	watershed(stg, labeled)
	print "The number of cells is: " + str(nr_objects)
	return labeled,nr_objects

def watershed(stg, labeled):
	dna = Image.open(stg)
	dna = dna.convert('L')
	dna = np.array(dna)
	labeled = mh.cwatershed(dna.max() - dna, labeled)
	plt.imshow(labeled)
	plt.show()
	return
	
def kohonen():
	som = KohonenMap(2, 10)

	pylab.ion()
	p = pylab.plot(som.neurons[:,:,0].flatten(), som.neurons[:,:,1].flatten(), 's')

	for i in range(25000):
		# one forward and one backward (training) pass
		som.activate(random.random(2))
		som.backward()

		# plot every 100th step
		if i % 100 == 0:
			p[0].set_data(som.neurons[:,:,0].flatten(), som.neurons[:,:,1].flatten())
			pylab.draw()
	
def twod_bar():
	fig = plt.figure()
	ax = fig.add_subplot(111)

	## the data
	N = 15
	measuredMeans = [((1.3*10**5) + (1.1*10**5))/2, ((2.3*10**5) + (1.5*10**5))/2, 
					((2.4*10**5) + (1.5*10**5))/2, ((2.4*10**5) + (2.0*10**5))/2, 
					((2.0*10**5) + (2.7*10**5))/2, ((0.9*10**5) + (1.0*10**5))/2,
					((1.1*10**5) + (0.7*10**5))/2, ((1.3*10**5) + (1.1*10**5))/2,
					((2.3*10**5) + (1.5*10**5))/2, ((2.4*10**5) + (2.0*10**5))/2,
					((1.4*10**5) + (2.0*10**5))/2, ((0.9*10**5) + (1.0*10**5))/2,
					((1.1*10**5) + (0.7*10**5))/2, ((2.4*10**5) + (2.4*10**5))/2,
					((2.3*10**5) + (2.7*10**5))/2]
	measuredStd =   [((1.3*10**5) - (1.1*10**5)), ((2.3*10**5) - (1.5*10**5)), 
					((2.4*10**5) - (1.5*10**5)), ((2.4*10**5) - (2.0*10**5)), 
					((2.0*10**5) - (2.7*10**5)), ((0.9*10**5) - (1.0*10**5)),
					((1.1*10**5) - (0.7*10**5)), ((1.3*10**5) - (1.1*10**5)),
					((2.3*10**5) - (1.5*10**5)), ((2.4*10**5) - (2.0*10**5)),
					((1.4*10**5) - (2.0*10**5)), ((0.9*10**5) - (1.0*10**5)),
					((1.1*10**5) - (0.7*10**5)), ((2.4*10**5) - (2.4*10**5)),
					((2.3*10**5) - (2.7*10**5))]
	nnMeans = [(1.4*10**5), (2.1*10**5), (2.4*10**5), (1.9*10**5), (2.3*10**5),(0.8*10**5)
				,(1.3*10**5), (1.3*10**5), (1.3*10**5), (2.1*10**5), (2.3*10**5),(0.95*10**5)
				,(1.1*10**5), (2.5*10**5), (2.2*10**5)]
# 	nnMatrix = np.random.randint(0,5,(10,10))
# 	hinton(nnMatrix)
	names = ['Square T1', 'Square T2', 'Square T3', 'CircleLg T1', 'CircleLg T2', 'CircleLg T3', 'CircleSm T1', 
			'CircleSm T2', 'CircleSm T3', 'Stressed 1', 'Stressed 2', 'Stressed 3', 'Control N1', 'Control N2', 'Control N3']
	nnStd = [(1.4*10**5), (2.1*10**5), (2.4*10**5), (1.9*10**5), (2.3*10**5),(0.8*10**5)
				,(1.3*10**5), (1.3*10**5), (1.3*10**5), (2.1*10**5), (2.3*10**5),(0.95*10**5)
				,(1.1*10**5), (2.5*10**5), (2.2*10**5)]
				
	for x in range(0,15):
		nnStd[x] =   measuredMeans[x] - nnMeans [x]

	## necessary variables
	ind = np.arange(N)                # the x locations for the groups
	width = 0.35                      # the width of the bars

	## the bars
	rects1 = ax.bar(ind, measuredMeans, width,
					color='black',
					yerr=measuredStd,
					error_kw=dict(elinewidth=2,ecolor='red'))

	rects2 = ax.bar(ind+width, nnMeans, width,
						color='red',
						yerr=nnStd,
						error_kw=dict(elinewidth=2,ecolor='black'))

	# axes and labels
	ax.set_xlim(-width,len(ind)+width)
	# ax.set_ylim(0,45)
	ax.set_ylabel('Concentration')
	ax.set_title('NN Estimation vs Measured')
	xTickMarks = [names[i] for i in range(0,15)]
	ax.set_xticks(ind+width)
	xtickNames = ax.set_xticklabels(xTickMarks)
	plt.setp(xtickNames, rotation=45, fontsize=10)

	## add a legend
	ax.legend( (rects1[0], rects2[0]), ('Measured', 'Neural Net') )

	plt.show()
	return
	
	
def threed_bar():
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
		xs = np.arange(20)
		ys = np.random.rand(20)

		# You can provide either a single color or an array. To demonstrate this,
		# the first bar of each set will be colored cyan.
		cs = [c] * len(xs)
		cs[0] = 'c'
		ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

	ax.set_xlabel('5 Neuron Packet')
	ax.set_ylabel('Iteration Number')
	ax.set_zlabel('Relative Error (Nromalized)')
	ax.set_title('Error Variance in NN over Iterations')

	plt.show()
	return

def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    pylab.fill(xcorners, ycorners, colour, edgecolor=colour)
    
def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if pylab.isinteractive():
        pylab.ioff()
    pylab.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    pylab.fill(np.array([0,width,width,0]),np.array([0,0,height,height]),'gray')
    pylab.axis('off')
    pylab.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        pylab.ion()

def plothint():
	x = np.array([[0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84],
				  [0.46,0.2,0.13,0.96,0.7,0.45,0.67,0.84,0.39,0.84]])
	y = abs(np.random.randn(15,15))
	for i in range(0,16):
		if (i == 9 or i == 10 or i == 11):
			y[i] = -y[i]
	hinton(abs(np.random.randn(15,15)))
	plt.title('Hinton Diagram after Training')
	plt.show()

stg = 'fibronectin.jpeg'
stg2 = 'dna.jpeg'

# countcells(stg)
# countcells(stg2)
# kohonen()
twod_bar()
# threed_bar()
# plothint()
