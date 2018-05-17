trafficLights = ('gneJ0', 'gneJ2', 'gneJ3', 'gneJ5')

others = ('gneJ1', 'gneJ11', 'gneJ12', 'gneJ4', 'gneJ9')

Edges = { 'gneE0':('gneJ0', 'gneJ1'), \
			'gneE1':('gneJ1', 'gneJ0'), \
			'gneE10':('gneJ3', 'gneJ5'), \
			'gneE11':('gneJ5', 'gneJ3'), \
			'gneE14':('gneJ2', 'gneJ9'), \
			'gneE15':('gneJ9', 'gneJ2'), \
			'gneE18':('gneJ3', 'gneJ11'), \
			'gneE19':('gneJ11', 'gneJ3'), \
			'gneE2':('gneJ0', 'gneJ2'), \
			'gneE20':('gneJ5', 'gneJ12'), \
			'gneE21':('gneJ12', 'gneJ5'), \
			'gneE3':('gneJ2', 'gneJ0'), \
			'gneE4':('gneJ0', 'gneJ3'), \
			'gneE5':('gneJ3', 'gneJ0'), \
			'gneE6':('gneJ0', 'gneJ4'), \
			'gneE7':('gneJ4', 'gneJ0'), \
			'gneE8':('gneJ2', 'gneJ5'), \
			'gneE9':('gneJ5', 'gneJ2'), \
		}

'''
linkEdge = {}
for tl in trafficLights:
	linkEdge[tl] = {'in':[], 'out':[]}
for tl in others:
	linkEdge[tl] = {'in':[], 'out':[]}
for edge in Edges:
	node = Edges[edge]
	out = node[0]
	i = node[1]
	linkEdge[i]['in'].append(edge)
	linkEdge[out]['out'].append(edge)
print(linkEdge)
'''

'''
edges = []
for edge in Edges:
	edges.append(edge)
print(edges)
'''


linkEdges = {'gneJ2': {'in': ['gneE15', 'gneE2', 'gneE9'], 'out': ['gneE8', 'gneE3', 'gneE14']}, \
			'gneJ5': {'in': ['gneE8', 'gneE10', 'gneE21'], 'out': ['gneE9', 'gneE11', 'gneE20']}, \
			'gneJ12': {'in': ['gneE20'], 'out': ['gneE21']}, \
			'gneJ4': {'in': ['gneE6'], 'out': ['gneE7']}, \
			'gneJ0': {'in': ['gneE3', 'gneE5', 'gneE1', 'gneE7'], 'out': ['gneE2', 'gneE0', 'gneE4', 'gneE6']}, \
			'gneJ9': {'in': ['gneE14'], 'out': ['gneE15']}, \
			'gneJ3': {'in': ['gneE4', 'gneE11', 'gneE19'], 'out': ['gneE5', 'gneE10', 'gneE18']}, \
			'gneJ11': {'in': ['gneE18'], 'out': ['gneE19']}, \
			'gneJ1': {'in': ['gneE0'], 'out': ['gneE1']}}


edgelist = ['gneE1', 'gneE19', 'gneE14', 'gneE3', 'gneE6', 'gneE4', \
			'gneE0', 'gneE9', 'gneE5', 'gneE18', 'gneE8', 'gneE15', \
			'gneE7', 'gneE10', 'gneE2', 'gneE20', 'gneE21', 'gneE11']



actionDuration = 5

useSpeed = True

useHalting = True

usePhase = True

useStep = True

useDuration = True

generateTrips = "generateTrips.py"

netXml = "net4.net.xml"

tripsXml = "net4.trip.xml"

rouXml = "net4.rou.xml"

trafficLightsNum = 4

preProcessDataFile = 'prepro4.pt'

sumocfg = "net4.sumocfg"

