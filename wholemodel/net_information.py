
trafficLights = ('gneJ1', 'gneJ3')

linkEdges = { 'gneJ1':{'in':('gneE7', 'gneE0', 'gneE3', 'gneE5'), 'out':('gneE6','gneE1','gneE2','gneE4')} , \
				'gneJ3':{'in':('gneE9', 'gneE4', 'gneE11', 'gneE13'), 'out':('gneE8', 'gneE5', 'gneE10', 'gneE12')} }

actionDuration = 5

useSpeed = True

useHalting = True

usePhase = False

generateTrips = "2net/generateTrips.py"

netXml = "2net/2net.net.xml"

tripsXml = "2net/7200net2.trip.xml"

rouXml = "2net/7200net2.rou.xml"

trafficLightsNum = 2

preProcessDataFile = 'feature/prepro3.npy'

sumocfg = "2net/7200.sumocfg"