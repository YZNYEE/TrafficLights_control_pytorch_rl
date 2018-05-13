trafficLights = ('gneJ0', 'gneJ1', 'gneJ2')

linkEdges = { 'gneJ0':{'in':('gneE7', 'gneE5', 'gneE3', 'gneE1'), 'out':('gneE6','gneE4','gneE2','gneE0')} , \
				'gneJ1':{'in':('gneE15', 'gneE0','gneE13'), 'out':('gneE14', 'gneE1', 'gneE12')} , \
				'gneJ2':{'in':('gneE2', 'gneE9', 'gneE11'), 'out':('gneE3', 'gneE8', 'gneE10')}}

adjTrafficLights = {'gneJ0':('gneJ1', 'gneJ2'), \
					'gneJ1':('gneJ0', 'gneJ2'), \
					'gneJ2':('gneJ0', 'gneJ1')}

actionDuration = 5

useSpeed = True

useHalting = True

usePhase = True

useStep = True

useDuration = True

generateTrips = "3net/generateTrips.py"

netXml = "3net/3net.net.xml"

tripsXml = "3net/7200net3.trip.xml"

rouXml = "3net/7200net3.rou.xml"

trafficLightsNum = 3

preProcessDataFile = 'feature/prepro3net.npy'

sumocfg = "3net/7200.sumocfg"