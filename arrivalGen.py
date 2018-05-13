
secondsInMin = 60
secondsInHour = 60 * 60
secondsInDay = 60 * 60 * 24


# tripsList attri: [depart, from, to]
def writeTripsXml(tripsList, file):

	with open(file, 'w') as trips:
		trips.write("""<?xml version=\"1.0\"?>""" + '\n' + '\n')
		trips.write("""<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">""" \
			+ '\n')
		cnt = 0
		for i in tripsList:
			depart = i[0]
			fm = i[1]
			to = i[2]
			trips.write("""    <trip id=\"""" + str(cnt) +"""\" depart=\"""" + str(depart) + """\" from=\"""" + str(fm) + """\" to=\"""" + str(to) + """\"/>\n""")
			cnt+=1
		trips.write("</routes>\n")

# arrivalList attri: [depart, rouEdgeList]
def writeRouXml(arrivalList, file):

	with open(file, 'w') as routes:
		routes.write("""<?xml version=\"1.0\"?>""" + '\n' + '\n')
		routes.write("""<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">""" \
			+ '\n')
		cnt = 0
		for i in arrivalList:
			depart = i[0]
			edges = i[1]
			routes.write("""	<vehicle id=\"""" + str(cnt) +"""\" depart=\"""" +str(depart) + """\">\n """)
			edgeList = ""
			for j in edges:
				edgeList = edgeList + j + ' '
			routes.write("""		<route edges=\"""" + edgeList + """\"/>\n """)
			routes.write("	</vehicle>\n")
			cnt += 1
		routes.write("</routes>\n")

# informationFlow: [from, to, begin, end, probability]
def writeFlowXml(informationFlow, file):

	cnt = 0
	with open(file, 'w') as flow:
		
		for infor in informationFlow: 
			
			fr = infor[0]
			to = infor[1]
			begin = infor[2]
			end = infor[3]
			probability = infor[4]

			flow.write('<flow')
			flow.write(""" id=\"""" + str(cnt) + """\"""")
			flow.write(""" from=\"""" + str(fr) + """\"""")
			flow.write(""" to=\"""" + str(to) + """\"""")
			flow.write(""" begin=\"""" + str(begin) + """\"""")
			flow.write(""" end=\"""" + str(end) + """\"""")
			flow.write(""" probability=\"""" + str(probability) + """\"/>\n""")
			flow.write("</flow>\n")

			cnt+=1

