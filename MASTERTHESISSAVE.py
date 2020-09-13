# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:36:06 2020

@author: Michal
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# IMPORTING MODULES
# =============================================================================
# Data manipution
import pandas as pd

# Maps API used for scraping
import googlemaps

# Used for distance calculation
from math import sin, cos, sqrt, atan2, radians

import math

# Data storage
import pickle

# Matematical optimization module
from ortools.sat.python import cp_model

# Various
import time
from collections import namedtuple, defaultdict
import sys
import numpy as np

import conda

import os #The next part is only used for the population and coverage graph seen in the Thesis
#os.environ['PROJ_LIB'] = r'C:\Users\Michal\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
#os.environ['PROJ_LIB'] = r"C:\Users\Michal\Anaconda3\pkgs"

#from mpl_toolkits.basemap import Basemap

#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#Used for constructing the have_items array used in the optimization tool 
Set = namedtuple("Set", ['index', 'items'])

 

#Dataset manipulation part

#reading the population data for each "area unit"
poplocations = pd.read_csv(r"C:\Users\Michal\Desktop\MasterThesis\Datasets\Explored\used\2013-mb-dataset-Total-New-Zealand-Individual-Part-1.csv", encoding='cp1252')

# Reading the GPS coordinates for the same areas
areas_GPS = pd.read_csv(r"C:\Users\Michal\Desktop\Truedata.csv", sep = ";")
areas_GPS.columns = ["Description_GPS2", "LAT", "LON"]

#Picking only the relevant information
poplocations = poplocations[46630:48641]
poplocations = poplocations[["Description","2013_Census_census_usually_resident_population_count(1)"]]
poplocations = poplocations[poplocations["2013_Census_census_usually_resident_population_count(1)"] > 0]

#The "-" before the LAT was not copyied durng the manual extraction of the data
areas_GPS["LAT"] = areas_GPS["LAT"].apply(lambda x: x*(-1))

#Merging the data for the GPS locations with the data about population counts
pop_and_GPS = pd.merge(areas_GPS, poplocations, how = "right", left_on = "Description_GPS2", right_on = "Description")
pop_and_GPS = pop_and_GPS.dropna()

# LOADING THE TWO DATASETS of HOSPITALS on NEW ZEALAND
ngoHospitals = pd.read_csv(r"C:\Users\Michal\Desktop\MasterThesis\Datasets\Explored\used\LegalEntitySummaryNGOHospital.csv")
privHospitals = pd.read_csv(r"C:\Users\Michal\Desktop\MasterThesis\Datasets\Explored\used\LegalEntitySummaryPublicHospital.csv")

# Concatenation of the two datasets of hospitals as I will not make any distinction between the types                        
ngoHospitals2 = pd.concat([ngoHospitals,privHospitals])

addresses_data = ngoHospitals2[[" Premises Address", "Premises Name"]]
addresses_data["Precise Address"] = ngoHospitals2[" Premises Address"] + " " + ngoHospitals2["Premises Name"]
addresses_data["Latitude"] = None
addresses_data["Longitude"] = None
addresses_data = addresses_data[["Precise Address","Latitude","Longitude"]]


  
 
"""

This next section was used to download the needed GPS coordinates for hospitals
this time through Google Maps API (the coordinates were gathered manually for
the population density from a special map consisting of the area units not visible
on Google)

"""
## =============================================================================
## gmaps_key = googlemaps.Client(key = "xxxxx")
## 
## for i in range(0, len(addresses_data), 1):
##     geocode_result = gmaps_key.geocode(addresses_data.iat[i,0])
##     try: 
##         Latitude = geocode_result[0]["geometry"]["location"]["lat"]
##         Longitude = geocode_result[0]["geometry"]["location"]["lng"]
##         addresses_data.iat[i, addresses_data.columns.get_loc("Latitude")] = Latitude
##         addresses_data.iat[i, addresses_data.columns.get_loc("Longitude")] = Longitude
## 
##     except:
##         Latitude = None
##         Longitude = None
## 
## print(geocode_result[0])
## 
## =============================================================================
# #Data stored to a file in order to save money (the Google API is paid per download)
## =============================================================================
## with open("addresses_data.txt", "wb") as fp:
##     pickle.dump(addresses_data, fp)
##     
## =============================================================================



#Google API data loaded
with open("addresses_data.txt", "rb") as fp:
    addresses_data = pickle.load(fp)
    
# Getting rid of NaN data in Hospital DataFrame
addresses_data = addresses_data.dropna()

# Defining the calculation of distance through Haversine equation 
def distance_from_GPS(lat1,lon1,lat2,lon2):

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    #The Haversine equation
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = 6378 * c
    
    return distance 


#Initiating list that maps all the accesible regions to every hospital 
list_of_coverage1 = []
for i in range(len(addresses_data)):
    one_hospital1 = []
   
    for j in range(len(pop_and_GPS)):
      distance = distance_from_GPS(addresses_data.iloc[i,1], addresses_data.iloc[i,2], pop_and_GPS.iloc[j,1], pop_and_GPS.iloc[j,2])

# The distance restriction is imposed due to short possible time of flight for the eVOTLs 
      if distance <= 70: 
          one_hospital1.append(j)
          
    list_of_coverage1.append(one_hospital1)

# As an input to the opitimization algorithm we only need the regions we are actually able to cover
covered = set(x for l in list_of_coverage1 for x in l)


##The uncovered regions only inhabit around 0.005% of the whole population (22,000 people)
##We will exclude them as we are simply unable to cover them without a significan increase in the reach of the eVTOLs
uncovered_regions = list(set(range(len(pop_and_GPS))) - set(covered))
pop_and_GPS = pop_and_GPS.drop(uncovered_regions)

#Lets now create another list_of_coverage, this time only with the reachable regions
list_of_coverage2 = []

for i in range(len(addresses_data)):
    one_hospital2 = []
    
    for j in range(len(pop_and_GPS)):
      distance = distance_from_GPS(addresses_data.iloc[i,1], addresses_data.iloc[i,2], pop_and_GPS.iloc[j,1], pop_and_GPS.iloc[j,2])

# The distance restriction is imposed due to short possible time of flight for the eVOTLs 
      if distance <= 70: 
          one_hospital2.append(j)
          
    list_of_coverage2.append(one_hospital2)
 

## We need to store the population in different regions in a list for the model
population_list = list(pop_and_GPS["2013_Census_census_usually_resident_population_count(1)"])

# creating matrix of distances between the hospitals and the regions, also needed as an imput for the model
matrix_of_distances = np.zeros([len(pop_and_GPS), len(addresses_data)], dtype ='int')

for i in range(len(pop_and_GPS)):
    for j in range(len(addresses_data)):
        matrix_of_distances[i][j] = distance_from_GPS(addresses_data.iloc[j,1], addresses_data.iloc[j,2], pop_and_GPS.iloc[i,1], pop_and_GPS.iloc[i,2])
  
#Setting up of variables for the optimization model
#Creating sets that can be easily used for storage of multiple information
sets = []
for i in range(len(list_of_coverage2)):
    sets.append(Set(i-1, set(map(int, list_of_coverage2[i]))))
    
#Support variables further used in the model
#setrange is the number of hospitals that can be used for the placement of the eVTOLs
setrange = range(len(list_of_coverage2))

#itemrange is the number of regions we need to serve with the ambulance
itemrange = range(len(population_list))

#two matrices conveying reachability of regions from the different hospitals 
have_item = np.zeros([len(setrange), len(population_list)], dtype ='int') 
have_item2 = np.zeros([len(setrange), len(population_list)], dtype ='int') 
# 
"""The DSM model that will be used requires two different distances r1 and r2
  between the closest hospital and the region. R1 has more strict rules and 
  shorter needed arrival times of the ambulance.""" 
            
for s in setrange:
    for i in sets[s].items:
            have_item[s,i] = 1
            
for s in setrange:
    for i in itemrange:
        if matrix_of_distances[i][s] <= 100:
            have_item2[s,i] = 1
        
            
column_indices = np.where((have_item == 0).all(0))[0]

population_list = [int(i) for i in population_list]


"""The OR-tools developed by Google used in the thesis require some adjustments coded below
  in order to print the result and the details of the model in the needed form """
 
class VarArrayAndObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.start = time.time()
        self.start_interval = time.time()

    def on_solution_callback(self):
        t1 = time.time()
        time_used = t1 - self.start
        interval_used = t1 - self.start_interval
        self.start_interval = t1
        print('Interval using %.4f, Accu using %.4f, Solution %i' % (interval_used, time_used, self.__solution_count), end = ', ')
        print('objective value = %i' % self.ObjectiveValue())
        self.__solution_count += 1

    def solution_count(self):
        return self.__solution_count

"""
    Here starts the actual solver of the system we are aiming to optimize
    as an imput, it is taking the number of hospitals, regions and the coverages
    of the individual hospitals
"""
    
def solver(set_count, item_count, sets, max_minutes=10):

    # Initiates the model from the or-tools library
    model = cp_model.CpModel()

    # Initiates the output variables optimized by the model
    amb = [0]*set_count
    x = [0]*set_count
    is_located = [0]*set_count
    
    # Initiates the variables that intermediate the construction of the model
    y1 = [0]*item_count
    y2 = [0]*item_count
            
    # The library requires the setup of both the output and intermediary variables
    # in order to speed up the optimization
    
    # These variables are set up for every potential ambulance site
    for s in setrange:
        amb[s] = model.NewIntVar(0, 50, name = 'amb%s'%(s))
        x[s] = model.NewBoolVar(name = 'x%s'%(s))
        is_located[s] = model.NewBoolVar( 'is_located%s'%(s))

    # These variables are set up for every region that needs to be covered
    for i in itemrange:
        y1[i] = model.NewBoolVar(name = 'y1%i'%(i))
        y2[i] = model.NewBoolVar(name = 'y2%i'%(i))

        
    # Initializes the theoretical constrains given by DSM
   
    # Constrains implemented for every region
    for i in itemrange: 
        # Modeling of y1 = BoolVar == 1 iff the point is covered once in r1
        model.Add(y1[i] == 1).OnlyEnforceIf(any(amb[s]*matrix_of_distances[i][s] <= 70*amb[s] for s in setrange))
        
        # Modeling of y2 = BoolVar == 1 iff the point is covered twice in r1        
        model.Add(y2[i] == 1).OnlyEnforceIf(sum([1 for s in setrange if amb[s]*matrix_of_distances[i][s] <= 70*amb[s]]) >= 2) 
        
        
        # Sum of ambulance vehicles in r1 must be same of higher than how many times the point is covered
        model.Add(sum(amb[s]*have_item[s][i] for s in setrange) >= 6) # Theoretical constraint n.4                                             

        # We want to cover all the people within r2
        #model.Add(sum(have_item2[s][i]*amb[s] for s in setrange) >= 1) # Theoretical constraint n.2 this does not actually play any role in our case CAN BE DELETED
                      

    # How many eVTOLs can we afford
    model.Add(sum(amb) <= 500) # Theoretical constrain n.6                                                                                
    
    # Proportion of people covered in r1 from the total population (international standards)
    model.Add(sum(population_list[i]*y1[i] for i in itemrange) >= int((sum(population_list[i] for i in itemrange))*0.7)) # Theoretical constrain n.3

    # Constrains implemented for every potential ambulance spot            
    for s in setrange:
        
        # Maximal number of eVTOLs on one ambulance site
        model.Add(amb[s] <= 50) # Theoretical constraint n.7     
        
        # Setting up the is_located boolean variable that defines if the potential site is actually used for ambulance with eVTOLs
        model.Add(amb[s] >= 1).OnlyEnforceIf(x[s])
        model.Add(amb[s] == 0).OnlyEnforceIf(x[s].Not())
        
        model.Add(is_located[s] == 1).OnlyEnforceIf(x[s])
        model.Add(is_located[s] == 0).OnlyEnforceIf(x[s].Not())
        
        # Constraint added on top of the DSM model setting maximal coverage of people by one ambulance spot
        # The 50,000 constant refers to the number of people that is usually reliant on one ground ambulance nowdays
        model.Add(50000*(amb[s]) >= sum(population_list[i]*is_located[s]*have_item[s][i] for i in itemrange))
        

    # Initializes the main objective function
    model.Maximize(sum(population_list[i]*y2[i] for i in itemrange)) # 1
    
    # Creates the solver and solves the previously set up system
    solver = cp_model.CpSolver()
    
    # Setting boundary for solving time
    solver.parameters.max_time_in_seconds = 60*max_minutes
   
    # Using a fixed search tool for speeding up the search
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    
    # Using the previously updated class to print out results
    solution_printer = VarArrayAndObjectiveSolutionPrinter(amb)
    status = solver.SolveWithSolutionCallback(model, solution_printer)
    
    #Printing out the results
    print('----------------')
    print('Status       : %s' % solver.StatusName(status))
    print('#sol found   : %i' % solution_printer.solution_count())
    print('Branches     : %i' % solver.NumBranches())
    print('Wall time    : %f s' % solver.WallTime())

    obj = solver.ObjectiveValue()
    
    # The main results we are looking for, the number of eVTOLs at every possible ambulance spot
    solution = [0]*set_count
    for idx, xi in enumerate(amb):
        solution[idx] = solver.Value(xi)
    
    # Setting up the printing of the properties of the model
    is_optimal = -1
    if status == cp_model.OPTIMAL:
        is_optimal = 1
    elif status == cp_model.FEASIBLE:
        is_optimal = 0
    print('Obj          : %s' %(obj))
    print('Solution     : %s' %(','.join(map(str, solution))))
    print('----------------')
    
    return obj, is_optimal, solution

# The results will be later used in the simulation
obj, is_optimal, solution = solver(matrix_of_distances.shape[1], matrix_of_distances.shape[0], sets, max_minutes=10)
print(sum(solution))

##"""
#
#    The next part of the code is dedicated to creation of a graphs for the Thesis.
#    Main goal is to project all the regions and ambulance sites used by the model
#    to get an idea about how effective the coverage is.
#    
#"""
#
#Choosing just the hospitals with eVTOLs assigned by the model
hospitals_to_plot = [i for i, e in enumerate(solution) if e != 0]
hospitals_to_plot2 = addresses_data.iloc[hospitals_to_plot,:]
hospitals_to_plot2 = hospitals_to_plot2[["Latitude","Longitude"]]
hospitals_to_plot3 = addresses_data.iloc[hospitals_to_plot,:]

## Creation of the plot, usage of the Basemap module 
#fig = plt.figure(figsize=(8, 8))
#map = Basemap(projection='lcc', resolution="h",
#            width=2E6, height=2E6, 
#            lat_0=-40, lon_0=170,)
#map.etopo(scale=0.5, alpha=0.5)
#
## Setting up the hospitals with eVTOL ambulances proposed
#lats = list(hospitals_to_plot2["Latitude"])
#lons = list(hospitals_to_plot2["Longitude"])
#
## Setting up the regions
#lats2= list(pop_and_GPS["LAT"])
#lons2= list(pop_and_GPS["LON"])
#
## Mapping to the right coordinate system required by the graph
#x, y = map(lons, lats)
#x2 , y2 = map(lons2, lats2)
#
##Mapping both the hospitals and the regions
#map.scatter(x, y, marker = 'D')
#map.scatter(x2, y2, marker = 'D', s = 5)
#
##Including the reach distances of the respective eVTOLs stationed in the hospitals
#for lo, la in zip(lons,lats):
#    circle = Circle(xy=map(lo,la),radius=65000, fill=False, color = "black")
#    plt.gca().add_patch(circle)
#
#plt.show()

######################### Beginning of the simulation part ###########################################

"""Creating the probability distribution used for drawing of emergency situations.
  Assumption made is that probability of the nodes is increasing with the amount 
  of people living in the area"""
  
elements = range(len(population_list))
probabilities = [i/(sum(i for i in population_list)) for i in population_list]

"""From literature: daily number of calls to the NZ emergency is around 700, 
  that number is distributed according to a distribution found in the literature.
  All is described in the Thesis."""
  
list_of_frequencies = [4,4,3,3,3,3,3,4,5,6,7,7,7,7,7,7,7,7,7,6,5,4,4,3]

# used for the perturbation of demand in peak times
list_of_frequencies2 = [4,4,3,3,3,3,3,4,5,6,7,7,10,10,10,10,10,10,7,6,5,4,4,3]

# used for the perturbation of demand in peak times
list_of_frequencies3 = [4,4,9,9,9,9,9,4,5,6,7,7,7,7,7,7,7,7,7,6,5,4,4,3]

#Defining the drawing of WHERE will the incident that needs ambulance happen
def draw(elements, list_of_frequencies ,probabilities):
    
    return np.random.choice(elements, list_of_frequencies, p=probabilities)
    
#Defining generation of the individual ambulance sites
def ambulances_generation(solution, list_list_of_coverage):
    ambulances = []
    for ambulance_site in range(len(list_list_of_coverage)):
        if solution[ambulance_site] > 0:
            ambulances.append(AmbulanceSite(solution[ambulance_site], list_list_of_coverage[ambulance_site]))
    
    return ambulances
            
#Defining the main class used for the simulation 
class AmbulanceSite:
    
    #num_ambulances is equal to the available eVTOLs in that specific site
    def __init__(self, num_ambulances, list_of_coverage):
        self.num_ambulances = num_ambulances
        self.list_of_coverage = list_of_coverage
        self.failures = 0
        
        # This list serves as tracking through time of the ambulances that left the site
        self.dispatched_ambulances = []
        
    #Function simulates the departure of one eVTOL that leave for the average of 60 minutes
    def dispatch(self, time_period):
        if self.num_ambulances > 0:
            self.num_ambulances -= 1
            
            self.dispatched_ambulances[time_period] += 1
            
        if self.num_ambulances == 0:
            print("Station Empty")
            self.failures += 1
            
            return False 
          
    #Signaling of the return of the eVTOL, needed for availibity tracking
    def returned(self, time_period):
        self.num_ambulances += 1

""" Initialization of the simulation using the number of ambulances, coverage 
    of demand points and location of the ambulance sites from the coverage model """
    
ambuls = ambulances_generation(solution, list_of_coverage2)


# Set up for the perturbation study of the parameters for 200 runs on each parameter
for repetition in range(200):
    
    #List used for time tracking needed for the returns of the ambulances
    times_out = []
    
    #Used in the assertion at the end to check if all calls were addressed with a departure
    num_of_dispatches = 0
    
    timing = -1
    
    #24hrs simulation of the NZ emergency situations
    for hour in range(24):
        
        #Every hour is modeled into 6 intervals long 10 minutes each
        for i in range(6): 
            
            timing += 1
            
            #Respective amount of incidents is drawn each 10 minutes according to the distribution found in literature
            emergencies = draw(elements, list_of_frequencies[hour], probabilities)
    
            #Loop kept for tracking of the departures from individual stations        
            for k in range(len(ambuls)):
                ambuls[k].dispatched_ambulances.append(0)
                
            """The dispatching part of the simulation, every emergency is addressed by 
              one dispatched eVTOL from a station that can reach the demand point"""
              
            for emergency in emergencies:
                
                a = [ambuls[wh] for wh in range(len(ambuls)) if emergency in ambuls[wh].list_of_coverage]
                b = max(site.num_ambulances for site in a)
                for h in a:
                    if h.num_ambulances == b:
                        h.dispatch(time_period = timing)
                        num_of_dispatches +=1 # musime pocitat neuspesny vyslani 
                        break 
            
            #Cohort approach taken to track which ambulances should be already back
            times_out.append(0)
            for j in range(len(times_out)):
                times_out[j] += 10
                
                #We only return the eVTOLs that have been out for the average of 60 mins
                if times_out[j] == 60:
                    for number in range(len(ambuls)):
                        
                        #For the ambulance sites that that dispatched some eVTOLs at the specific time, we return the amount of eVTOLs sent out 
                        if ambuls[number].dispatched_ambulances[j] > 0:
                          for l in range(ambuls[number].dispatched_ambulances[j]):
                              ambuls[number].returned(time_period = j)
                              #print("ambulance number {} returned".format(number))
    
        
        #print(times_out)    
        #print(num_of_dispatches)
        #print([ambul.num_ambulances for ambul in ambuls])
print(sum(c.failures for c in ambuls))
print(sum(solution))
        
#We are checking if every call was addressed
assert num_of_dispatches == sum(list_of_frequencies)*6
    
########################### END OF THE SIMULATION ###################################################

#Plotting of the Emergency calls distribution
xline = range(len(list_of_frequencies))

fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1])
ax.bar(xline, list_of_frequencies)
ax.set_xbound(lower = -1, upper = 24)
ax.set_ybound(lower = 0, upper = 8)
ax.set_title("Distribution of the Emergency Calls")
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Calls per 10 minutes")
ax.grid(False)

#Plotting the perturbations of multiples 
x = [2,3,4,5,6,7,8]
y_evtols = [187,202,224,252,274,296,311]
y_fails = [9,7,2,1,0,0,0]

plt.rc("axes", labelsize = 12)

fig4, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(x, y_evtols, color = color)
ax1.set_ybound(lower = 0, upper = 349)

ax1.set_xlabel("Coverage Multiple")
ax1.set_ylabel("EVTOLs in The System", color = color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.plot(x, y_fails, color = color)
ax2.set_xlabel("Coverage Multiple")
ax2.set_ylabel("Unanswered Emergency Calls (%)", color = color)

plt.axvline(x = 6, dashes = (5,2))

fig4.tight_layout()


#Plotting the perturbations of People per Ambulance ratio
x = [30,40,50,55,60,70,80]
y_evtols = [351,303,274,274,249,238,229]
y_fails = [0,0,0,0,1,3,5]

plt.rc("axes", labelsize = 12)

fig5, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(x, y_evtols, color = color)
ax1.set_ybound(lower = 0, upper = 359)

ax1.set_xlabel("Covered People by One Vehicule (thousands)")
ax1.set_ylabel("EVTOLs in The System", color = color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.plot(x, y_fails, color = color)
ax2.set_ylabel("Unanswered Emergency Calls (%)", color = color)

plt.axvline(x = 50, dashes = (5,2))

fig5.tight_layout()

#Plotting the perturbations of Number of Emergency Calls 
x = [350,700,840,1050,1400]
y_evtols = [202,274,274,311,440]
y_neededMultiples = [3,6,6,8,12]

plt.rc("axes", labelsize = 12)

fig6, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(x, y_evtols, color = color)
ax1.set_ybound(lower = 0, upper = 459)

ax1.set_xlabel("Number of Emergency Calls Per Day")
ax1.set_ylabel("EVTOLs in The System", color = color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.plot(x, y_neededMultiples, color = color)
ax2.set_ylabel("Coverage Multiple Needed", color = color)
ax2.set_ybound(lower = 0, upper = 13)

fig6.tight_layout()

#Plotting the perturbations of Time spent on one dispatch 
x = [20,40,60,70,80,90]
y_evtols = [202,224,274,274,274,311]
y_neededMultiples = [3,4,6,6,6,8]

plt.rc("axes", labelsize = 12)

fig6, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(x, y_evtols, color = color)
ax1.set_ybound(lower = 0, upper = 459)

ax1.set_xlabel("Time Spent Responding To Emergency Call (Minutes)")
ax1.set_ylabel("EVTOLs in The System", color = color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.plot(x, y_neededMultiples, color = color)
ax2.set_ylabel("Coverage Multiple Needed", color = color)
ax2.set_ybound(lower = 0, upper = 10)

fig6.tight_layout()

#Plotting the perturbations of Max Range
x = [70,90,110,150]
y_evtols = [274,218,218,179]
y_neededMultiples = [6,5,5,6]

plt.rc("axes", labelsize = 12)

fig6, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(x, y_evtols, color = color)
ax1.set_ybound(lower = 0, upper = 459)

ax1.set_xlabel("Maximal Range of eVTOLs (Kilometers)")
ax1.set_ylabel("EVTOLs in The System", color = color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.plot(x, y_neededMultiples, color = color)
ax2.set_ylabel("Coverage Multiple Needed", color = color)
ax2.set_ybound(lower = 0, upper = 8)

fig6.tight_layout()

#Plotting of the Emergency calls distribution As Well As the two Peaks
xline = range(len(list_of_frequencies))

fig7, (ax1, ax2) = plt.subplots(1,2)
ax1.bar(xline, list_of_frequencies)
ax1.set_xbound(lower = -1, upper = 24)
ax1.set_ybound(lower = 0, upper = 10)
fig7.suptitle("Distribution of the Emergency Calls")
ax1.set_xlabel("Hour")
ax1.set_ylabel("Number of Calls per 10 minutes")
ax1.grid(False)

color = 'tab:red'

ax2.bar(xline, list_of_frequencies2, color = color, alpha = 0.5)
ax2.bar(xline, list_of_frequencies3, color = "tab:green", alpha = 0.5)
ax2.set_xlabel("Hour")
