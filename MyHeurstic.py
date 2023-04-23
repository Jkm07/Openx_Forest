import pandas as ps
import numpy as np

Wilderness_Area = [
    "Rawah",
    "Neota",
    "Comanche Peak",
    "Cache la Poudre"
]

Soil_Type = np.arange(1, 41).astype(str)

Soil_Type = np.char.add("Soil_type_", Soil_Type)

names = ["Elevation", 
         "Aspect", 
         "Slope", 
         "Horizontal_Distance_To_Hydrology", 
         "Vertical_Distance_To_Hydrology", 
         "Horizontal_Distance_To_Roadways", 
         "Hillshade_9am",
         "Hillshade_Noon",
         "Hillshade_3pm",
         "Horizontal_Distance_To_Fire_Points"]

names = np.concatenate([names, Wilderness_Area, Soil_Type, ["Cover_Type"]])
data = ps.read_csv("covtype.data", sep=",", header=0, names=names)

class MyHeuristic:
    def __init__(self, sums):
        self.sums = sums
    def predict(self, x):
        soil_types = x.iloc[:, 14:54].idxmax(axis=1)
        return soil_types.apply(lambda soil_type: self.sums["Soil_type_" + str(soil_type)].idxmax()).values
    
my_heuristic = MyHeuristic(data.iloc[:,14:55].groupby(['Cover_Type']).sum())