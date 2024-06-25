import sys

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import ast
import codecs

def init_df(file, max=44):
    df = pd.read_csv(file, sep=",", nrows=max)
    if not "coords" in df.columns:
        prep_dist(df)
        df.to_csv(file, index=False, mode='a')
    else:
        df['coords'] = df['coords'].apply(ast.literal_eval)
    return df
def berechne_distanz(coords1, coords2):
    # Beispielhafte Distanzberechnung zwischen zwei Koordinaten
    return ((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)**0.5

def prep_dist(df):
    df['coords'] = df['city'].apply(coordinates)
    for i, row_i in df.iterrows():
         df[i] = [distance(row_i['coords'], row_j['coords']) for j, row_j in df.iterrows()]

def coordinates(city):
    geolocator = Nominatim(user_agent="1")
    location = geolocator.geocode(city)
    return [location.latitude, location.longitude]

def distance(coord1, coord2):
    # better distances than geodesic?
    d = geodesic(coord1, coord2).kilometers
    return int(d)

def printMatrix(m):
    print('\n'.join(['\t'.join(["." if str(cell) == "0" else str(cell) for cell in row]) for row in m]))

# from a matrix corresponding to one matchday,
# returns a list of groups, with the host as first element
def getGroups(matrix):
    groups = []
    for i, x in enumerate(matrix):
        if x[i] == 1:
            guests = [j for j, v in enumerate(x) if v == 1 and j != i]
            groups.append([i] + guests)
    return groups

def drawScatter(allGroups, loc, name):
    clrs = [
        'b',  # Blue
        'g',  # Green
        'r',  # Red
        'c',  # Cyan
        'm',  # Magenta
        'y',  # Yellow
        'k',  # Black
        '#FFA07A',  # LightSalmon
        '#20B2AA',  # LightSeaGreen
        '#9400D3',  # DarkViolet
        '#32CD32',  # LimeGreen
        '#FF4500',  # OrangeRed
        '#6A5ACD',  # SlateBlue
        '#00CED1',  # DarkTurquoise
        '#FF69B4',  # HotPink
        '#800000',  # Maroon
        '#008080',  # Teal
        '#FFD700',  # Gold
        '#8B4513',  # SaddleBrown
        '#2E8B57',  # SeaGreen
    ]
    gdf = gpd.read_file("DEU_adm.zip")
    teams = range(len(loc))
    if allGroups[0]:
        days = len(allGroups)
        plt.figure().set_figheight(days * 4)
        for k, groups in enumerate(allGroups):
            ax = plt.subplot(days, 1, k + 1)
            gdf.plot(ax=ax, color='white', edgecolor='black')
            for i, g in enumerate(groups):
                plt.scatter([loc[x][1] for x in g[1:]], [loc[x][0] for x in g[1:]], s=80, c=clrs[i])
                plt.scatter(loc[g[0]][1], loc[g[0]][0], s=80, c=clrs[i], marker='D', edgecolor="black", linewidths=2)
    else:
        gdf.plot(color='white', edgecolor='black')
        for x in teams:
            plt.scatter(loc[x][1], loc[x][0], s=80, c=clrs[x])        
    plt.savefig(name + ".png", bbox_inches='tight')


def evaluate(allGroups, df):
    sum = 0
    perteam = {name: 0 for name in df['name']}
    out = ""
    for k in range(len(allGroups)):
        print(f"--day {k+1}--")
        out += f"--day {k+1}--\n"  
        for g in allGroups[k]:
            host = df['name'][g[0]]
            string = f"{host}: "
            for i in g[1:]:
                guest = df['name'][i]
                string += f"{guest}, "
                distance = df[str(i)][g[0]]
                #print(f"distance {host} - {guest}: {distance}km")
                sum += distance
                #print(f"sum is now {sum}")
                perteam[guest] += distance

            print(string[:-2].encode('utf-8').decode('utf-8'))
            out += string[:-2].encode('utf-8').decode('utf-8')+ "\n"
    
    min_key = min(perteam, key=lambda k: perteam[k])
    max_key = max(perteam, key=lambda k: perteam[k])
    print(f"aggregate distance travelled: {sum} km")
    out += f"-- summary -- \n"
    out += f"aggregate distance travelled: \t{sum} km\n"
    out += f"average distance travelled: \t{int(sum/len(perteam))} km\n"
    out += f"minimal distance travelled: \t{perteam[min_key]} km, by {min_key}\n"
    out += f"maximal distance travelled: \t{perteam[max_key]} km, by {max_key}\n"
    print(f"minimal distance per team: \t{perteam[min_key]} km, by {min_key}")
    print(f"maximal distance per team: \t{perteam[max_key]} km, by {max_key}")
    out += f"-- all teams --\n"
    for key in sorted(perteam, key= lambda x : -perteam[x]):
        tab =  "\t\t" if len(key) < 7 else "\t"
        out += f"{key}:{tab}{perteam[key]} km\n"
    print(out)
    return out
