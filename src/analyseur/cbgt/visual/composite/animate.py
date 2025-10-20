
# ~/analyseur/cbgt/visual/composite/animate.py
#
# Documentation by Lungsi 20 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#

import re

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from analyseur.cbgt.loader import LoadSpikeTimes
from analyseur.cbgt.visual.peristimulus import PSTH
from analyseur.cbgt.visual.popurate import PSRH
from analyseur.cbgt.visual.popact import PopAct
from analyseur.cbgt.visual.raster import rasterplot

# def __get_spiketimes_superset(filename, rootpath, dir_number):
#     pattern_with_nucleus_name = r"\_(.*?)\."
#     nucelus = re.search(pattern_with_nucleus_name, filename)
#
#     nucleus_title = nucelus + " (" + str(np.round(dir_number*100, decimals=1)) + "% decay)"

def animate(frame):
    plt.clf()

    nucleus_title = "PTN ("+str(np.round(decayfolderid[dirlist[frame]]*100, decimals=1))+"% decay)"

    filepath = rootpath + dirlist[frame] + filename

    loadST = LoadSpikeTimes(filepath)
    spiketimes_superset = loadST.get_spiketimes_superset()

    #pact = PopAct(spiketimes_superset)
    #fig = psth._anim_plot(nucleus=nucleus_title) #f'Frame {frame}'
    #fig = pact.plot(nucleus=nucleus_title) #f'Frame {frame}'
    fig = rasterplot(spiketimes_superset, nucleus=nucleus_title, neurons=range(400))

    fig.savefig("fig"+dirlist[frame]) # png files

    return []

rootpath = "/home/lungsi/DockerShare/data/parameter_search/6aMar2025/CORTEX/"
filename = "/spikes_PTN.csv"
decayfolderid = {
    "0": 0, "1": 0.10, "2": 0.15, "3": 0.20, "4": 0.25, "5": 0.30,
    "6": 0.35, "7": 0.40, "8": 0.45, "9": 0.50, "10": 0.55, "11": 0.60,
    "12": 0.65, "13": 0.70, "14": 0.75, "15": 0.80, "16": 0.85, "17": 0.90,
    "18": 0.95, "19": 1.0,
    }
dirlist = list(decayfolderid.keys())

fig = plt.figure(figsize=(10, 6))
ani = animation.FuncAnimation(fig, animate, frames=len(dirlist),
                              interval=500, repeat=True, blit=False)

#ani.save("PSTH_of_GPi_all_decays.gif", writer="pillow", fps=5)
#ani.save("PSTH_of_GPi_all_decays.mp4", writer="ffmpeg", fps=5,
#         extra_args=["-vcodec", "libx264"])

#from matplotlib.animation import HTMLWriter
#ani.save("PSTH_of_GPi_all_decays.html", writer=HTMLWriter(fps=5))

plt.show()
#plt.close()