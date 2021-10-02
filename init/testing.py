#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file has the code required to create a 'general' phase mask. So instead of
only having 4 quadrant we can specify however many we want.

Created on Thu Feb 21 08:22:28 2019

@author: oem
"""
import matplotlib.pyplot as plt
import numpy as np

pi = np.pi

phaseMask = {1: [-pi*0.25, pi*0.25],
             2: [pi*0.25, pi*0.75],
             3: [pi*0.75, pi*1.25],
             4: [pi*1.25, pi*1.75],
             }

phaseColors = ['red', 'yellow', 'blue', 'green']

# Do some tests first
corrPhaseMasks = {}
allVals = np.array(list(phaseMask.values()))
for phaseKey in phaseMask:
    vals = phaseMask[phaseKey]

    if len(vals) != 2:
        msg = "The dividers for the phase sections should have a minimum value"
        msg += " and a maximum value in a list as a value to a dictionary"
        msg += " entry e.g.\n"
        msg += "{1: [-pi*0.25, pi*0.25],\n 2: [pi*0.25, pi*0.75],"
        msg += "\n 3: [pi*0.75, pi*1.25],\n 4: [pi*1.25,"
        msg += "pi*1.75]}"
        msg += "\n\nThe key `%s' has %i values" % (str(phaseKey), len(vals))
        raise SystemExit(msg)

    if any(int(i) == i for i in vals):
        msg = "WARNING!\n\nThe"
        msg += "key `%s' values are actually integers. This" % str(phaseKey)
        msg += " is OK if you want!\n\nThis can be due to integer division in "
        msg += "python2 though. If you know that all phases should be floats "
        msg += "then check you have not divided by an integer using python2!"
        print(msg + "\n\nWARNING!")

    if vals[0] >= vals[1]:
        msg = "The first value in the Phase Mask should be less than the "
        msg = "second value.\n\nBad Key = `%s' " % str(phaseKey)
        msg = "Bad Vals = (%s)" % ", ".join([str(i) for i in vals])
        raise SystemExit(msg)

maxTakeMin = np.diff(allVals, axis=1)
totRot = np.sum(maxTakeMin)
if not (pi*(2-1e-10) < totRot < pi*(2+1e-10)):
    msg = "The total rotation covered by the phase mask is not 2 pi!\n"
    msg += "Please make sure that all angles between 0 and 2 pi are covered"
    msg += " E.g.\n"
    msg += "{1: [-pi*0.25, pi*0.25],\n 2: [pi*0.25, pi*0.75],"
    msg += "\n 3: [pi*0.75, pi*1.25],\n 4: [pi*1.25,"
    msg += "pi*1.75]}"
    raise SystemExit(msg)

for phaseKey in phaseMask:
    Min, Max = phaseMask[phaseKey]
    if Min < pi < Max:
        corrPhaseMasks[phaseKey] = [[Min, pi],
                                    [-pi, -pi + (Max - pi)]]
    elif Min > pi:
        minDiff, maxDiff = Min - pi, Max - pi
        corrPhaseMasks[phaseKey] = [[minDiff - pi, maxDiff - pi]]
    else:
        corrPhaseMasks[phaseKey] = [[Min, Max]]

r = 1
angles = np.arange(-pi, pi, 0.0001)
x = np.cos(angles) * r
y = np.sin(angles) * r
for i, phaseKey in enumerate(corrPhaseMasks):
    for vals in corrPhaseMasks[phaseKey]:
        mask = vals[0] < angles
        mask *= angles < vals[1]
        plotX = x[mask]
        plotY = y[mask]
        plt.plot(plotX, plotY, color=phaseColors[i], lw=5)
