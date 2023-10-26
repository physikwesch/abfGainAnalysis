"""
DESCRIPTION:
Module containing all data analysis functions. Each function iterates over every trial, calculates desired quantities,
and saves them into a user specified location. All the quantities used for the final output are specifically
saved into a folder called cellName+'averages/'. All folders and data are created and data stored according to described
tauName/groupName/cellName/trialName convention. The contents of this module are imported for use in other scripts. This
module of analysis functions requires importing all methods functions.

NOTES:
All of these functions could be subsumed into a newly defined class, which would avoid the need of passing in most
function arguments; specifically pathName, tauName, and groupName to every function. However, this makes the syntax
significantly more involved and therefore harder to adapt, edit, or understand for other borrowing users. Only one
function, f_thesaurusData() returns anything. All other functions take inputs and save inputs, returning None.

UPDATE:
Last edited by Neil Wesch on 12 October 2023.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import re
import os
import sys
import random
import time
import pickle
import natsort
import collections
import pyabf
import seaborn as sns

from functionsMethods import f_rolledPositiveFourierTransform
from functionsMethods import f_gaussianFilter
from functionsMethods import f_localVariationCoefficient
from functionsMethods import f_complexConfidenceInterval

"""
f_thesaurusData()
### Takes the axon binary files (.abf) specified in dataNames and parses out the voltage and current data. These are
renamed according to tauName/groupName/cellName/trialName convention and converted into numpy arrays. The time step
total duration of the recording are saved as dt and nPoints. Using regular expressions, the .abf file summary is searched
for the protocol name, which if conforming with naming convention, finds the input correlation time and changes it from
"OU_YZms" to 'tauYZms/'.
N.B. The .abf files must be saved in a folder called 'rawData/'. The user must specify the data file names and directory
where the project is stored for all outputs to be saved correctly. This is done in the abfDataInfo.json file.
"""

def f_thesaurusData(pathName, dataNames):

    if not os.path.isdir(pathName+'thesauri/'):  # if folder does not exist
        os.mkdir(pathName+'thesauri/')  # make new folder called thesauri

    tauNames = []  # declare a variable which holds the protocol names extracted from each .abf file

    for dataName in dataNames:  # iterate over each .abf file

        file = pyabf.ABF(pathName+'rawData/'+dataName[:-1]+'.abf')  # uses module pyabf to open specified data file

        protocol = re.search(r'"(\w.)+_+(\w+)"', str(file))  # find a protocol name like "OU_25" which specifies correlation time
        if protocol == None:  # if a different protocol is used, or naming convention not followed, script quits
            sys.exit('invalid protocol name')  # having the correct naming convention for the protocol is crucial for consistency
        protocol = str(protocol.group(0)).replace('"', '').replace('_', '').replace('OU', 'tau')+'/'  # convert "OU_25" to tau25/

        tauNames.append(protocol)  # creates a list of all protocol names extracted from .abf files

    groupNames = [z.rsplit('_', 1)[0] for z in dataNames]  # groupNames are the data part of dataNames
    groupNames = [z + '/' for z in groupNames]  # append an / to the end to simplify syntax for file saving
    cellNames = [z.rsplit('_', 1)[-1] for z in dataNames]  # cell Names are the terminal designation of dataNames

    for tauName, dataName, groupName, cellName in zip(tauNames, dataNames, groupNames, cellNames):  # iterate over each four-pair

        if not os.path.isdir(pathName+tauName):  # if folder does not exist
            os.mkdir(pathName+tauName)  # make folder called tauName in working folder
        if not os.path.isdir(pathName+tauName+groupName):  # if folder does not exist
            os.mkdir(pathName+tauName+groupName)  # make folder called pathName/tauName/groupName
        if not os.path.exists(pathName+tauName+groupName+cellName):  # if folder does not exist
            os.mkdir(pathName+tauName+groupName+cellName)  # create folder tauName/groupName/cellName in working folder

        if not os.path.exists(pathName+tauName+groupName+cellName+'current/'):  # if folder does not exist
            os.mkdir(pathName+tauName+groupName+cellName+'current/')  # create folder current in working folder
        if not os.path.exists(pathName+tauName+groupName+cellName+'voltage/'):  # if folder does not exist
            os.mkdir(pathName+tauName+groupName+cellName+'voltage/')  # create folder voltage in working folder
        if not os.path.isdir(pathName+tauName+groupName+cellName+'averages/'):  # if fold does not exist
            os.mkdir(pathName+tauName+groupName+cellName+'averages/')  # create folder averages to save final outputs

        thesaurus = collections.defaultdict(list)  # declare dictionary to hold cellName/trialName designations

        file = pyabf.ABF(pathName+'rawData/'+dataName[:-1]+'.abf')  # uses module pyabf to open specified data file

        nPoints = file.sweepPointCount  # [int] the total number of points in the voltage/current arrays
        dt = file.sweepLengthSec / file.sweepPointCount  # [s] the data recording time step
        tRecord = nPoints * dt  # [s] length of total voltage/current recording

        for m in range(file.sweepCount-1):  # iterate over every sweep

            trialName = 'sweep{}'.format(m)  # define a trialName string for saving data
            thesaurus[cellName].append(trialName)  # to each cellName key add trialName values (all strings)

            file.setSweep(sweepNumber=m, channel=0)  # look at abf data in sweep number m and channel zero
            voltage = np.array(file.sweepY)  # channel zero Y data is voltage, extract and convert to array
            np.save(pathName+tauName+groupName+cellName+'/voltage/'+trialName+'_V', voltage)  # save each voltage

            file.setSweep(sweepNumber=m, channel=1)  # look at abf data in sweep number m and channel one
            current = np.array(file.sweepY)  # channel one Y data is current, extract and convert to array
            np.save(pathName+tauName+groupName+cellName+'/current/'+trialName+'_I', current)  # save each current

        with open(pathName+'thesauri/'+'thesaurus_'+tauName[:-1]+'_'+groupName[:-1]+'.pkl', 'wb') as z:  # create a new thesaurus based on tauName and groupName
            pickle.dump(thesaurus, z)  # save the dictionary of cellName keys and trialName values as pickled object

        np.save(pathName+tauName+groupName+cellName+'averages/'+'vtrace', voltage)  # save a copy of last extracted voltage seperately for plotting
        np.save(pathName+tauName+groupName+cellName+'averages/'+'itrace', current)  # save a copy of last extracted current separately for plotting

    return (tauNames, groupNames, cellNames, nPoints, dt, tRecord)


"""
f_inputAutoCorrelation
### Calculates the normalized mean-subtracted autocorrelation of the input current for each trial. Only the central window 
of duration tWindown seconds is kept. The standard deviation and variance of the input are calculated and saved; done 
separately for quality control.
N.B. scipy.correlate does not perform normalization which is required to obtain the correct order of magnitude for 
dynamic gain calculations. This step is given its own line for clarity.
"""


def f_inputAutoCorrelation(pathName, tauNames, groupNames, nPoints, nWindow):

    for tauName, groupName in zip(tauNames, groupNames):  # iterate over each tauName/groupName pair

        with open(pathName+'thesauri/thesaurus_'+tauName[:-1]+'_'+groupName[:-1]+'.pkl', 'rb') as z:  # open the current tauName/groupName thesaurus file
            thesaurus = pickle.load(z)  # open the dictionary of cellName keys and trialName values (all strings)
        cellNames = list(thesaurus.keys())  # create a list containing each cell name

        for cellName in cellNames:  # iterate over all cells in the dataset
            trialNames = thesaurus.get(cellName)  # for each cell create a list containing all trial names
            trialNames = natsort.natsorted(trialNames)  # ensures consistency with natsorted current and voltage data

            if not os.path.isdir(pathName+tauName+groupName+cellName+'/iac/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'/iac/')  # make iac folder in working folder
            if not os.path.isdir(pathName+tauName+groupName+cellName+'/istd/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'/istd/')  # make istd folder in working folder
            if not os.path.isdir(pathName+tauName+groupName+cellName+'/ivar/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'/ivar/')  # make ivar folder in working folder

            currentRaw = []  # declare list to hold the names of all current arrays
            for fileName in os.listdir(os.path.join(pathName+tauName+groupName+cellName+'/current/')):  # iterate over all files in this folder
                if fileName.endswith('I.npy'):  # only select files in the folder which end with I.npy
                    currentRaw.append(fileName)  # create a list of all files ending with I.npy
            currentData = natsort.natsorted(currentRaw)  # sort the elements of the list according to natsort

            for y, trialName in enumerate(trialNames):  # iterate over each trial

                inoise = np.load(pathName+tauName+groupName+cellName+'/current/'+currentData[y], allow_pickle=True)  # load each current trace into numpy array
                istd = np.std(inoise)  # calculate the standard deviation of the input current
                ivar = np.var(inoise)  # calculate the variation of the input current

                iac = sp.signal.correlate(inoise-np.mean(inoise), inoise-np.mean(inoise), mode='full', method='fft')  # use Fourier transform method to calculate autocorrelation, mode='full' must be used to account for edge effects
                iac = iac/nPoints  # signal.correlate() does not normalize, this step is crucial to obtain correct order of magnitude for quantities like gain
                iac = iac[int(nPoints-nWindow//2):int(nPoints+nWindow//2)]  # only the central portion of the autocorrelation is needed

                np.save(pathName+tauName+groupName+cellName+'/iac/'+trialName+'_iac', iac)  # save each input autocorrelation as a binary .npy file
                np.save(pathName+tauName+groupName+cellName+'/istd/'+trialName+'_istd', istd)  # save each input standard deviation as a binary .npy file
                np.save(pathName+tauName+groupName+cellName+'/ivar/'+trialName+'_ivar', ivar)  # save each input variance as a binary .npy file

    return ()


"""
f_spikeTriggedAverageCurrent()
### Calculates the spike triggerd average current, and associated firing rate and coefficient of variation, for each 
trial. Separately isolates the average AP waveform and calculates the phase plot for each trial. Spikes are defined as
the positive voltage crossing of the value detect. Spikes which occur in the terminal nWindow//2 sections cannot be 
windowed and are discarded. Spikes included in the analysis are termed permissible spikes. The start/end nWindow//2 
part of each input current is also discarded. tWindow determines frequency resolution, but must be compared to the
total recording duration such that only a small percentage of data is discarded. The loading of all file names containing
the voltage and current and subsequent natsort is done to avoid mismatching. 
"""


def f_spikeTriggeredAverageCurrent(pathName, tauNames, groupNames, nPoints, nWindow, dt, detect, rapid, tSpike):

    for tauName, groupName in zip(tauNames, groupNames):  # iterate over each tauName/groupName pair

        tabdata = []  # declare list to hold each row of sweep data

        with open(pathName+'thesauri/thesaurus_'+tauName[:-1]+'_'+groupName[:-1]+'.pkl', 'rb') as z:  # open the current tauName/groupName thesaurus
            thesaurus = pickle.load(z)  # open the dictionary of cellName keys and trialName values (all strings)
        cellNames = list(thesaurus.keys())  # create a list containing each cell name

        for cellName in cellNames:  # iterate over each cell
            trialNames = thesaurus.get(cellName)  # get the trial names from the pickled dictionary
            trialNames = natsort.natsorted(trialNames)  # ensures consistency with natsorted current and voltage data

            nWaveform = int(np.ceil(tSpike / dt))  # the number of points in each centered spike segment
            deltat = np.arange(-nWaveform // 2, nWaveform // 2) * dt  # the forward/backward time from zero defined as index of detect crossing
            np.save(pathName + tauName + groupName + cellName + 'averages/' + 'deltat', deltat)  # save deltat specifically for plotting

            if not os.path.isdir(pathName+tauName+groupName+cellName+'sta/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'sta/')  # make new folder called sta
            if not os.path.isdir(pathName+tauName+groupName+cellName+'codex/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'codex/')  # make new folder called codex
            if not os.path.isdir(pathName+tauName+groupName+cellName+'events/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'events/')  # make new folder called codex
            if not os.path.isdir(pathName+tauName+groupName+cellName+'nu/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'nu/')  # make new folder called nu
            if not os.path.isdir(pathName+tauName+groupName+cellName+'isi/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'isi/')  # make new folder called isi
            if not os.path.isdir(pathName+tauName+groupName+cellName+'lv/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'lv/')  # make new folder called lv
            if not os.path.isdir(pathName+tauName+groupName+cellName+'cv/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'cv/')  # make new folder called cv
            if not os.path.isdir(pathName+tauName+groupName+cellName+'waveform/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'waveform/')  # make new folder called waveform
            if not os.path.isdir(pathName+tauName+groupName+cellName+'vdot/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'vdot/')  # make new folder called vdot
            if not os.path.isdir(pathName+tauName+groupName+cellName+'vbar/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'vbar/')  # make new folder called vbar
            if not os.path.isdir(pathName+tauName+groupName+cellName+'fwhm/'):  # if folder does not exist
                os.mkdir(pathName+tauName+groupName+cellName+'fwhm/')  # make new folder called fwhm

            voltageRaw = []  # declare variable which will contain all voltage traces for specified correlation time
            for fileName in os.listdir(os.path.join(pathName+tauName+groupName+cellName+'/voltage/')):  # iterate over all files in specified folder
                if fileName.endswith('V.npy'):  # select all .npy files ending with V in their name
                    voltageRaw.append(fileName)  # create a list of all the included voltage data files
            voltageData = natsort.natsorted(voltageRaw)  # natsort the file name order to ensure matching with current

            currentRaw = []  # declare variable which will contain all current traces for specified correlation time
            for fileName in os.listdir(os.path.join(pathName+tauName+groupName+cellName+'/current/')):  # iterate over all files in specified folder
                if fileName.endswith('I.npy'):  # select all .npy files ending with I in their name
                    currentRaw.append(fileName)  # create a list of all the included current data files
            currentData = natsort.natsorted(currentRaw)  # natsort the file name order to ensure matching with voltage

            for y, trialName in enumerate(trialNames):  # iterate over all trials, with y being an index moving through natsort order

                vsoma = np.load(pathName+tauName+groupName+cellName+'/voltage/'+str(voltageData[y]), allow_pickle=True)  #  load yth trial voltage data
                inoise = np.load(pathName+tauName+groupName+cellName+'/current/'+str(currentData[y]), allow_pickle=True)  # load yth trial current data

                spkdex = np.diff((np.array(vsoma) > detect) * 1.)  # all points above detect are set to unity (True) and the difference calculated
                spkdex = np.array(np.where(spkdex == 1))[0]  # positive crossings will be the only indices remaining where a zero (False) has changed to a one (True)

                codex = [x for x in spkdex if nWindow//2 < x < nPoints-nWindow//2]  # iterate over every possible spike, adding permitted times if not occuring during start/end half-window duration

                isi = dt*np.diff(codex)  # interspike intervals are number of points between adjacent spikes multiplied by time step
                lv = f_localVariationCoefficient(isi)  # calculate the local coefficient of variation as defined in [Shinomoto (2005) BioSystems]
                cv = np.std(isi)/np.mean(isi)  # coefficient of variation is standard deviation over mean of interspike intervals

                events = len(codex)  # total number of included spikes
                nu = events/((nPoints-nWindow)*dt)  # firing rate is number of included spikes over duration of truncated voltage trace

                staSpike = np.zeros((events, nWindow))  # matrix to contain spike-triggered current for each event (spike) to be subsequently averaged
                waveformSpike = np.zeros((events, nWaveform))  # declare matrix to store all spike waveforms for a trial
                vdotSpike = np.zeros((events, nWaveform-1))  # declare a matrix to store all membrane potential derivative values for a trial
                vbarSpike = np.zeros((events, nWaveform-1))  # declare a matrix to store all membrane potential adjacent point averaged values for a trial
                fwhmSpike = np.zeros(events)  # declare a vector to store all full-width at half-maximum values for a trial

                for m in range(events):  # iterate over every permissible spike

                    count = np.arange(codex[m]-nWaveform//2, codex[m]+nWaveform//2)  # indices of the waveform window around each spike
                    waveformSpike[m, :] = vsoma[count.astype(int)]  # take preceeding and procedding nWaveform//2 points around spike
                    vdotSpike[m, :] = np.diff(waveformSpike[m, :])/dt  # [mV/ms] the voltage difference between two adjacent point in a spike divided by time step
                    vbarSpike[m, :] = np.convolve(waveformSpike[m, :], np.ones(2), mode='valid')/2  # rolling average of adjacent membrane potential values

                    count = np.arange(codex[m]-nWindow//2, codex[m]+nWindow//2)  # [int] create an array of nWindow length containing the window indices around each spike
                    staSpike[m, :] = inoise[count.astype(int)]  # [pA] isolate the input current at the indices around each spike to calculate spike triggered average current

                    thresh = np.argwhere(vdotSpike[m, :] > rapid)[0]  # [mV] voltage at which dV/dt is greater than user specified onset rapidity in mV/s
                    peak = np.transpose(np.argmax(vbarSpike[m, :]))  # [mV] peak voltage reached during the action potential
                    vhm = (vbarSpike[m, peak]+vbarSpike[m, thresh])/2  # [mV] value of membrane potential at half the spike height
                    fwhmSpike[m] = (np.argwhere(vbarSpike[m, :] > vhm)[-1] - np.argwhere(vbarSpike[m, :] > vhm)[0])*dt  # [ms] number of indices between spike at half height during rise and fall

                sta = np.mean(staSpike, axis=0)  # convert per spike list to single trial total array
                waveform = np.mean(waveformSpike, axis=0)  # convert per spike list to single trial total array
                vdot = np.mean(vdotSpike, axis=0)  # convert per spike list to single trial total array
                vbar = np.mean(vbarSpike, axis=0)  # convert per spike list to single trial total array
                fwhm = np.mean(fwhmSpike, axis=0)  # convert per spike list to single trial total array

                rowData = [trialName, '{:.2f}'.format(nu), '{:.3f}'.format(1e3*fwhm), '{:.2f}'.format(lv), '{:.2f}'.format(cv), '{:d}'.format(events)]  # electrophysiology characteristics per sweep
                tabdata.append(rowData)  # every row contains data from a different sweep

                np.save(pathName+tauName+groupName+cellName+'sta/'+trialName+'_sta', sta)  # save trial total spike triggered average current
                np.save(pathName+tauName+groupName+cellName+'nu/'+trialName+'_nu', nu)  # save trial total firing rate
                np.save(pathName+tauName+groupName+cellName+'codex/'+trialName+'_codex', codex)  # save trial total spike times
                np.save(pathName+tauName+groupName+cellName+'isi/'+trialName+'_isi', isi)  # save trial total interspike intervals
                np.save(pathName+tauName+groupName+cellName+'lv/'+trialName+'_lv', lv)  # save trial total local variation coefficient
                np.save(pathName+tauName+groupName+cellName+'cv/'+trialName+'_cv', cv)  # save trial total coefficient of variation
                np.save(pathName+tauName+groupName+cellName+'waveform/'+trialName+'_waveform', waveform)  # save trial average AP waveform
                np.save(pathName+tauName+groupName+cellName+'vdot/'+trialName+'_vdot', vdot)  # save trial total average dV/dt
                np.save(pathName+tauName+groupName+cellName+'vbar/'+trialName+'_vbar', vbar)  # save trial total (V[t]+V[t+1])/2
                np.save(pathName+tauName+groupName+cellName+'fwhm/'+trialName+'_fwhm', fwhm)  # save trial total average AP full-width half-maximum
                np.save(pathName+tauName+groupName+cellName+'events/'+trialName+'_events', events)  # save trial total number of permissible spikes

        np.save(pathName+tauName+groupName+cellName+'averages/'+'tabdata', tabdata)  # save the electrophysiology characteristics per sweep data

    return ()


"""
f_dynamicGain()
### Calculates a single dynamic gain as the average of dynamic gains for each trial. That is, an average across trials 
of spike triggered average current, input autocorrelation, noise standard deviation, noise variance, and firing rate
and calculated. Using the formula for dynamic gain these produce a single complex valued quantity. Simultaneously, a
grand average (across all spikes over all trials) coeffcient of variation, local variation coefficient, average AP
waveform, membrane potential derivative, membrane potential adjacent point average, and AP full-width at half-maximum 
are calculated; saved and used for output plots.
"""


def f_dynamicGain(pathName, tauNames, groupNames, nWindow, dt, freqPos, freqFilt):

    for tauName, groupName in zip(tauNames, groupNames):  # iterate over every tauName/groupName pair

        with open(pathName+'thesauri/thesaurus_'+tauName[:-1]+'_'+groupName[:-1]+'.pkl', 'rb') as z:  # open the current tauName/groupName thesaurus
            thesaurus = pickle.load(z)  # open the dictionary of cellName keys and trialName values (all strings)
        cellNames = list(thesaurus.keys())  # create a list containing each cell name

        staAvg = []  # declare list to iteratively store spike triggered average per trial
        iacAvg = []  # declare list to iteratively store input autocorrelation per trial
        nuAvg = []  # declare list to iteratively store firing rate per trial
        lvAvg = []  # declare list to iteratively store local variation coeffcient [Shinomoto (2005) BioSyst] per trial
        cvAvg = []  # declare list to iteratively store coefficient of variation per trial
        istdAvg = []  # declare list to iteratively store noise standard deviation per trial
        ivarAvg = []  # declare list to iteratively store noise variance  per trial
        waveformAvg = []  # declare list to iteratively store waveform shape per trial
        vdotAvg = []  # declare list to iteratively store voltage derivative per trial
        vbarAvg = []  # declare list to iteratively store membrane potential adjacent point average per trial
        fwhmAvg = []  # declare list to iteratively store full-width half-maximum per trial
        eventsAvg = []  # declare list to iteratively store number of permissible spikes per trial

        for cellName in cellNames:  # iterate over all cells in the dataset

            trialNames = thesaurus.get(cellName)  # get the trial names from thesaurus_tauName_groupName
            trialNames = natsort.natsorted(trialNames)  # natsort trialNames to ensure correct file ordering

            for trialName in trialNames:  # iterate over each sweep

                sta = np.load(pathName+tauName+groupName+cellName+'sta/'+trialName+'_sta.npy')  # load spike triggered average
                iac = np.load(pathName+tauName+groupName+cellName+'iac/'+trialName+'_iac.npy')  # load input autocorrelation
                nu = np.load(pathName+tauName+groupName+cellName+'nu/'+trialName+'_nu.npy')  # load average firing rate
                lv = np.load(pathName+tauName+groupName+cellName+'lv/'+trialName+'_lv.npy')  # load local variation
                cv = np.load(pathName+tauName+groupName+cellName+'cv/'+trialName+'_cv.npy')  # load variation coeffcient
                istd = np.load(pathName+tauName+groupName+cellName+'istd/'+trialName+'_istd.npy')  # load current standard deviation
                ivar = np.load(pathName+tauName+groupName+cellName+'ivar/'+trialName+'_ivar.npy')  # load current variance
                waveform = np.load(pathName+tauName+groupName+cellName+'waveform/'+trialName+'_waveform.npy')  # load AP waveform
                vdot = np.load(pathName+tauName+groupName+cellName+'vdot/'+trialName+'_vdot.npy')  # load dV/dt of membrane potential
                vbar = np.load(pathName+tauName+groupName+cellName+'vbar/'+trialName+'_vbar.npy')  # load (V[t]+V[t+1])/2 of membrane potential
                fwhm = np.load(pathName + tauName + groupName + cellName + 'fwhm/' + trialName + '_fwhm.npy')  # load AP full-width half-maximum
                events = np.load(pathName+tauName+groupName+cellName+'events/'+trialName+'_events.npy')  # load permissible spike number

                staAvg.append([sta/istd])  # append normalized sta to average sta list
                iacAvg.append([iac/ivar])  # append normalized iac to average iac list
                nuAvg.append([nu])  # append trial firing rate to total average firing rate list
                lvAvg.append([lv])  # append trial local variation to total average local variation list
                cvAvg.append([cv])  # append variation coefficient to total average cariation coefficient list
                istdAvg.append([istd])  # append trial current standard deviation to total average current standard deviation list
                ivarAvg.append([ivar])  # append trial current variance to total average current variance list
                waveformAvg.append([waveform])  # append trial averaged AP waveform to total average AP waveform list
                vdotAvg.append([vdot])  # append trial dV/dt to total average dV/dt
                vbarAvg.append([vbar])  # append trial (V[t]+V[t+1])/2 to total average (V[t]+V[t+1])/2
                fwhmAvg.append([fwhm])  # append trial average full-width half-maximum to total average full-width half-maximum
                eventsAvg.append([events])  # append trial permissible spike number to total average permissible spike number

        staBar = np.mean(np.vstack(staAvg), axis=0)  # convert trial list to single mean array
        iacBar = np.mean(np.vstack(iacAvg), axis=0)  # convert trial list to single mean array
        nuBar = np.mean(np.vstack(nuAvg), axis=0)  # convert trial list to single mean array
        lvBar = np.mean(np.vstack(lvAvg), axis=0)  # convert trial list to single mean array
        cvBar = np.mean(np.vstack(cvAvg), axis=0)  # convert trial list to single mean array
        istdBar = np.mean(np.vstack(istdAvg), axis=0)  # convert trial list to single mean array
        ivarBar = np.mean(np.vstack(ivarAvg), axis=0)  # convert trial list to single mean array
        waveformBar = np.mean(np.vstack(waveformAvg), axis=0)  # convert trial list to single mean array
        vdotBar = np.mean(np.vstack(vdotAvg), axis=0)  # convert trial list to single mean array
        vbarBar = np.mean(np.vstack(vbarAvg), axis=0)  # convert trial list to single mean array
        fwhmBar = np.mean(np.vstack(fwhmAvg), axis=0)  # convert trial list to single mean array
        eventsBar = np.mean(np.vstack(eventsAvg), axis=0)  # convert trial list to single mean array

        staBar, _ = f_rolledPositiveFourierTransform(dt, nWindow, staBar)  # calculate average sta in frequency domain (_ indicates frequency points not used)
        iacBar, _ = f_rolledPositiveFourierTransform(dt, nWindow, iacBar)  # calculate average iac in frequency domian (_ inidcates frequency points not used)

        gain = (nuBar * np.conjugate(staBar) * istdBar) / (iacBar * ivarBar)  # calculate complex dynamic gain
        gain = f_gaussianFilter(gain, freqPos, freqFilt)  # filter complex dynamic gain to obtain complex denoised array

        np.save(pathName+tauName+groupName+cellName+'averages/'+'gain', gain)  # save total averaged gain array
        np.save(pathName+tauName+groupName+cellName+'averages/'+'nu', nuBar)  # save single value of total average firing rate
        np.save(pathName+tauName+groupName+cellName+'averages/'+'lv', lvBar)  # save single value of total average local variation
        np.save(pathName+tauName+groupName+cellName+'averages/'+'cv', cvBar)  # save single value of total average variation coefficient
        np.save(pathName+tauName+groupName+cellName+'averages/'+'waveform', waveformBar)  # save total average AP waveform array
        np.save(pathName+tauName+groupName+cellName+'averages/'+'vdot', vdotBar)  # save total average dV/dt array
        np.save(pathName+tauName+groupName+cellName+'averages/'+'vbar', vbarBar)  # save total average (V[t]+V[t+1])/2 array
        np.save(pathName+tauName+groupName+cellName+'averages/'+'fwhm', fwhmBar)  # save single value of total average full-width half-maximum
        np.save(pathName+tauName+groupName+cellName+'averages/'+'events', eventsBar)  # save single value of total number of permissible spikes

    return ()


"""
f_gainConfidenceIntervals()
### Uses bootstrapping with replacement to calculate a user specified number, nBtsp, of bootstrapped dynamic gain curves
that are projected into complex plane along the axis defined by the true gain and the upper and lower user specified
percentiles are found. Note these bootstrapped values are filtered using Gaussian filter bank prior to sorting.
"""


def f_gainConfidenceIntervals(pathName, tauNames, groupNames, nWindow, nBtsp, dt, freqPos, freqFilt):

    if nBtsp < 40:  # if the number of user defined noise floor iterations is below forty
        nBtsp = 40  # set the value to fourty so random nonsensical confidence intervals do not occur

    for tauName, groupName in zip(tauNames, groupNames):

        poolNames = ()  # declare variable which will be a list containing all cellName+trialName entries possible

        with open(pathName+'thesauri/thesaurus_'+tauName[:-1]+'_'+groupName[:-1]+'.pkl', 'rb') as z:
            thesaurus = pickle.load(z)  # open the dictionary of keys, items, data which was created when running module_importCurrentVoltage.py script
        cellNames = list(thesaurus.keys())  # create a list containing each cell name

        for cellName in cellNames:  # iterate over all cells in the dataset
            trialNames = thesaurus.get(cellName)  # get the trial names from thesaurus_tauName_groupName
            trialNames = natsort.natsorted(trialNames)

            gain = np.load(pathName + tauName + groupName + cellName + 'averages/' + 'gain.npy')

            for trialName in trialNames:  # iterate over each trial
                poolNames = np.append(poolNames, cellName+'/'+trialName)  # combine into a single list containing all cell name, trial name designations

            gainBtsp = np.zeros((nBtsp, freqFilt.shape[0]), dtype=complex)  # create a matrix with each row containing a single bootstrapped gain curve

            for n in range(nBtsp):  # each loop calculates one bootstrapped gain curve, from which the confidence intervals are obtained

                random.seed(time.time())  # choose new random seed for each iteration
                nTrials = int(poolNames.shape[0])  # the number of trails used for bootstrapped gain curve equals the number used for true gain curve
                r = np.random.randint(0, poolNames.size, (1, nTrials))  # draw nTrials random numbers between 0 and the number of total trials (duplicates possible)

                poolBtsp = poolNames[r.astype(int)].reshape(nTrials)  # for this iteration the included trials are randomly selected from the pool of all trials
                cellBtsp = [x.rsplit('/', 1)[0::2] for x in poolBtsp]  # isolate the randomly selected cell names
                cellBtsp = np.transpose(cellBtsp).reshape(len(cellBtsp), )  # reshape into a horizontal one-dimensional list
                trialBtsp = [x.rsplit('/', 1)[1::2] for x in poolBtsp]  # isolate the randomly selected trial names
                trialBtsp = np.transpose(trialBtsp).reshape(len(trialBtsp), )  # reshape into a horizontal one-dimensional list

                staBtsp = []  # declare variable to hold the bootstrapped spike triggered average current for this iteration
                iacBtsp = []  # declare variable to hold the bootstrapped input autocorrelation for this iteration
                nuBtsp = []  # declare variable to hold the bootstrapped firing rate for this iteration
                istdBtsp = []  # declare variable to hold the bootstrapped standard deviation of the input current for this iteration
                ivarBtsp = []   # declare variable to hold the bootstrapped variance of the input current for this iteration

                for m in range(nTrials):  # total number of randomly chosen trials for bootstrapping is same number as total number of trials for original gain calculation

                    sta = np.load(pathName+tauName+groupName+cellBtsp[m]+'/sta/'+trialBtsp[m]+'_sta.npy')  # load the spike triggered average current from randomly chosen trial
                    iac = np.load(pathName+tauName+groupName+cellBtsp[m]+'/iac/'+trialBtsp[m]+'_iac.npy')  # load the input autocorrelation from randomly chosen trial
                    nu = np.load(pathName+tauName+groupName+cellBtsp[m]+'/nu/'+trialBtsp[m]+'_nu.npy')  # load the firing rate from randomly chosen trial
                    istd = np.load(pathName+tauName+groupName+cellBtsp[m]+'/istd/'+trialBtsp[m]+'_istd.npy')  # load the current standard deviation from randomly chosen trial
                    ivar = np.load(pathName+tauName+groupName+cellBtsp[m]+'/ivar/'+trialBtsp[m]+'_ivar.npy')  # load the current variance from randomly chosen trial

                    staBtsp.append([sta/istd])  # append normalized sta to bootstrap average sta list
                    iacBtsp.append([iac/ivar])  # append normalized iac to bootstrap average iac list
                    nuBtsp.append([nu])  # append firing rate to bootstrap average sta list
                    istdBtsp.append([istd])  # append current standard deviation to bootstrap average current standard deviation list
                    ivarBtsp.append([ivar])  # append current variance to bootstrap average current variance list

                staBtsp = np.mean(np.vstack(staBtsp), axis=0)  # convert trial list to single mean array
                iacBtsp = np.mean(np.vstack(iacBtsp), axis=0)  # convert trial list to single mean array
                nuBtsp = np.mean(np.vstack(nuBtsp), axis=0)  # convert trial list to single mean array
                istdBtsp = np.mean(np.vstack(istdBtsp), axis=0)  # convert trial list to single mean array
                ivarBtsp = np.mean(np.vstack(ivarBtsp), axis=0)  # convert trial list to single mean array

                staBtsp, _ = f_rolledPositiveFourierTransform(dt, nWindow, staBtsp)  # calculate average sta in frequency domain (_ indicates frequency points not used)
                iacBtsp, _ = f_rolledPositiveFourierTransform(dt, nWindow, iacBtsp)  # calculate average iac in frequency domain (_ indicates frequency points not used)

                tfBtsp = (nuBtsp * np.conjugate(staBtsp) * istdBtsp) / (iacBtsp * ivarBtsp)  # calculate complex bootstrapped dynamic gain
                gainBtsp[n, :] = f_gaussianFilter(tfBtsp, freqPos, freqFilt)  # filter complex dynamic gain to obtain complex denoised array which is assigned to matrix

            gainUpp = f_complexConfidenceInterval(0.975, gain, gainBtsp)  # find upper 97.5% of the bootstrapped gain curves (complex valued vector)
            gainLow = f_complexConfidenceInterval(0.025, gain, gainBtsp)  # find lower 2.5% of the bootstrapped gain curves  (complex valued vector)

            np.save(pathName+tauName+groupName+cellName+'averages/'+'gainUpp', gainUpp)  # save upper 97.5% of the bootstrapped gain curves (single complex valued array)
            np.save(pathName+tauName+groupName+cellName+'averages/'+'gainLow', gainLow)  # save lower 2.5% of the bootstrapped gain curves (single complex valued array)

    return ()


"""
f_gainNoiseFloor()
### Using bootstrapping with replacement of spike shifted voltage traces to calculate user specified, nNofl, number of
noise floor gain curves. For each randomly choosen trial all spikes are randomly shifted by the same amount to ensure
spiking pattern remains unchanged. After shifting the terminal nWindow//2 segments have to be discarded. With the
associated noisy input current a dynamic gain curve is calculated. These curves are projected into the complex
plane alone the axis set by the true gain and the upper 95th percentile found to obtain the noise floor. 
N.B. Gain values below the noise floor are statistically indistinguishable from the gain of random spike trains.
"""


def f_gainNoiseFloor(pathName, tauNames, groupNames, nWindow, nNofl, dt, freqPos, freqFilt):

    if nNofl < 20:  # if the number of user defined noise floor iterations is below twenty
        nNofl = 20  # set the value to twenty so random nonsensical noise floor does not occur

    for tauName, groupName in zip(tauNames, groupNames):  # iterate over each tauName/groupName pair

        poolNames = ()  # declare variable which will be a list containing all cellName+trialName entries possible

        with open(pathName+'thesauri/thesaurus_'+tauName[:-1]+'_'+groupName[:-1]+'.pkl', 'rb') as z:  # open the current tauName_groupName thesaurus
            thesaurus = pickle.load(z)  # open the dictionary of keys, items, data which was created when running module_importCurrentVoltage.py script
        cellNames = list(thesaurus.keys())  # create a list containing each cell name

        for cellName in cellNames:  # iterate over all cells in the dataset
            trialNames = thesaurus.get(cellName)  # get the trial names from thesaurus_tauName_groupName
            trialNames = natsort.natsorted(trialNames)

            gain = np.load(pathName+tauName+groupName+cellName+'averages/'+'gain.npy')  # load the complex dynamic gain to be used for confidence interval calculation

            for trialName in trialNames:  # iterate over each trial
                poolNames = np.append(poolNames, cellName+'/'+trialName)  # combine into a single list containing all cell name, trial name designations

            gainNofl = np.zeros((nNofl, freqFilt.shape[0]), dtype=complex)  # create a matrix with each row containing a single noise floor gain curve

            for n in range(nNofl):  # index runs over user specified number of total noise floor iterations

                random.seed(time.time())  # random seed is based on internal machine clock (consider replacing with fixed seed)
                nTrials = int(poolNames.size)  # each bootstrapped noise floor realization must have same number of trials as original data
                r = np.random.randint(0, poolNames.size, (1, nTrials))  # create a one-dimensional numpy array of random numbers of size equal to total trial number

                poolNofl = poolNames[r.astype(int)].reshape(nTrials)  # create a list of randomly chosen cellName_trialName to be used for calculating a noise floor gain curve
                cellNofl = [x.rsplit('/', 1)[0::2] for x in poolNofl]  # from this random pool isolate the cellName string
                cellNofl = np.transpose(cellNofl).reshape(len(cellNofl), )  # reshape into one-dimensional list
                trialNofl = [x.rsplit('/', 1)[1::2] for x in poolNofl]  # from this random pool isolate the trialName string
                trialNofl = np.transpose(trialNofl).reshape(len(trialNofl), )  # reshape into one-dimensional list

                staNofl = []  # declare a variable to iteratively sum each randomly chosen STA
                iacNofl = []  # declare a variable to iteratively sum each original input power spectrum
                istdNofl = []  # declare a variable to iteratively sum each input current standard deviation
                ivarNofl = []  # declare a variable to iteratively sum each input current variance
                nuNofl = []  # declare a variable to iteratively sum each firing rate of randomly chosen trial

                for m in range(nTrials):  # each loop calculates one bootstrapped gain curve, from which the noise floor is obtained

                    iac = np.load(pathName+tauName+groupName+cellNofl[m]+'/iac/'+trialNofl[m]+'_iac.npy')  # load input autocorrelation from randomly chosen trial
                    istd = np.load(pathName+tauName+groupName+cellNofl[m]+'/istd/'+trialNofl[m]+'_istd.npy')  # load current standard deviation from randomly chosen trial
                    ivar = np.load(pathName+tauName+groupName+cellNofl[m]+'/ivar/'+trialNofl[m]+'_ivar.npy')  # load current variance from randomly chosen trial

                    iacNofl.append([iac/ivar])  # append normalized input autocorrelation to noise floor total average input autocorrelation
                    istdNofl.append([istd])  # append current standard deviation to noise floor total average current standard deviation
                    ivarNofl.append([ivar])  # append current variance to noise floor total avarege current variance

                    inoise = np.load(pathName + tauName + groupName + cellNofl[m] + '/current/' + trialNofl[m] + '_I.npy')  # load current from randomly chosen tiral
                    codex = np.load(pathName+tauName+groupName+cellNofl[m]+'/codex/'+trialNofl[m]+'_codex.npy')  # load codex from randomly chosen trial

                    nPoints = int(inoise.shape[0])  # number of points in the voltage/current traces (congruent size is fixed when extracted from raw data)
                    codexNofl = np.mod(codex+np.random.randint(nWindow, nPoints), nPoints) # with periodic boundaries shift each spike time by integer multiple of nWindow steps

                    codex = [x for x in codexNofl if (nWindow//2) < x < (nPoints-nWindow//2)]  # remove spike times occuring in the start/end nWindow/2 duration since they cannot be windowed

                    events = int(len(codex))  # total number of permissible spikes
                    nu = events/((nPoints-nWindow)*dt)  # firing rate is number of permissible spikes over duration of truncated voltage trace

                    staSpike = np.zeros((events, nWindow))  # matrix to contain spike-triggered current for each event (spike) to be subsequently averaged
                    for k in range(events):  # iterate over every permissible spike (i.e. event)
                        count = np.arange(codex[k]-nWindow//2, codex[k]+nWindow//2)  # create an array of nWindow length containing the indices around each spike
                        staSpike[k, :] = inoise[count.astype(int)]  # isolate the input current at the indices around each spike
                    sta = np.mean(staSpike, axis=0)  # take the mean of the input currents around all spikes to obtain STA

                    staNofl.append([sta/istd])  # append normalized spike triggered average current to noise floor total average spike triggered average current
                    nuNofl.append([nu])  # append firing rate to noise floor total average firing rate

                staNofl = np.mean(np.vstack(staNofl), axis=0)  # convert trial list to single mean array
                iacNofl = np.mean(np.vstack(iacNofl), axis=0)  # convert trial list to single mean array
                nuNofl = np.mean(np.vstack(nuNofl), axis=0)  # convert trial list to single mean array
                istdNofl = np.mean(np.vstack(istdNofl), axis=0)  # convert trial list to single mean array
                ivarNofl = np.mean(np.vstack(ivarNofl), axis=0)  # convert trial list to single mean array

                staNofl, _ = f_rolledPositiveFourierTransform(dt, nWindow, staNofl)  # calculate average noise floor sta in frequecny domain (_ indicates frequency points not used)
                iacNofl, _ = f_rolledPositiveFourierTransform(dt, nWindow, iacNofl)  # calculate average noise floor iac in frequecny domain (_ indicates frequency points not used)

                tfNofl = (nuNofl * np.conjugate(staNofl) * istdNofl) / (iacNofl * ivarNofl)  # for each iteration calculate a noise floor dynamic gain
                gainNofl[n, :] = f_gaussianFilter(tfNofl, freqPos, freqFilt)  # filter complex dynamic gain to obtain complex denoised array which is saved to matrix

            noiseFloor = f_complexConfidenceInterval(0.95, gain, gainNofl)  # using all noise floor gains finding the 95% upper confidence interval

            np.save(pathName+tauName+groupName+cellName+'averages/'+'noiseFloor.npy', noiseFloor)  # save single complex valued noise floor gain curve

    return ()


"""
f_baseballCard()
### Plots all desired outputs in a single large figure, colloquially termed a baseball card. All of the quantities which
can be called results have been saved in the averages/ folder. Each card has the file and protocol name in the top right.
Data per trial (i.e. per sweep) is output into a table. The other plots are created with automatically generated axes
limits. Each card is saved as a .png and a .svg file in a separate figures/ folder.
N.B. Instantaneous firing rate is simply 1/isi. Since 1/x is not a linear mapping the probability distribution of the
instantaneous firing rate and interspike interval are different.
"""


def f_baseballCard(pathName, dataNames, tauNames, groupNames, nTrace, dt, fup, freqFilt):

    plt.rcParams['axes.spines.right'] = False  # removes right vertical axis from all subfigures
    plt.rcParams['axes.spines.top'] = False  # removes upper horizontal axis from all subfigures
    plt.rcParams['font.family'] = 'Arial'  # sets the font type for all subfigures
    plt.rc('xtick', labelsize=14)  # sets the size of horizontal tick mark labels for all subfigures
    plt.rc('ytick', labelsize=14)  # sets the size of vertical tick mark labels for all subfigures

    if not os.path.exists(pathName+'figures/'):  # if folder does not exist
        os.mkdir(pathName+'figures/')  # create a folder called 'figures/' in the working directory

    for dataName, tauName, groupName in zip(dataNames, tauNames, groupNames):  # iterate over each three-pair

        with open(pathName+'thesauri/thesaurus_'+tauName[:-1]+'_'+groupName[:-1]+'.pkl', 'rb') as z:  # open the current tauName_groupName thesaurus
            thesaurus = pickle.load(z)  # open the dictionary of keys, items, data which was created when running module_importCurrentVoltage.py script
        cellNames = list(thesaurus.keys())  # create a list containing each cell nam
        for cellName in cellNames:  # iterate over all cells in the dataset
            trialNames = thesaurus.get(cellName)  # get the trial names from thesaurus_tauName_groupName
            trialNames = natsort.natsorted(trialNames)  # natsort trial names to ensure consistent file matching

            gain = np.abs(np.load(pathName+tauName+groupName+cellName+'averages/'+'gain.npy'))  # load total average dynamic gain and calculate magnitude
            gainUpp = np.abs(np.load(pathName+tauName+groupName+cellName+'averages/'+'gainUpp.npy'))  # load total average upper gain confidence interval and calculate magnitude
            gainLow = np.abs(np.load(pathName+tauName+groupName+cellName+'averages/'+'gainLow.npy'))  # load total average lower gain confidence interval and calculate magnitude
            phase = np.angle(np.load(pathName+tauName+groupName+cellName+'averages/'+'gain.npy'))  # load total average dynamic gain and calculate phase
            phaseUpp = np.angle(np.load(pathName+tauName+groupName+cellName+'averages/'+'gainUpp.npy'))  # load total average upper gain confidence interval and calculate phase
            phaseLow = np.angle(np.load(pathName+tauName+groupName+cellName+'averages/'+'gainLow.npy'))  # load total average lower gain confidence interval and calculate phase
            noiseFloor = np.abs(np.load(pathName+tauName+groupName+cellName+'averages/'+'noiseFloor.npy'))  # load total average gain noise floor and calculate magnitude

            nu = np.load(pathName+tauName+groupName+cellName+'averages/'+'nu.npy')[0]  # load average firing rate ([0] converts to float)
            lv = np.load(pathName+tauName+groupName+cellName+'averages/'+'lv.npy')[0]  # load average local variation ([0] converts to float)
            cv = np.load(pathName+tauName+groupName+cellName+'averages/'+'cv.npy')[0]  # load average variation coeffcient ([0] converts to float)
            events = np.load(pathName+tauName+groupName+cellName+'averages/'+'events.npy')[0]  # load averge number of permissible spikes ([0] converts to float)
            fwhm = np.load(pathName+tauName+groupName+cellName+'averages/'+'fwhm.npy')[0]  # load average full-width half-maximum ([0] converts to float)
            deltat = np.load(pathName+tauName+groupName+cellName+'averages/'+'deltat.npy')  # load average time difference vector for AP waveform

            tabdata = np.load(pathName+tauName+groupName+cellName+'averages/'+'tabdata.npy')  # load the sweep averaged data
            tabavgs = [r'$\bf{average}$', f'{nu:.2f}', f'{1e3*fwhm:.3f}', f'{lv:.2f}', f'{cv:.2f}', f'{events:.0f}']  # generates row of averages from each column
            tabtot = np.vstack([tabdata, ['-'*14, '-'*14, '-'*14, '-'*11, '-'*11, '-'*11], tabavgs])  # generates dotted line to separte average row from sweep rows

            vtrace = np.load(pathName+tauName+groupName+cellName+'averages/'+'vtrace.npy')  # load exemplary voltage trace (default is from last sweep)
            itrace = np.load(pathName+tauName+groupName+cellName+'averages/'+'itrace.npy')  # load exemplary current trace (default is from last sweep)

            fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(18, 31), height_ratios=[0.5, 0.9, 0.9, 0.35, 0.7, 0.7], width_ratios=[0.75, 1])  # create baseball card

            ax = axes[0, 0]  # specify title location
            ax.text(0.1, 0.5, f'FILE: {dataName}'+'\n'+'\n'+f'PROTOCOL: {tauName}', fontsize=24, weight='bold', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')  # remove unwanted lines

            ax = axes[1, 0]  # specify subfigure location
            colour = iter(plt.cm.YlGn(np.linspace(0, 1, len(trialNames))))  # specify the colours for each waveform
            for trialName in trialNames:  # plot the average waveform for each sweep
                waveform = np.load(pathName+tauName+groupName+cellName+'waveform/'+trialName+'_waveform.npy')  # load the previously obtained average AP
                ax.plot(1e3*deltat, waveform, linewidth=1, color=next(colour))  # plot the average AP (factor of 1e3 converts to ms)
            ax.set_xlabel(r'$\Delta t$  [ms]', fontsize=18)  # time runs backward/forward from 0 at the user defined detection voltage
            ax.set_ylabel('voltage  [mV]', fontsize=18)  # membrane potential

            ax = axes[2, 0]  # specify subfigure location
            colour = iter(plt.cm.BuPu(np.linspace(0, 1, len(trialNames))))  # specify colours for each line
            for trialName in trialNames:  # each trace contains data from one sweep
                vbar = np.load(pathName+tauName+groupName+cellName+'vbar/'+trialName+'_vbar.npy')  # load (V[t]+V[t+1])/2 for respective trial
                vdot = np.load(pathName+tauName+groupName+cellName+'vdot/'+trialName+'_vdot.npy')  # load dV/dt for respective trial
                vdot = 1e-3 * vdot  # [mV/ms] factor of 0.001 converts from seconds to milliseconds
                ax.plot(vbar, vdot, linewidth=1, color=next(colour))  # create phase space trajectory of spike
            ax.set_xlabel('voltage  [mV]', fontsize=18)  # membrane potential
            ax.set_ylabel('dV/dt  [mV/ms]', fontsize=18)  # membrane potential rate of change

            ax = axes[3, 0]  # specify subfigure location
            ax.plot(np.arange(0, nTrace)*dt, itrace[:nTrace], color='tab:red')  # plot exemplary current from last sweep
            ax.set_xlim([0, nTrace*dt])  # only plot user specified duration (default is three seconds)
            ax.set_xlabel('time  [s]', fontsize=18)  # recording time in seconds
            ax.set_ylabel('current  [pA]', fontsize=18)  # Gaussian coloured noise current in picoamperes

            ax = axes[4, 0]  # specify figure location
            colour = iter(plt.cm.bone(np.linspace(0, 1, len(trialNames))))  # specify colours for each line
            for trialName in trialNames:  # each histogram contains data from one sweep
                isi = 1e3*np.load(pathName+tauName+groupName+cellName+'isi/'+trialName+'_isi.npy')
                sns.ecdfplot(data=isi, color=next(colour), ax=axes[4, 0])  # create histogram of interspike interval distribution
            ax.set_xlim([0, 500])  # set xaxis limits from 0 to 300 ms
            ax.set_xlabel(r'isi  [ms]', fontsize=18)  # interspike intervals
            ax.set_ylabel('CDF', fontsize=18)  # [norm] cumulative distribution function

            ax = axes[5, 0]  # specify subfigure location
            colour = iter(plt.cm.copper(np.linspace(0, 1, len(trialNames))))  # specify colours for each line
            for trialName in trialNames:  # each histogram contains data from one sweep
                isi = np.load(pathName+tauName+groupName+cellName+'isi/'+trialName+'_isi.npy')  # load interspike interval data from each trial
                sns.ecdfplot(data=1/isi, stat='proportion', color=next(colour), ax=axes[5, 0])  # create histogram of instantaneous firing rate distribution
            ax.set_xlim([0, 50])  # set xaxis limits from 0 to 40 Hz
            ax.set_xlabel(r'$\nu_{inst} \ [Hz]$', fontsize=18)  # instantaneous firing rate
            ax.set_ylabel('CDF', fontsize=18)  # [norm] cumulative distribution function

            ax = axes[0, 1]  # specify subfigure location
            ax.plot(np.arange(0, nTrace)*dt, vtrace[:nTrace], color='tab:pink')  # plots last voltage trace in data set
            ax.set_xlim([0, nTrace*dt])  # shows user specified duration of recording from start of sweep
            ax.set_xlabel('time  [s]', fontsize=18)  # recording time
            ax.set_ylabel('voltage  [mV]', fontsize=18)  # membrane potential

            ax = axes[1, 1]  # specify table location
            tab = ax.table(cellText=tabtot, colLabels=[r'$\bf{trialName}$', r'$\bf{\nu \ [Hz]}$', r'$\bf{width \ [ms]}$', r'$\bf{LV}$', r'$\bf{CV}$', r'$\bf{spikes}$'], colWidths=[0.20, 0.22, 0.22, 0.15, 0.15, 0.15], cellLoc='center', edges='open', loc='center')
            tab.auto_set_font_size(False)  # prevent automatic font size selection
            tab.set_fontsize(18)  # specify size of table entries
            tab.scale(1, 2)  # specifies tables relative dimensions
            ax.axis('off')  # removes unwanted lines

            ax = axes[2, 1]  # specify subfigure location
            ax.semilogx(freqFilt, gain, color='tab:orange', linewidth=4, label='mean')  # log-linear plot of dynamic gain magnitude in Hz/pA as a funciton of frequency
            ax.fill_between(freqFilt, gainLow, gainUpp, color='tab:orange', alpha=0.33, label='95%CIs')  # create shaded region between upper and lower confidence interval
            ax.semilogx(freqFilt, noiseFloor, linewidth=3, linestyle='dotted', color='tab:orange', label='noise floor')  # dotted line indicates noise floor
            ax.legend(loc='lower left', framealpha=1, fontsize=14, title=r'$\tau_s = {} \ [ms]$'.format(tauName[-3:-1]), title_fontsize=12)  # legend title contains input correlation time
            ax.set_xlim([1, fup])  # set xaxis limits from 1 up to user specified fup value in Hz
            ax.set_ylabel('gain magnitude  [Hz/pA]', fontsize=18)  # magnitude of dynamic gain
            ax.set_xlabel('frequency  [Hz]', fontsize=18)  # positive frequencies up to Nyquist

            ax = axes[3, 1]  # specify subfigure position
            ax.semilogx(freqFilt, phase, linewidth=4, color='tab:brown')  # log-linear plot of dynamic gain phase in radians as a function of frequency
            ax.fill_between(freqFilt, phaseLow, phaseUpp, color='tab:brown', alpha=0.33)  # create shaded region between upper and lower confidence interval
            ax.set_xlim([1, fup])  # set xaxis limits from 1 up to user specified fup value in Hz
            ax.set_ylabel('gain phase  [rad]', fontsize=18)  # phase shift of dynamic gain
            ax.set_xlabel('frequency  [Hz]', fontsize=18)  # positive frequencies up to Nyquist

            ax = axes[4, 1]  # specify subfigure position
            colour = iter(plt.cm.bone(np.linspace(0, 1, len(trialNames))))  # specify colours for each line
            for trialName in trialNames:  # each histogram contains data from one sweep
                isi = 1e3*np.load(pathName+tauName+groupName+cellName+'isi/'+trialName+'_isi.npy')  # load interspike interval data from each trial
                ax.hist(isi, bins=50, density=True, histtype='step', color=next(colour))  # create histogram of interspike interval distribution
            ax.set_xlim([0, 500])  # set xaxis limits from 0 to 300 ms
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # set yaxis tick marks to three decimal places
            ax.set_xlabel('isi  [ms]', fontsize=18)  # interspike intervals
            ax.set_ylabel('PDF', fontsize=18)  # [norm] probability distribution function

            ax = axes[5, 1]  # specify subfigure position
            colour = iter(plt.cm.copper(np.linspace(0, 1, len(trialNames))))  # specify colours for each line
            for trialName in trialNames:  # each histogram contains data from one sweep
                isi = np.load(pathName+tauName+groupName+cellName+'isi/'+trialName+'_isi.npy')  # load interspike interval data from each trial
                ax.hist(1/isi, bins=50, density=True, histtype='step', color=next(colour))  # create histogram of instantaneous firing rates
            ax.set_xlim([0, 50])  # set xaxis limits from 0 to 40 Hz
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # set yaxis tick marks to two decimal places
            ax.set_xlabel(r'$\nu_{inst} \ [Hz]$', fontsize=18)  # instantaneous firing rate
            ax.set_ylabel('PDF', fontsize=18)  # [norm] probability distribution function

            plt.savefig(pathName+'figures/'+groupName[:-1]+'_'+cellName[:-1]+'_'+'baseballCard.svg')  # save as a scalable vector graphics file
            plt.savefig(pathName+'figures/'+groupName[:-1]+'_'+cellName[:-1]+'_'+'baseballCard.png', bbox_inches='tight')  # save as a portable network graphics file

    return ()

