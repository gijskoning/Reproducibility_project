from _pyio import IOBase

import xml.etree.ElementTree as ElementTree


class TrafficLightPhases():
    '''
    Contains all phases of all traffic lights that do not involve yellow.
    Usually read from a file.
    The file follows the SUMO format from
    https://sumo.dlr.de/wiki/Simulation/Traffic_Lights#Defining_New_TLS-Programs
    
    We search for <tlLogic> elements in the XML (can be at any depth) 
    and collect all settings.
    Each tlLogic element must have a unique id (traffic light reference).
    '''
    
    def __init__(self, filename:str):
        '''
        @param filename the file containing XML text. NOTE this really
        should not be a "filename" but a input stream; unfortunately 
        ElementTree does not support this.
        '''
        tree = ElementTree.parse(filename)
        self._phases = {}
        for element in tree.getroot().findall('tlLogic'):
            intersectionid = element.get('id')
            if intersectionid in self._phases:
                raise Exception('file ' + filename + ' contains multiple tlLogic elements with id=' + id)
            
            newphases = []
            for item in element:
                state = item.get('state')
                if 'y' in state or 'Y' in state:
                    continue  # ignore ones with yY: handled by us.
                newphases.append(state)
            self._phases[intersectionid] = newphases
    
    def getIntersectionIds(self) -> list:
        '''
        @return all intersection ids (list of str)
        '''
        return sorted(list(self._phases.keys()))

    def getNrPhases(self, intersectionId:str) -> int:
        '''
        @param intersectionId the intersection id 
        @return number of available phases (int). 
        If n is returned, Phases 0..n-1 are available
        '''
        return len(self._phases[intersectionId])
    
    def getPhase(self, intersectionId:str, phasenr: int) -> str:
        """
        @param intersectionId the intersection id 
        @param phasenr the short number given to this phase
        @return the phase string (eg 'rrGG') for given lightid 
        and phasenr. Usually this
        is the index number in the file, starting at 0.
        """
        return self._phases[intersectionId][phasenr]

