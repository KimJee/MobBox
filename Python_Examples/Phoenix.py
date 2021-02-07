from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #1: Run simple mission

from builtins import range

import random

try:
    import MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython

import malmoutils
import errno

"""
    Custom imports
""" 
import os
import sys
import time


# try:
#     agent_host.parse(sys.argv)
# except RuntimeError as e:
#     print('ERROR:', e)
#     print(agent_host.getUsage())
#     exit(1)
# if agent_host.receivedArgument("help"):
#     print(agent_host.getUsage())
#     exit(0)


"""
    Global Variables
"""
ARENA_SIZE = 20
MOB_TYPE = "Chicken"
ENTITY_DENSITY = 0.02
spawn_end_tag = ' type="' + MOB_TYPE + '"/>'


def spawn_mob():
    entities = "" # Output XML String
    for x in range(-ARENA_SIZE,ARENA_SIZE):
        for z in range(-ARENA_SIZE,ARENA_SIZE):
            if random.random() < ENTITY_DENSITY:
                entities += f"<DrawEntity x='{x}' y='2' z='{z}'" + spawn_end_tag
    return entities

def get_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>CS175 Test</Summary>
                </About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>12000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <AllowSpawning>true</AllowSpawning>
                        <AllowedMobs>''' + MOB_TYPE + '''</AllowedMobs>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>''' + \
                        spawn_mob() + \
                        '''</DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Spectator">
                    <Name>CS175Test</Name>
                    <AgentStart>
                        <Placement x="0" y="2" z="0" pitch="0" yaw="0"/>
                        <Inventory>
                        <InventoryItem slot="0" type="diamond_pickaxe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                              <VideoProducer>
                                <Width>860</Width>
                                <Height>480</Height>
                              </VideoProducer>
                              <ColourMapProducer>
                                    <Width>860</Width>
                                    <Height>480</Height>
                                </ColourMapProducer>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


# var recordedFileName = $"./mission_data{DateTime.Now:hh mm ss}.tgz";
# var missionRecord = new MissionRecordSpec(recordedFileName);
# missionRecord.recordCommands();
# missionRecord.recordMP4(20, 400000);
# missionRecord.recordRewards();
# missionRecord.recordObservations();
# _agentHost.startMission(mission, missionRecord);

from datetime import datetime



def run():
    agent_host = MalmoPython.AgentHost()                  # Create our agent_host
    malmoutils.parse_command_line(agent_host)             # Emulates the commandline functionlity from team_reward_test
    my_xml = get_xml()                                    # Grabs the xml "environment-settings"
    my_mission = MalmoPython.MissionSpec(my_xml, True)    # Describes the mission specifications
    my_mission.timeLimitInSeconds(15)

    date_time_string = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    print(date_time_string)
    #file_name = "\\Users\\Jee\\CS175Project\\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7 (1)\\Github\\Phoenix\\Python_Examples\\mission_records\\" + date_time_string + ".tgz"
    #print(file_name)
    my_mission_record = MalmoPython.MissionRecordSpec("./video/" + date_time_string + ".tgz")

    my_mission_record.recordRewards()
    my_mission_record.recordCommands()
    my_mission_record.recordObservations()

    my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)
    my_mission_record.recordMP4(MalmoPython.FrameType.VIDEO, 24, 2000000, False)


    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                2
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission running ", end=' ')

    # Loop until mission ends:
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(0.1)

        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission ended")
    # Mission has ended.

if __name__ == "__main__":
    run()
