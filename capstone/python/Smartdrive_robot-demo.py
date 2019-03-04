#!/usr/bin/python
# coding: utf-8
# Program Name: Samrtdrive_robot-demo.py 
# ===========================                                          
#                                                                      
# Copyright (c) 2014 by openelectrons.com                                
# Email: info (<at>) openelectrons (<dot>) com                           
#                                                                      
# This program is free software. You can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation; version 3 of the License.              
# Read the license at: http://www.gnu.org/licenses/gpl.txt             
#                                                                      
#############################################################################
#
# When        Who             Comments
# 07/8/13    Nitin Patil    Initial authoring.
# 04/21/14   Michael Giles  SmartDrive modification
#
# Python library for fir Openelectrons.com  Accelerometer/Compass Sensor LSM303.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import pygame 
import  sys, os

from SmartDrive import SmartDrive

def move(motor, speed):
    dir = 1
    if(speed <0):
        dir = 0
        speed = speed * -1
    if speed > 100:
        speed  = 100
    SmartDrive.SmartDrive_Run_Unlimited(motor, dir, speed)   


# ===========================================================================
# Example Code
# ===========================================================================


SmartDrive  = SmartDrive()
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init() # main joystick device system

try:
    j = pygame.joystick.Joystick(0) # create a joystick instance
    j.init() # init instance
    print 'Enabled joystick: ' + j.get_name()
except pygame.error:
    print 'no joystick found.'

while True:
    pygame.event.get()
    x1 = 90*j.get_axis(0)
    y1 = 90*j.get_axis(1)
    x2 = j.get_axis(2)
    y2 = j.get_axis(3)
    print " X=  " , x1 
    print " Y=  " , y1
    print " X2=  " , x2 
    print " Y2=  " , y2
    if j.get_button(1):
        print "exit"
        move(1,0)
        move(2,0)
       # sys.exit(1)
    	
    
    lMotor =  x1 + y1
    rMotor =  x1 - y1
    if lMotor > 100 :
        lMotor = 100
    if rMotor > 100 :
        rMotor = 100    
    if lMotor < -100  :
        lMotor = -100
    if rMotor <  -100:
        rMotor = -100    
    
    move(1,lMotor)
    move(2,-rMotor)
    print " lMotor=  " , lMotor
    print " rMotor=  " , rMotor
    
    #time.sleep(.050)
    
