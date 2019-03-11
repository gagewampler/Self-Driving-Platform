#!/usr/bin/env python
#
# Copyright (c) 2014 OpenElectrons.com
# SmartDrive example script.
# for more information about SmartDrive,  please visit:
# http://www.Openelectrons.com/index.php?module=pagemaster&PAGE_user_op=view_page&#PAGE_id=34
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
# History:
# Date      Author        Comments
# 01/28/14  Michael       Initial authoring.
# 02/03/14  Michael       Modified for OpenElectron_i2c
# 03/12/14  Michael       Modified for customer example and comment
# 04/21/14  Michael       SmartDrive modification
#

import time
import os, sys
# import SmartDrive class from SmartDrive Library
from SmartDrive import SmartDrive

# Define run variables
SmartDrive = SmartDrive()
speed = 30
duration = 5
degrees = 90
rotations = 2
tacho = 90
test = 1

# Reset encoder values
SmartDrive.command(SmartDrive.R)



# Forward 5 Seconds
SmartDrive.SmartDrive_Run_Seconds(SmartDrive.SmartDrive_Motor_Both,
                                                  SmartDrive.SmartDrive_Direction_Forward, speed,
                                                  duration, SmartDrive.SmartDrive_Completion_Wait_For,
                                                  SmartDrive.SmartDrive_Next_Action_Brake)
# Backward 5 Seconds
SmartDrive.SmartDrive_Run_Seconds(SmartDrive.SmartDrive_Motor_Both,
                                                  SmartDrive.SmartDrive_Direction_Reverse, speed,
                                                  duration, SmartDrive.SmartDrive_Completion_Wait_For,
                                                  SmartDrive.SmartDrive_Next_Action_Brake)

time.sleep(5)

# Forward 5 Seconds
SmartDrive.SmartDrive_Run_Seconds(SmartDrive.SmartDrive_Motor_Both,
                                  SmartDrive.SmartDrive_Direction_Forward, speed,
                                  duration, SmartDrive.SmartDrive_Completion_Wait_For,
                                  SmartDrive.SmartDrive_Next_Action_Brake)

