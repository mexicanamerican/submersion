#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:39:45 2019

@author: jjj
"""


import numpy as np
import time
from pygame import midi
#%%

class PrintMan():
    def __init__(self, source=None, write_logs=False, dp_log=None, str_append=""):
        self.verbose_level = 0

        if source is None:
            self.pretxt = ""
        else:
            self.pretxt = "{}: ".format(source)
    
        self.use_filter = False
        
        
        # logfile saving
        self.write_logs = write_logs
        self.logs = []
        
    def set_pretxt(self, source):
        self.pretxt = "{}: ".format(source)
        
    def set_filter(self, filter_txt):    
        self.use_filter = True
        self.filter_txt = filter_txt
        
        
    def print_msg(self, msg, verbose_level):
        if self.use_filter and not self.filter_txt in msg:
            return
        if verbose_level >= self.verbose_level:
            print(msg)
        
    def print0(self, msg):
        """Spammy print for debug."""
        self.print_msg("{}{}".format(self.pretxt, msg), 0)
        
    def print1(self, msg):
        """Print stuff that shows program flow."""
        self.print_msg("{}{}".format(self.pretxt, msg), 1)
        
    def print2(self, msg):
        """Print important stuff such as warnings."""
        self.print_msg("{}{}".format(self.pretxt, msg), 2)
        
    def print3(self, msg):
        """Print stuff that user must see!"""
        self.print_msg("================================================================================",3)
        self.print_msg("{}{}".format(self.pretxt, msg), 3)
        self.print_msg("================================================================================",3)
    
    def set_verbosity(self, new_verbose_level):
        print('Deprecated, use set_verbose_level')
        self.verbose_level = new_verbose_level
    
    def set_verbose_level(self, new_verbose_level):
        self.verbose_level = new_verbose_level


class EmoController():
    def __init__(self, allow_fail=True, use_device="midimix", idx_input=None, idx_output=None, verbose_level=2, init_midi=True):
#        pass
        self.use_device = use_device
        self.t_last_init = time.time()
        self.pm = PrintMan("midi_man")
        self.pm.set_verbose_level(verbose_level)
        self.init(allow_fail, idx_input, idx_output, init_midi)
    
    def init(self, allow_fail=True, idx_input=None, idx_output=None, init_midi=True):
        self.device_available = True
        self.megabreaker = False
        self.dt_min_reinit = 10 # Minimum time between reinit attempts if system fails!
        self.has_leds = True
        self.list_occupied = []
        
        self.init_emodisk()
        
        try:
            if init_midi:
                midi.quit()
                midi.init()
                midi.quit()
                midi.init()
                midi.quit()
                midi.init()
            self.devcount = midi.get_count()
    
            self.last_rowcol = None
            self.last_buttontype = None
    
            self.jumpval_saved = np.ones((8,)) * 16
    
            self.midi_keyboard_state = np.zeros((85-36,))
            self.midi_keyboard_event_time = np.zeros((85-36,))
    
            self.dict_caller_id = {}
    
            # train rhytms on led buttons and get them displayed
            self.rhythm_entrainer = False
    
            if self.rhythm_entrainer:
                self.init_rhythm()
            # else:
            #     self.init_buttons()
    
        
            if self.devcount == 0:
                if allow_fail:
                    print("FAIL ON INIT: DEVCOUNT=0")
                else:
                    raise ValueError("FAIL ON INIT: DEVCOUNT=0") 
            else:
                if idx_input is not None or idx_output is not None:
                    self.is_active = True
                    self.akaki = midi.Input(idx_input)
                    self.akako = midi.Output(idx_output)
                    print(f"manually setting idx_input {idx_input} idx_output {idx_output}")
                    
                else:
                    if self.devcount > 1 and self.devcount < 5:
                        self.is_active = True
                        self.akaki = midi.Input(3)
                        self.akako = midi.Output(2)
                    elif self.devcount >= 5:      # sound recorder is in
                        self.is_active = True
                        self.akaki = midi.Input(3)
                        self.akako = midi.Output(2)
                    else:
                        self.is_active = False
                        if not allow_fail:
                            raise ValueError("FAIL ON INIT: DEVCOUNT not supported: {}".format(self.devcount)) 
                        else:
                            print("FAIL ON INIT: DEVCOUNT not supported: {}".format(self.devcount))
                
            self.reset_leds()

        except Exception as e:
            self.is_active = False
            if allow_fail:
                self.pm.print3("FAIL ON INIT! {}".format(e))
            else:
                raise ValueError("FAIL ON INIT! {}".format(e))
            
            self.device_available = False

        if not allow_fail:
            assert hasattr(self, 'akaki'), "FAIL ON INIT: akaki missing!"
            assert hasattr(self, 'akako'), "FAIL ON INIT: akako missing!"


    def map_layout(self, dp, files=None, recursive=True, name='midi_man'):
        pass

    def show_layout(self):
        print(f"layout: {self.list_occupied}")

       
    def init_emodisk(self):
        self.midi_dict = dict()
        self.has_leds = False
        
        # read
        # buttons
        for j in range(6):
            # buttons
            self.midi_dict[(176, 1+j)] = [0,j,2]
            
        # state vars
        self.state = -np.ones((1,6))
        self.event_time = -np.ones((1,6))
        self.event_time_last = -np.ones((1,6))
        self.is_button = np.zeros((1,6)).astype(bool)
        self.is_button[:,:] = True
        
        self.force_refresh_button_leds = False
        

    def reinit(self):
        self.t_last_init = time.time()
        self.init()
        self.auto_register()
        self.pm.print3("reinit PERFORMED")

    def init_rhythm(self):
        self.rhythm_buffer = 3
        if self.use_device=="midimix":
            nmb_buttons = 8
            self.button_time = {}
            for i in range(nmb_buttons):
                self.button_time[0,i] = []
                self.button_time[1,i] = []

            self.button_period = -np.ones((2,nmb_buttons))
            self.button_changed = np.zeros((2,nmb_buttons)).astype(np.bool)
            self.button_flash_discrepancies = 9999999*np.ones((2,nmb_buttons))

        
    def reset_leds(self):
        self.pm.print0(("reset_leds called."))
        if self.use_device == 'midimix':
#            for i in range(2):
#                for j in range(4):
#                    self.set_led(i, j, False)
                    
            self.set_led(1, 8, False)
            self.set_led(2, 8, False)
            self.set_led(3,8,False)
            
            for i in range(8):
                self.set_led(3,i,False)
                self.set_led(4,i,False)
    
                    
        elif self.use_device == 'lpd8':
            for i in range(2):
                for j in range(4):
                    self.set_led(i, j, False)


    def register(self, caller_id, y, x):
        self.dict_caller_id[caller_id] = (y, x)
        
    def auto_register(self):
        if self.use_device == "midimix":
            
            # register all
            rows = ["A","B","C","D","E","F","G","H","I"]
            cols = [0,1,2,3,4,5]
            for ridx,rname in enumerate(rows):
                for c in cols:
                    self.register("{}{}".format(rname,c), c, ridx)
                    
                    
        if self.use_device == "lpd8":
            # register all
            rows = ["A","B","C","D","E","F","G","H"]
            cols = [0,1]
            for ridx,rname in enumerate(rows):
                for c in cols:
                    self.register("{}{}".format(rname,c), c, ridx)
                    
        if self.use_device == "emodisk":
            # register all
            rows = ["A","B","C","D", "E", "F"]
            cols = [0]
            for ridx,rname in enumerate(rows):
                for c in cols:
                    self.register("{}{}".format(rname,c), c, ridx)
                    
        if self.use_device == "deepsee":
            # register all
            rows = ["A","B","C","D","E","F"]
            cols = [0]
            for ridx,rname in enumerate(rows):
                for c in cols:
                    self.register("{}{}".format(rname,c), c, ridx)
                    
            self.reset_state() #reset the knobs to 0 
                    
                    


    def set_led(self, row, col, state):
        if not self.is_active or not self.has_leds:
            return
        dev_coords = self.midi_dict[(row,col)]
        msg = [[[dev_coords[0], dev_coords[1], state, 0], 0]]
        self.akako.write(msg)
        self.pm.print0(("set_led: setting yx {} {}: {}".format(row,col,state)))



    def update(self):
        # try:
            list_last_inp = []
            if self.device_available:
                self.pm.print0("update is called")
                while True:
                  inp = self.akaki.read(1)
                  if inp == []:
                    break
                  else:
                    list_last_inp = inp
                    
                if list_last_inp != []:
                    self.process_last_inp(list_last_inp)
    
                if self.force_refresh_button_leds:
                    for y in range(self.is_button.shape[0]):
                        for x in range(self.is_button.shape[1]):
                            if self.is_button[y,x]:
                                if self.state[y,x] == 1:
                                    self.set_led(y,x,True)
                                else:
                                    self.set_led(y,x,False)
                
    def shock_update(self):
        # print("update")
        list_last_inp = []
        if self.device_available:
            while True:
              inp = self.akaki.read(1)
              if inp == []:
                break
              else:
                list_last_inp = inp
                
            if list_last_inp != []:
                last_inp = list_last_inp[0]
                # print('process_last_inp {}'.format(last_inp))
                if last_inp[0][1] > 35 and last_inp[0][1] < 86:
                    if last_inp[0][0] == 144:
                        self.midi_keyboard_state = np.zeros((85-36,))
                        self.midi_keyboard_state[last_inp[0][1]-36] = 1
                        self.midi_keyboard_event_time[last_inp[0][1]-36] = time.time()
                    else:
                        self.midi_keyboard_state[last_inp[0][1]-36] = 0
                else:
                    list_last_inp[0][0][1] += 15
                    
                    self.process_last_inp(list_last_inp)
                    


    def process_last_inp(self, list_last_inp):
        last_inp = list_last_inp[0]

        self.pm.print1('process_last_inp {}'.format(last_inp))

        # Prog CHNG set! quit
        if last_inp[0][0] == 192:
            self.pm.print3('please set CONTROL MODE TO CC!')
        else:
            if (last_inp[0][0],last_inp[0][1]) in self.midi_dict.keys():
                last_inp_rowcolinfo = [self.midi_dict[(last_inp[0][0],last_inp[0][1])], last_inp[0][2],last_inp[1]]
                self.last_rowcol = tuple(last_inp_rowcolinfo[0][0:2])
    
                value = np.clip(last_inp_rowcolinfo[1] / 127.0, 0, 1)
                
                # print(f'value = {value}')
                
                if last_inp_rowcolinfo[0][2] != 2: #dont set state for button
                    if self.use_device == "deepsee":
                        self.process_encoder(self.last_rowcol, value)
                    else:
                        self.state[self.last_rowcol[0], self.last_rowcol[1]] = value
                    
                self.event_time_last[self.last_rowcol[0], self.last_rowcol[1]] = self.event_time[self.last_rowcol[0], self.last_rowcol[1]]
                self.event_time[self.last_rowcol[0], self.last_rowcol[1]] = time.time()

                # process buttons
                
                if last_inp_rowcolinfo[0][2] == 2:
                    if self.rhythm_entrainer:
                        self.set_timing(last_inp_rowcolinfo)
                    else:
                        self.process_button(last_inp_rowcolinfo)
                        
            # button release event (only works for LPD8)
            if self.use_device == "lpd8":
                try:
                    y = self.last_rowcol[0]
                    x = self.last_rowcol[1]
                    if last_inp[0][2] == 127 and self.button_marked_for_hold[y,x] == 1:
                        self.button_marked_for_hold[y,x] == 0
                        last_inp_rowcolinfo = [self.midi_dict[(144,last_inp[0][1])], last_inp[0][2],last_inp[1]]
                        self.state[y, x] = 0
                        self.set_led(last_inp_rowcolinfo[0][0], last_inp_rowcolinfo[0][1], False)
                except Exception as e:
                    print(f"bad: {e}")

    def process_encoder(self, last_rowcol, value_current):
        # check if there was an increment or decrement. before that check if there was a jump!
        value_last = self.value_last[last_rowcol[0], last_rowcol[1]] 
        # crement = 0.08 #put to self, maybe init
        # crement = 0.1 # plain without abs_diff
        crement = 2.5
        
        # Jumpover case (from 127 to 0 and 0 to 127)
        # print(f"value_current: {value_current}")
        if np.abs(value_current - value_last) > 0.5:
            self.pm.print0("process_encoder: there was a jump!")
            if value_current > value_last: #break through GOING DOWN -> decrement
                abs_diff = np.abs(value_last + 1 - value_current)
                # print(f"GOING DOWN: abs_diff {abs_diff} {value_last} {value_current}")
                state_diff = -crement*abs_diff
            elif value_current < value_last: #break through from pos to neg -> increment
                abs_diff = np.abs(1 - value_last + value_current)
                state_diff = +crement*abs_diff
                # print(f"GOING UP: abs_diff {abs_diff} {value_last} {value_current}")
                
        else:
            abs_diff = np.abs(value_current-value_last)
            if value_current > value_last:
                # print("regular up")
                state_diff = +crement*abs_diff
            else:
                # print("regular down")
                state_diff = -crement*abs_diff
            
            
        if self.use_device == 'deepsee': #hotfix for knob3: invert it
            if last_rowcol[1] == 4:
                state_diff = -state_diff
                
        self.pm.print0(f"process_encoder: state_diff = {state_diff}")
                
        self.value_last[last_rowcol[0], last_rowcol[1]] = value_current
        state_last = self.state[last_rowcol[0], last_rowcol[1]]
        state_new = state_last+state_diff
        state_new = np.clip(state_new, -1, 1)
        self.state[last_rowcol[0], last_rowcol[1]] = state_new

    def process_button(self, last_inp_rowcolinfo):
#        if self.use_device == "midimix":
        self.pm.print0("process_button is called with last_inp_rowcolinfo: {}".format(last_inp_rowcolinfo))
        x = last_inp_rowcolinfo[0][1]
        y = last_inp_rowcolinfo[0][0]
        if self.is_button[y,x]:
            if self.use_device == 'deepsee':
                if last_inp_rowcolinfo[1] == 127:
                    self.state[y,x] = 1
                elif last_inp_rowcolinfo[1] == 0:
                    self.state[y,x] = 0
                self.pm.print0(("process_button: ON is yx {} {}: {}".format(y,x,self.state[y,x])))
            else:
                self.pm.print1("process_button: before state is yx {} {}: {}".format(y,x,self.state[y,x]))
                # button state is 0:
                if self.state[y,x] <= 0:
                    self.state[y,x] = 1
                    self.set_led(y, x, True)
                    self.pm.print0(("process_button: ON is yx {} {}: {}".format(y,x,self.state[y,x])))
                else:
                    self.state[y,x] = 0
                    self.set_led(y, x, False)
                    self.pm.print0(("process_button: OFF  for yx {} {}: {}".format(y,x,self.state[y,x])))
                
    def reset_state(self):
        if self.use_device == "deepsee":
            self.state[:,2:] = 0

    def set_timing(self, last_inp_rowcolinfo):
        if self.use_device == "midimix":
            if last_inp_rowcolinfo[0][0] == 3 or last_inp_rowcolinfo[0][0] == 4:
                x = last_inp_rowcolinfo[0][1]
                y = last_inp_rowcolinfo[0][0] - 3

                # print("y = {}".format(y))



                self.button_time[y,x].append(time.time())
                self.button_changed[y,x] = True
                if len(self.button_time[y,x]) > self.rhythm_buffer:
                    self.button_time[y,x].pop(0)
                if len(self.button_time[y,x]) > 1:
                    self.button_period[y,x] = np.mean(np.diff(self.button_time[y,x]))
                    # self.button_period[x] = self.button_time[x][-1] - self.button_time[x][-2]
                    # self.button_period[x] = 1


    def get_value(self, caller_id=None, val_min=0, val_max=1, val_default=None, mode=None):
        if not caller_id in self.dict_caller_id.keys():
            self.pm.print3("midiman get_value: caller_id {} not registered".format(caller_id))
            return
        y, x = self.dict_caller_id[caller_id]
        
        if caller_id not in self.list_occupied:
            self.list_occupied.append(caller_id)
            self.list_occupied.sort()
            
        
        # process button
        if self.is_button[y,x]:
            # print("get_value is y x {} {}: {}".format(y,x,self.state[y,x]))
            
            if self.use_device == "lpd8":
                if mode == "hold" and self.state[y,x] <= 0 and self.button_marked_for_hold[y,x] == 0:
                    self.button_marked_for_hold[y,x] = 1
                    
                if mode != "hold":
                    self.button_marked_for_hold[y,x] = 0
            
            if self.state[y,x] <= 0:
                value_return = False
            else:
                if mode == "switch":
                    self.state[y,x] = -1
                    self.set_led(y, x, False)
                    
                value_return = True
            
            if self.use_device == "midimix":
                self.saved_akaki_grid[ord(caller_id[0])-65, int(caller_id[1])] = int(value_return)
                
            return value_return
            
        
        # sliders knobs
        else:
            if val_default is None:
                val_default = 0.5*(val_min+val_max)
            
            if (x is not None and y is not None and val_min is not None and val_max is not None and val_default is not None):
                if self.use_device == "deepsee":
                    return self.state[y,x]
                else:
                
                    if self.state[y,x] == -1:
                        value_return = val_default
                    else:
                        value_return = val_min + (val_max-val_min)*self.state[y,x]
                    
                # print("midiman value_return {}".format(value_return))
                
            if mode=="int":
                value_return = int(value_return)
                
            if self.use_device == "midimix":
                self.saved_akaki_grid[ord(caller_id[0])-65, int(caller_id[1])] = self.state[y,x]
            
            return value_return
        
        
    def set_value(self, caller_id, value):
        if not caller_id in self.dict_caller_id.keys():
            self.pm.print3("set_value: caller_id {} not registered".format(caller_id))
            return
        
        y, x = self.dict_caller_id[caller_id]
        assert self.is_button[y,x], "currently only buttons are allowed. please contact support"
        
        # print("get_value is y x {} {}: {}".format(y,x,self.state[y,x]))
        if value == True:
            self.state[y,x] = 1
            self.set_led(y, x, True)
        elif value == False:
            self.state[y,x] = 0
            self.set_led(y, x, False)
        else:
            self.pm.print3("set_value: value {} unsupported. Needs to be True/False".format(value))
        
    def set_values_from_saved_grid(self):
        for c in range(self.saved_akaki_grid.shape[0]):
            for d in range(self.saved_akaki_grid.shape[1]):
                caller_id = chr(65+c)+str(d)
                if self.saved_akaki_grid[c,d] != -9999:
                    y, x = self.dict_caller_id[caller_id]
                    if self.is_button[y,x]:
                        self.set_value(caller_id, self.saved_akaki_grid[c,d])
                    else:
                        self.state[y,x] = self.saved_akaki_grid[c,d]
            


# for i in range(1000):
#%%
if __name__ == '__main__':
    print("WARNING! MIDI MAN DEUBGGING IS RUNNING")
    
    midi_man = EmoController(use_device='emodisk', allow_fail=False,verbose_level=2)
    midi_man.auto_register()
    midi_man.pm.set_verbose_level(3)
    # self.megabreaker=True
    # #
    while True:
        time.sleep(0.05)
        midi_man.update()
    
        for key in ['A','B','C','D','E','F']:
            val = midi_man.get_value(f"{key}0", 'hold')
            
            if val:
                print(key)
    






