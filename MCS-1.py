MCS-1.py
Not shared
Type
Text
Size
34 KB (34,507 bytes)
Storage used
34 KB (34,507 bytes)
Location
mcs-ai
Owner
me
Modified
Feb 12, 2021 by me
Opened
8:16 PM by me
Created
7:08 PM with Google Drive Web
Add a description
Viewers can download
# Monte Carlo Simulation - Pandemic Outbreak - 0
from tkinter import *
from PIL import Image
import cv2
# @title Default title text
# simulation
# 1 : up
# 2 : left
# 3 : down
# 4 : right
import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math

from gym.utils import seeding

print("Start...")
SIZE_L = 50
SIZE_B = SIZE_L
# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255),
     4: (28, 25, 43),
     5: (115, 21, 27),
     6: (255,255,255)}





import threading
def start_another_thread(submit):
            global submit_thread
            submit_thread = threading.Thread(target=submit)
            submit_thread.daemon = True
            submit_thread.start()


mw = Tk()
mw.title('MonteCarloSimulation-0')
#mw.geometry("900x500")
mw.configure(bg="#2B2B2B")

def submit():
    count = pc.get()
    print('submit',count)
    credentials = []
    credentials.append(count)
    credentials.append(Size.get())
    credentials.append(Lockdown.get())
    credentials.append(expansion.get())
    credentials.append(days.get())
    credentials.append(NOI.get())
    credentials.append(MED.get())
    credentials.append(ST.get())
    credentials.append(INT.get())
    credentials.append(DR.get())
    credentials.append(str(var_bot.get()))

    with open(f'MCS-0-credentials.txt', 'w') as f:
        for i in credentials:
            f.writelines(i + "\n")
    mw.destroy()

Label(mw,text='MCS-Pandemic Outbreak',font=("bold",30),bg="#2B2B2B",fg="#bebebe").grid_configure(row=0,column=0,
                                                                            columnspan=4,pady=30,padx=100,ipadx=100)
pc = Entry(mw)#population count
pc.grid_configure(row=1,column=1,ipadx=10)
Label(mw,text="Population Count : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=1,column=0,pady=5)
pc.insert(0,200)

Size = Entry(mw)#Size
Size.grid_configure(row=2,column=1,ipadx=10)
Size.insert(0,50)
Label(mw,text="Size : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=2,column=0,pady=5)

Lockdown = Entry(mw)#population count
Lockdown.grid_configure(row=3,column=1,ipadx=10)
Lockdown.insert(0,0)
Label(mw,text="Lockdown %: ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=3,column=0,pady=5)


expansion = Entry(mw)#Vaccine
expansion.grid_configure(row=6,column=1,ipadx=10)
expansion.insert(0,1)
Label(mw,text="HAD/VAD-expansion<SIZE : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=6,column=0,pady=5)

days = Entry(mw)#Vaccine
days.grid_configure(row=7,column=1,ipadx=10)
days.insert(0,100)
Label(mw,text="Days : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=7,column=0,pady=5)

NOI = Entry(mw)#Vaccine
NOI.grid_configure(row=8,column=1,ipadx=10)
NOI.insert(0,1)
Label(mw,text="NOI : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=8,column=0,pady=5)

MED = Entry(mw)#Vaccine
MED.grid_configure(row=1,column=3,ipadx=10)
MED.insert(0,5)
Label(mw,text="MED : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=1,column=2,pady=5,padx=5) #DAYS AFTER MEDICINE WORKS: INFECTION OUT

ST = Entry(mw)#Vaccine
ST.grid_configure(row=2,column=3,ipadx=10)
ST.insert(0,20)
Label(mw,text="ST : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=2,column=2,pady=5,padx=5) #SELF TREATMENT

INT = Entry(mw)#Vaccine
INT.grid_configure(row=3,column=3,ipadx=10)
INT.insert(0,1)
Label(mw,text="INT : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=3,column=2,pady=5,padx=5) #SELF TREATMENT

DR = Entry(mw)#Vaccine
DR.grid_configure(row=4,column=3,ipadx=10)
DR.insert(0,1000)
Label(mw,text="DR : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=4,column=2,pady=5,padx=5) #SELF TREATMENT

var_bot=IntVar()
Checkbutton(mw, text="AI",variable=var_bot,font=("courier"),bg="#2B2B2B",fg="#bebebe").grid(row=5,column=2, sticky=W,pady=5,padx=5)

Button(mw,text="Start",command = submit,bg="red").grid_configure(row=9,column=1,ipadx=20,pady=5)
mw.mainloop()




credits = []
with open(f'MCS-0-credentials.txt', 'r') as o:
        up = o.readlines()
        for s in up:
            print(s)
            credits.append(s)

count=credits[0]
SIZE_L=int(credits[1])
SIZE_B=int(credits[1])
Lockdown = int(credits[2])
expansion = int(credits[3])
Days = int(credits[4])
NOI = int(credits[5])
Med = int(credits[6])
ST = int(credits[7])
interval = int(credits[8])
death_rate=int(credits[9])
Bot=bool(credits[10])
# print(en)
class Person:
    def __init__(self, env, x=None, y=None, ):

        self.x = x
        self.y = y
        self.subject_sign = d[1]
        self.contact = False
        self.immune = False
        self.env = env
        self.env[x][y] = self.subject_sign
        self.treatment_step = 0
        self.medicines = Med
        self.current_step = 0
        self.treatment = False
        self.income = 0
        self.prev_post = None
        self.earning = random.randint(1,10)
        self.infected_times = 0
        self.infected_days = None
        self.infection_period = ST
        self.death = False
        self.infection_rate=IR
        self.temp = 0
        self.reported = False
        # print(self.env)

    def step_update(self, step,IR):
        self.current_step = step
        self.infection_rate=IR
        if self.treatment is True:

            if self.current_step > self.treatment_step:
                self.contact = False
                self.treatment = False
        if self.reported:
            self.reported=False
            yum = self.temp
            self.temp = 0
            return  yum,self.infection_rate
        else:return 0,self.infection_rate

    def move(self, direct, infected=None, contagion=False):
        if not self.treatment:self.subject_sign=d[1]
        if self.immune:self.subject_sign=d[6]
        prev_state = self.contact
        if direct == 1:  # left
            if self.x == 0:
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x - 1][self.y] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.x = self.x - 1

        elif direct == 2:  # up
            if self.y == 0:
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x][self.y - 1] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.y = self.y - 1

        elif direct == 3:  # right
            if self.x == (SIZE_B - 1):
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x + 1][self.y] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.x = self.x + 1

        elif direct == 4:  # down
            if self.y == (SIZE_L - 1):
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x][self.y + 1] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.y = self.y + 1

        elif direct == 6:  # right-down
            if self.x == (SIZE_B - 1):
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x + 1][self.y] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.x = self.x + 1

            if self.y == (SIZE_L - 1):
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x][self.y + 1] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.y = self.y + 1


        elif direct == 7:  # right-up
            if self.x == (SIZE_B - 1):
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x + 1][self.y] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.x = self.x + 1

            if self.y == 0:
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x][self.y - 1] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.y = self.y - 1

        elif direct == 8:  # left-up
            if self.x == 0:
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x - 1][self.y] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.x = self.x - 1

            if self.y == 0:
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x][self.y - 1] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.y = self.y - 1

        elif direct == 9:  # left-down
            if self.x == 0:
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x - 1][self.y] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.x = self.x - 1

            if self.y == (SIZE_L - 1):
                self.env[self.x][self.y] = self.subject_sign
            else:
                self.env[self.x][self.y + 1] = self.subject_sign
                self.env[self.x][self.y] = 0
                self.y = self.y + 1

        elif direct == 5:  # No MOvement
            pass

        elif direct == 5:  # No MOvement
            pass
        duplicate = []
        # print(posi)
        # if (self.x,self.y) in posi:posi.remove((self.x,self.y))
        for a in infected:
            if a == (self.x, self.y) and random.random()<=self.infection_rate:
                duplicate.append(a)
                if self.contact: self.temp += 1
        self.contagion = contagion

        if self.contact == False and contagion == True and infected.count((self.x,self.y))>=1 and self.immune == False and self.treatment is False:  # steps to contact
            self.contact = True
            self.subject_sign = '4'

            # print("posi--",posi,"Duplicate : ",duplicate,"\n\n")
            # print("Virus}|}}{{}{{}}{}{{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{{}{}{}{}{}{}{}{}{}")

        if self.contact == True and prev_state == False:
            # if self.contact:print(self.infected_days,self.current_step)
            self.infected_days = self.current_step + self.infection_period
            self.infected_times = self.infected_times + 1

        if self.infected_days != None and self.contact == True and self.current_step > self.infected_days:
            # print(self.infected_days)
            self.contact = False
            self.reported = True
            # print('Self Treatment')
            self.infected = None
            if random.random() < 1 / death_rate:
                self.death = True

        xi = [p[0] for p in infected]
        yi = [p[1] for p in infected]
        y=0
        for x in xi:
            self.env[x][yi[y]] = d[3]
            y = y + 1
        return self.env, self.x, self.y, self.contact

    def vacinated(self, x1=8, x2=12, y1=8, y2=12):
        if self.x >= x1 and self.x <= x2 and self.y >= y1 and self.y <= y2:
            self.contact = False
            self.immune = True
            self.subject_sign = d[6]
        return self.immune

    def hospital(self, x1, x2, y1, y2):
        if self.x >= x1 and self.x <= x2 and self.y >= y1 and self.y <= y2:
            if self.immune == False and self.contact == True and self.treatment == False:
                self.subject_sign = d[2]
                self.treatment = True
                self.treatment_step = self.current_step + self.medicines
                # print("Treatment..............................\t\t\t\t\t\t.........")

    def Earning(self, TP, AC, D, LP):

        ########### important here ##########

        if ((self.x, self.y) is not self.prev_post) and self.contact == True:
             return self.income + self.earning - self.earning * (
                        0.62 * (math.log(1 + (AC / TP), 10) + 1) - math.log(1 + (D / TP),
                                                                        2.71828)) - self.earning * 0.2 - self.earning * LP

        elif ((self.x, self.y) is not self.prev_post) and self.contact == False:
             return self.income + self.earning - self.earning * (
                        0.62 * (math.log(1 + (AC / TP), 10) + 1) - math.log(1 + (D / TP), 2.71828)) - self.earning * LP

        else:return self.income

    def Death(self, ):
        '''if random.random() < self.infected_times/100000:
          self.death=True'''

        if random.randrange(0, 200000) <= self.infected_times and self.contact:
            self.death = True
        return self.death

    def details_pop(self):
        return self.infection_period


"""
def step(psn=None,posts=None):
    ipt = random.randint(1,4)
    if ipt == 1:
       env,x,y,state=psn.move(1,posts)
    elif ipt == 2:
      env,x,y,state= psn.move(2,posts)
    elif ipt == 3:
      env,x,y,state=psn.move(3,posts)
    elif ipt == 4:
      env,x,y,state= psn.move(4,posts)
    return env,x,y,state"""

maps=[]
class simulation_control:
    def __init__(self, NOP=None, env=None):
        self.NOP = NOP
        self.immune_code = d[5]
        self.hospital_code = d[4]
        self.contact = 0
        self.contact_list = []
        self.vacinated = []
        self.steps = []
        self.immune_steps = []
        self.prev_contacts = 0
        self.cnt = []
        self.total_recovered = []
        self.immune = []
        self.prev_immune = 0
        self.infected = []
        self.current_step = 1
        self.new_case_list = []
        self.recovery_list = []
        self.new_case_steps = []
        self.recovery_steps = []
        self.total_recovered_cases = 0
        self.Total_Economy_perstep = 0
        self.Total_income_list = []
        self.current_step_lst = []
        self.lockdown_percent = 0
        self.lockdown_list = []
        self.dead_people_list = []
        self.dead_people = 0
        self.death_steps = []
        self.data_1 = [0,0,0,0,0]
        self.datalst = [0,0,0,0,0,0]
        self.report = []
        self.reports_num = 0
        self.ir =0

        self.SIZE_L = SIZE_L
        self.en = np.zeros((SIZE_L, SIZE_B, 3), dtype=np.uint8)
        self.data = []
        self.reward = 0
        self.death_reward = -70
        self.new_cases_reward = -35
        self.new_recovery_reward = 70
        self.done = False
        self.Deaths = 0
        self.AC = 0
        self.TP = 0
        self.IF = 0
        self.per_day_reward = 600
        self.NOI = 0
        self.LOP_=[]

    def population_initalize(self):
        df = pd.DataFrame()
        df['Day'] = []
        df['Population Count'] = []
        df['Active Cases'] = []
        df['New Cases'] = []
        df["Recovery"] = []
        df["Total Death"] = []
        df.to_csv("{}".format("mcs-0-record.csv"))
        IP = 0  # initial population
        LOP = []  # list of people
        while IP < int(self.NOP):
            person = Person(self.en, x=random.randint(0, SIZE_B - 1), y=random.randint(0, SIZE_L - 1))
            self.IF = person.details_pop()
            LOP.append(person)

            self.vacinated.append(person.immune)
            self.report.append(0)
            IP = IP + 1
        self.working_list = LOP
        self.LOP = LOP
        self.TP = len(self.LOP)
        self.LOP_=LOP
        return LOP, self.en

    def location_registry(self):
        posts = []
        for person in self.LOP:
            posts.append((person.x, person.y))
        # print(posts)
        return posts

    def step(self, psn, infected=None, contagion=False, no_movement=False):
        ipt = random.randint(1, 9)
        if no_movement is True:
            ipt = 5
        if ipt == 1:
            env, x, y, state = psn.move(1, infected, contagion)
        elif ipt == 2:
            env, x, y, state = psn.move(2, infected, contagion)
        elif ipt == 3:
            env, x, y, state = psn.move(3, infected, contagion)
        elif ipt == 4:
            env, x, y, state = psn.move(4, infected, contagion)
        elif ipt == 5:
            env, x, y, state = psn.move(5, infected, contagion)
        elif ipt == 6:
            env, x, y, state = psn.move(6, infected, contagion)
        elif ipt == 7:
            env, x, y, state = psn.move(7, infected, contagion)
        elif ipt == 8:
            env, x, y, state = psn.move(8, infected, contagion)
        elif ipt == 9:
            env, x, y, state = psn.move(9, infected, contagion)
        return env, x, y, state

    def population_movement(self,IR):  # ONe MOVEMENT
        self.contact_list.clear()
        if len(self.infected) == 0:
            contagion = False
        else:
            contagion = True
        # print("check...","\n",self.en)
        self.infected.clear()


        for person in self.lockdown_list:
            a,self.ir = person.step_update(self.current_step,IR)
            if person in self.LOP:self.report[self.LOP_.index(person)]=a
            env, x0, y0, state0 = self.step(person, self.infected, contagion, no_movement=True)
            if person.Death() == True:
                self.reward = self.reward + self.death_reward  # death-reward
                self.lockdown_list.remove(person)

            if person in self.LOP:
                if state0:
                    self.infected.append((x0, y0))
                    self.contact_list.append(state0)
                self.Total_Economy_perstep = self.Total_Economy_perstep + person.Earning(self.TP, self.AC, self.Deaths,
                                                                                         self.lockdown_percent)
            else:
                if (x0, y0) in self.infected: self.infected.remove((x0, y0))

                # print(self.contact_list)

        for person in self.working_list:
            a,self.ir = person.step_update(self.current_step, IR)
            if person in self.LOP: self.report[self.LOP_.index(person)] = a
            env, x0, y0, state0 = self.step(person, self.infected, contagion, no_movement=False)
            if person.Death() == True:
                self.reward = self.reward + self.death_reward  # death-reward
                self.working_list.remove(person)

            if person in self.LOP:
                if state0:
                    self.infected.append((x0, y0))
                    self.contact_list.append(state0)
                self.Total_Economy_perstep = self.Total_Economy_perstep + person.Earning(self.TP, self.AC, self.Deaths,
                                                                                         self.lockdown_percent)
            else:
                if (x0, y0) in self.infected: self.infected.remove((x0, y0))

        hud=self.current_step%interval
        set = [0]
        for u in range(0,int(interval/2)):
            set.append(u)
        if hud in set:
            if display:self.display()


    def vacination(self, LOP, x1, x2, y1, y2):
        for u in range(y1, y2):
            self.en[x1][u] = self.immune_code
            for p in range(x1, x2):
                self.en[p][u] = self.immune_code

        for person in self.LOP:
            if person.vacinated(x1, x2, y1, y2) == True and self.vacinated[self.LOP.index(person)] == False:
              self.vacinated[self.LOP.index(person)] = True
        return self.en

    def hospitalization(self, LOP, x1, x2, y1, y2):
        for u in range(y1, y2):
            self.en[x1][u] = self.hospital_code
            for p in range(x1, x2):
                self.en[p][u] = self.hospital_code
        for person in self.LOP:
            person.hospital(x1, x2, y1, y2)

    def initialize_infection(self,n):
        self.NOI = n
        for r in range(0, self.NOI):
            rand = random.choice(self.LOP)
            rand.contact = True

    def economy_tracker(self):
        self.reward = self.reward - self.Total_Economy_perstep / 1000  # economy-reward   # 2.xxx for 40
        self.Total_income_list.append(self.Total_Economy_perstep)

    def display(self):
        img = Image.fromarray(self.en, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = np.array(img)
        if self.current_step==1 and SIZE_B>=100:
            img = cv2.putText(img, 'MCS-0', org=(int(SIZE_B/4), SIZE_B - int(SIZE_B / 3)), fontFace=cv2.FONT_ITALIC,

                              fontScale=0.4 * SIZE_B / 100, color=(255, 0, 0))

        if SIZE_B>=100:img = cv2.putText(img, '{}'.format(self.current_step),
                          org=(int(SIZE_B/10), int(SIZE_B/10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.3 * SIZE_B / 100,
                          color=(0, 0, 255))




        maps.append(img)

        if display:
            img = Image.fromarray(img, 'RGB')
            img = img.resize((600, 600))  # resizing so we can see our agent in all its glory.
            cv2.imshow("MonteCarloSimulation-0", np.array(img))  # show it!
            cv2.waitKey(1)
        if not display:cv2.destroyWindow("MonteCarloSimulation-0")
        #print(self.en, "\n\n")

    def lockdown(self, percent=0):
        self.lockdown_percent = percent / 100
        self.lockdown_list = random.sample(self.LOP, int(len(self.LOP) * self.lockdown_percent))
        self.working_list = []
        for sample in self.LOP:
            if sample not in self.lockdown_list:
                self.working_list.append(sample)
        if percent is not 0:print("+++++++++++++++++++++++++++++++++++ {} LOCKDOWN +++++++++++++++++++++++++++++++++++".format(percent))

    def save_to_csv(self,fileN="mcs-0-record.csv"):
        datalst=self.datalst
        d1 = datalst[0]
        d2 = datalst[1]
        d3 = datalst[2]
        d4 = datalst[3]
        d5 = datalst[4]
        d6 = datalst[5]

        user_list = pd.read_csv('{}'.format(fileN))
        FG = list(user_list['Day'])
        CG = list(user_list['Population Count'])
        LG = list(user_list['Active Cases'])
        F = list(user_list["New Cases"])
        T = list(user_list["Recovery"])
        P = list(user_list["Total Death"])

        FG.append(d1)
        CG.append(d2)
        LG.append(d3)
        F.append(d4)
        T.append(d5)
        P.append(d6)

        df = pd.DataFrame()
        df['Day'] = FG
        df['Population Count'] = CG
        df['Active Cases'] = LG
        df['New Cases'] = F
        df["Recovery"] = T
        df["Total Death"] = P

        df.to_csv("{}".format(fileN))

    def non_display_stats(self, y, Limit):
        # print(len(self.LOP),len(self.lockdown_list),len(self.working_list))
        global contact
        self.current_step = y + 1
        self.current_step_lst.append(y)
        self.new_case = 0
        recovered_cases = 0
        u = 0

        if u == 0:
            contact = 0
            imune = []
            immune = 0

            self.LOP = []
            for e in self.lockdown_list:
                self.LOP.append(e)
            for ee in self.working_list:
                self.LOP.append(ee)
            contact = len(self.contact_list)
            for c in self.contact_list:
                self.reward = self.reward + self.new_cases_reward  # new_cases_reward

            if contact > self.prev_contacts:
                self.steps.append(y)
                self.cnt.append(contact)
                self.new_case = contact - self.prev_contacts
                self.new_case_list.append(self.new_case)
                self.new_case_steps.append(y)

            elif contact < self.prev_contacts:
                self.steps.append(y)
                self.cnt.append(contact)
                recovered_cases = self.prev_contacts - contact
                self.total_recovered_cases = self.total_recovered_cases + (self.prev_contacts - contact)
                self.total_recovered.append(self.total_recovered_cases)
                self.recovery_list.append(recovered_cases)
                self.recovery_steps.append(y)

            self.prev_contacts = contact
            for i in self.total_recovered: self.reward = self.reward + self.new_recovery_reward  # recovery-reward
            for immune_state in self.vacinated:
                if immune_state: immune += 1

            if immune > self.prev_immune:
                self.immune_steps.append(y)
                d = immune - self.prev_immune
                if d != 0: self.immune.append(immune)
            self.prev_immune = immune
            Economy_state = self.Total_Economy_perstep
            self.Total_Economy_perstep = 0

        if (self.dead_people - len(self.LOP)) != 0:
            self.death_steps.append(y)
            self.dead_people_list.append(self.dead_people - len(self.LOP))
        self.dead_people = len(self.LOP)

        if contact <= self.NOI and len(self.recovery_list) != 0:
            self.done = True
            for person in self.LOP:
                if person.contact == True:
                    person.contact = False
        # print(self.contact_list)
        # print(self.dead_people_list,len(self.LOP))
        Total_deaths = 0
        Total_death_list = []
        for deaths in self.dead_people_list[1:]:
            Total_deaths = deaths + Total_deaths
            Total_death_list.append(Total_deaths)

        p = 0
        l=0
        for i in self.report:
            if i !=0:l+=1
            p = p + i
        if l!=0:b= int(p / l)
        else: b=0
        #print("------", p)
        if b!=0:self.reports_num=b
        print("DAYS : ", y, "ToTal POpulation : ", len(self.LOP), "Active Cases : ", contact, "New Cases : ",
              self.new_case, "Recovery : ", recovered_cases, "IMMUNE : ", immune, "Deaths : ", Total_deaths,"contacts/blob avg: ",self.reports_num*self.ir*2)

        self.data_1 = [contact,self.new_case,immune,Total_deaths,recovered_cases]
        self.datalst = [y,len(self.LOP),contact,self.new_case,recovered_cases,immune,Total_deaths]
        self.save_to_csv()
        if y == Limit:

            # fig,ax = plt.subplots()
            # print(self.recovery_steps,self.total_recovered)
            print(self.cnt)
            plt.plot(self.steps, self.cnt, 'red')
            #plt.plot(self.recovery_steps,self.total_recovered,'green')
            plt.plot(self.death_steps[1:], Total_death_list, color="Black")

            #plt.plot(self.immune_steps,self.immune,'blue')
            plt.plot(self.new_case_steps,self.new_case_list,color="pink")
            plt.plot(self.recovery_steps,self.recovery_list,color="orange")
            # plt.set(xlabel='STeps', ylabel='PoPulation',
            # title='MonteCarloSimulation')
            plt.grid()
            plt.show()

            print(self.Total_income_list)
            TC = []
            for income in self.Total_income_list:
                TC.append(income)
            plt.plot(self.current_step_lst[1:], TC, color="green", linestyle="dashed")
            plt.grid()
            plt.show()
        self.per_day_reward = self.per_day_reward - 1
        self.reward = self.reward + self.per_day_reward
        # print(self.reward)
        self.data = np.array(
            [y, len(self.LOP) / 100, contact, self.new_case, recovered_cases, immune, Total_deaths, Economy_state / 10,
             SIZE_L / 100, self.IF])
        return self.data, self.reward, self.done


mawa = Tk()
mawa.title("MonteCarloSimulation-0-ControlPanel")
mawa.configure(bg="#2B2B2B")
########################################################################################################################

var_2 = IntVar()
scale_2 = Scale(mawa,variable=var_2,activebackground="#65ff00",cursor="dot",bg="#2B2B2B",fg="#bebebe",orient=HORIZONTAL
                  ,sliderlength=30,length=138,to=Days*10,troughcolor="grey")
scale_2.set(Days)
scale_2.grid_configure(row=1,column=1)
Label(mawa,text="Days : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=1,column=0)


var_1 = IntVar()
scale_1 = Scale(mawa,variable=var_1,activebackground="#65ff00",cursor="dot",bg="#2B2B2B",fg="#bebebe",orient=HORIZONTAL
                  ,sliderlength=30,length=138,to=100,troughcolor="grey")
scale_1.grid_configure(ipady=2,pady=2,row=2,column=1)
scale_1.set(0)
Label(mawa,text="Lockdown% : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=2,column=0)

var_3 = IntVar()
scale_3 = Scale(mawa,variable=var_3,activebackground="#65ff00",cursor="dot",bg="#2B2B2B",fg="#bebebe",orient=HORIZONTAL
                  ,sliderlength=30,length=138,to=SIZE_B-1,troughcolor="grey")
scale_3.grid_configure(ipady=2,pady=2,row=3,column=1)
scale_3.set(10)
Label(mawa,text="expansion : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=3,column=0)

var_4 = IntVar()
scale_4 = Scale(mawa,variable=var_4,activebackground="#65ff00",cursor="dot",bg="#2B2B2B",fg="#bebebe",orient=HORIZONTAL
                  ,sliderlength=30,length=138,from_=0.000,to=1.000,digit=4,resolution=0.001,troughcolor="grey")
scale_4.grid_configure(ipady=2,pady=2,row=4,column=1)
scale_4.set(0.2)
Label(mawa,text="InfectionRate : ",font=("courier"),bg="#2B2B2B",fg="#bebebe").grid_configure(row=4,column=0)
IR=scale_4.get()
var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
var1.set(0.01)
display=True
hospital=False
vaccine=False
def set_display():
    global  display
    if var1.get()==0:display=False
    else: display=True
def hospitals_control():
    global hospital
    if var2.get()==1:hospital=True
    else: hospital=False
def vaccine_control():
    global vaccine
    if var3.get()==1:vaccine=True
    else:vaccine=False
Checkbutton(mawa, text="Hospitalization", command= hospitals_control,variable=var2,font=("courier"),bg="#2B2B2B",fg="#bebebe").grid(row=6,column=1, sticky=W)
Checkbutton(mawa, text="Vacination", command= vaccine_control,variable=var3,font=("courier"),bg="#2B2B2B",fg="#bebebe").grid(row=7,column=1, sticky=W)
Checkbutton(mawa, text="Display", command= set_display,variable=var1,font=("courier"),bg="#2B2B2B",fg="#bebebe").grid(row=5,column=1, sticky=W)
########################################################################################################################
read=False


class PandemicOutbreak:
    def __init__(self):
        self.action_space = np.array([r for r in range(101)])
        self.observation_space = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # self.observation_space = gym.spaces.Discrete(6)
        self.SC = None
        self.PT = None
        self.day = None
        self.Limit = None

    def reset(self):
        self.SC = simulation_control(count)
        self.PI, state = self.SC.population_initalize()
        print("POpulation INITIALIZING...")
        self.day = 0
        self.Limit = var_2.get()
        self.reward = 0
        new_state, reward, done = self.SC.non_display_stats(self.day, self.Limit)
        return new_state

    def step(self, action):
        expansion = scale_3.get()
        self.Limit=scale_2.get()
        self.day += 1
        if scale_1.get()==0:self.SC.lockdown(action)
        else:self.SC.lockdown(scale_1.get())
        if self.day == 2: self.SC.initialize_infection(NOI)
        for t in range(0, 30):
            self.SC.population_movement(scale_4.get())
            if hospital: self.SC.hospitalization(self.PI, 0, expansion, 0, expansion)
            if vaccine : self.SC.vacination(self.PI, 0, expansion, 0, expansion)
        self.SC.economy_tracker()
        done = False
        if self.day % 1 == 0: new_state, reward, done = self.SC.non_display_stats(self.day, self.Limit)
        if self.day == self.Limit:
            done = True

        return new_state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

def MCS_1():
    from DDQN_MCS_PC import DeepNetworkAgent
    env = PandemicOutbreak()
    print("-=-=-", env.observation_space.shape, env.action_space)
    print(env.observation_space.shape[0])
    discrete = False
    if discrete == False:
        state = 1
        act_state = 1
        for i in range(len(env.observation_space.shape)):
            state *= env.observation_space.shape[i]
        for i in range(len(env.action_space.shape)):
            act_state *= env.action_space.shape[i]
        print(state, act_state)
        DeepNetworkAgent(state, len(env.action_space),env)

def MCS_0():
    global read
    print('count',count)
    SC = simulation_control(count)
    PI = SC.population_initalize()
    print("POpulation INITIALIZING...")
    day = 0

    while True:
        expansion = scale_3.get()
        Limit = scale_2.get()
        day += 1
        # print(PI)
        if day >= 0: SC.lockdown(scale_1.get())
        if day == 2: SC.initialize_infection(NOI)
        for t in range(0, 30):
            SC.population_movement(scale_4.get())
            if hospital: SC.hospitalization(PI, 0, expansion, 0, expansion)
            if vaccine : SC.vacination(PI, 0, expansion, 0, expansion)
        SC.economy_tracker()
        if day % 1 == 0: SC.non_display_stats(day, Limit)
        # SC.display()
        time.sleep(0)

        if day == Limit:
            print("=========================")
            read=True

            break

if Bot:start_another_thread(MCS_1)
else:start_another_thread(MCS_0)

def display00(yp,weight):
    img = Image.fromarray(yp, 'RGB')
    img = img.resize((600, 600))  # resizing so we can see our agent in all its glory.
    cv2.imshow("MonteCarloSimulation-0", np.array(img))  # show it!
    cv2.waitKey(weight)
print(len(maps))
mainloop()
