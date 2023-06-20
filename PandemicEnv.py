#!pip install keras==2.2.4
#!pip install tensorflow==1.13.1
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,LearningRateScheduler
from keras.models import model_from_json
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
import os
from matplotlib import style
import time
import matplotlib.pyplot as plt

style.use("ggplot")

#@title Default title text
# simulation
# 1 : up
# 2 : left
# 3 : down
# 4 : right
import numpy as np
import random,time,math
import matplotlib
import matplotlib.pyplot as plt

SIZE_L = SIZE_B = 10
DIRECTIONS = {1: (-1, 0), 2: (0, -1), 3: (1, 0), 4: (0, 1), 5: (0, 0),
              6: (1, 1), 7: (1, -1), 8: (-1, -1), 9: (-1, 1)}
test=True

class Person:
    def __init__(self, env, x=None, y=None):
        self.x, self.y = x, y
        self.subject_sign, self.contact, self.immune = 1, False, False
        self.env, self.treatment_step, self.medicines = env, 0, 5
        self.current_step, self.treatment, self.income = 0, False, 0
        self.prev_post, self.earning = None, random.randint(1, 10)
        self.infected_times, self.infected_days = 0, None
        self.infection_period, self.death, self.infection_rate = 20, False, 0
        self.reported, self.env[x][y] = False, self.subject_sign
        self.infection_rate = 0.0005

    def step_update(self, step):
        self.current_step = step
        if self.treatment and self.current_step > self.treatment_step:
            self.contact, self.treatment, self.reported = False, False, False
            return self.temp, self.infection_rate
        return 0, self.infection_rate

    def move(self, direct, infected=None, contagion=False):
        dx, dy = DIRECTIONS.get(direct, (0, 0))
        new_x, new_y = max(0, min(SIZE_B - 1, self.x + dx)), max(0, min(SIZE_L - 1, self.y + dy))
        self.env[new_x][new_y], self.env[self.x][self.y] = self.subject_sign, 0
        self.x, self.y = new_x, new_y

        self.contagion = contagion
        if not (self.contact or self.immune or self.treatment) and contagion and infected.count((self.x, self.y)) >= 1 and random.random() <= self.infection_rate:

            self.contact, self.subject_sign, self.infected_times = True, 4, self.infected_times + 1
            self.infected_days = self.current_step + self.infection_period

        if self.infected_days is not None and self.contact and self.current_step > self.infected_days:
            self.contact, self.infected = False, None
            if random.random() < 3/100: self.death = True

        return self.env, self.x, self.y, self.contact

    def vacinated(self, x1=8, x2=12, y1=8, y2=12):
        if x1 <= self.x <= x2 and y1 <= self.y <= y2:
            self.contact, self.immune, self.subject_sign = False, True, '7'
        return self.immune

    def hospital(self, x1, x2, y1, y2):
        if x1 <= self.x <= x2 and y1 <= self.y <= y2 and self.contact and not self.immune and not self.treatment:
            self.subject_sign, self.treatment, self.treatment_step = 3, True, self.current_step + self.medicines

    def Earning(self, TP, AC, D, LP):
        if (self.x, self.y) != self.prev_post:
            try:
              reduction = self.earning * (0.62 * (math.log(1 + (AC / TP), 10) + 1) - math.log(1 + (D / TP), 2.71828)) + self.earning * LP
              reduction += self.earning * 0.2 if self.contact else 0
              return self.income + self.earning - reduction
            except:
              return self.income
        else:
            return self.income

    def Death(self):
        if random.randrange(0,200000) <= self.infected_times and self.contact:
            self.death = True
        return self.death

    def details_pop(self):
        return self.infection_period


class SimulationControl:
    def __init__(self, nop=None):
        self.contact_list = []
        self.SIZE_L = SIZE_L
        self.en = np.zeros((self.SIZE_L,self.SIZE_L))
        self.NOP = nop
        self.vacinated = []
        self.steps = []
        self.immune_steps = []
        self.prev_contacts = 0
        self.cnt = []
        self.total_recovered = []
        self.infected = []
        self.current_step = 1
        self.new_case_list = []
        self.recovery_list = []
        self.new_case_steps = []
        self.recovery_steps = []
        self.total_recovered_cases = 0
        self.Total_Economy_perstep = 0
        self.prev_immune=0
        self.Total_income_list = []
        self.current_step_lst = []
        self.lockdown_percent = 0
        self.lockdown_list = []
        self.dead_people_list = []
        self.dead_people = 0
        self.death_steps = []
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
        self.per_day_reward = 300
        self.NOI = 0

    def population_initalize(self):
        IP = 0 #initial population
        LOP = [] #list of people
        num_initially_infected = 3  # for example
        while IP<self.NOP:
            person = Person(self.en,x=random.randint(0,self.SIZE_L-1),y=random.randint(0,self.SIZE_L-1))
            if IP < num_initially_infected:
                person.infected = True  # assuming your Person class has an 'infected' attribute
            self.IF=person.details_pop()
            LOP.append(person)
            self.vacinated.append(person.immune)
            IP=IP+1
        self.working_list=LOP
        self.lop=LOP
        self.TP=len(self.lop)
        return LOP,self.en

    def handle_person(self,person, contagion, no_movement):
        a, self.ir = person.step_update(self.current_step)
        env,x0,y0,state0 = self.step(person,self.infected,contagion,no_movement)
        if person.Death()==True:
          self.reward = self.reward+self.death_reward #death-reward
          if person in self.lockdown_list:
            self.lockdown_list.remove(person)
          else:
            self.working_list.remove(person)

        if person in self.LOP:
          if state0:
            self.infected.append((x0,y0))
            self.contact_list.append(state0)
          self.Total_Economy_perstep = self.Total_Economy_perstep + person.Earning(self.TP,self.AC,self.Deaths,self.lockdown_percent)
        else:
          if (x0,y0) in self.infected:self.infected.remove((x0,y0))


    def population_movement(self,posts=None):   # ONe MOVEMENT
        s_ = time.time()
        self.contact_list.clear()
        contagion = len(self.infected) != 0

        for person in self.lockdown_list:
          self.handle_person(person, contagion, no_movement=True)

        for person in self.working_list:
          self.handle_person(person, contagion, no_movement=False)
        print('Time taken by ', person,'is ', s_-time.time())

    def location_registry(self):
        return [(person.x, person.y) for person in self.lop]

    def step(self, psn, infected=None, contagion=False, no_movement=False):
        ipt = random.randint(1,9) if not no_movement else 5
        return psn.move(ipt, infected, contagion)

    def vacination(self, lop, x1, x2, y1, y2):
        self.en[x1:x2, y1:y2] = self.immune_code
        [person.vacinated(x1, x2, y1, y2) for person in self.lop]
        return self.en

    def hospitalization(self, lop, x1, x2, y1, y2):
        self.en[x1:x2, y1:y2] = self.hospital_code
        [person.hospital(x1, x2, y1, y2) for person in lop]

    def initialize_infection(self):
        [setattr(random.choice(self.lop), 'contact', True) for _ in range(3)]

    def display(self):
        print(self.en, "\n\n")

    def economy_tracker(self):
        # This is just a placeholder example. Replace with your actual calculation.
        self.Total_Economy_perstep = sum([person.Earning(self.TP,self.AC,self.Deaths,self.lockdown_percent) for person in self.lop])
        self.reward = self.reward-self.Total_Economy_perstep/1000 #economy-reward   # 2.xxx for 40
        self.Total_income_list.append(self.Total_Economy_perstep)


    def lockdown(self, percent=0):
        if self.current_step < 20: percent = 0
        self.lockdown_percent = percent / 100
        self.lockdown_list = random.sample(self.lop, int(len(self.lop)*self.lockdown_percent))
        self.working_list = [sample for sample in self.lop if sample not in self.lockdown_list]
        if test:print("+++++++++++++++++++++++++++++++++++ {} LOCKDOWN +++++++++++++++++++++++++++++++++++".format(percent))


    def non_display_stats(self,y,Limit):
        #print(len(self.LOP),len(self.lockdown_list),len(self.working_list))
        self.current_step = y+1
        self.current_step_lst.append(y)
        self.new_case = 0
        recovered_cases=0
        u=0

        if u==0:
          contact=0
          imune=[]
          immune=0

          self.LOP=[]
          for e in self.lockdown_list:
            self.LOP.append(e)
          for ee in self.working_list:
            self.LOP.append(ee)
          contact=len(self.contact_list)
          for c in self.contact_list:
            self.reward = self.reward+self.new_cases_reward #new_cases_reward
          if contact>self.prev_contacts:
            self.steps.append(y)
            self.cnt.append(contact)
            self.new_case =contact - self.prev_contacts
            self.new_case_list.append(self.new_case)
            self.new_case_steps.append(y)

          elif contact<self.prev_contacts:
            self.steps.append(y)
            self.cnt.append(contact)
            recovered_cases = self.prev_contacts-contact
            self.total_recovered_cases =  self.total_recovered_cases + (self.prev_contacts-contact)
            self.total_recovered.append(self.total_recovered_cases)
            self.recovery_list.append(recovered_cases)
            self.recovery_steps.append(y)

          self.prev_contacts=contact
          for i in self.total_recovered:self.reward = self.reward+self.new_recovery_reward #recovery-reward
          for immune_state in self.vacinated:
            if immune_state:immune+=1

          if immune>self.prev_immune:
            self.immune_steps.append(y)
            d=immune-self.prev_immune
            if d != 0:self.immune.append(immune)
          self.prev_immune=immune
          Economy_state=self.Total_Economy_perstep
          self.Total_Economy_perstep = 0

        if (self.dead_people - len(self.LOP))!=0:
          self.death_steps.append(y)
          self.dead_people_list.append(self.dead_people - len(self.LOP))
        self.dead_people=len(self.LOP)
        if contact<=self.NOI and len(self.recovery_list)!=0:
          self.done=True
          for person in self.LOP:
            if person.contact==True:
              person.contact=False
      # print(self.contact_list)
        #print(self.dead_people_list,len(self.LOP))
        Total_deaths=0
        Total_death_list=[]
        for deaths in self.dead_people_list[1:]:
          Total_deaths=deaths+Total_deaths
          Total_death_list.append(Total_deaths)

        self.AC=contact
        self.Deaths=Total_deaths
        if test:print("DAYS : ",y,"ToTal POpulation : ",len(self.LOP),"Active Cases : ",contact,"New Cases : ",self.new_case,"Recovery : ",recovered_cases,"IMMUNE : ",immune,"Deaths : ",Total_deaths)
        if y>Limit and test:
          #fig,ax = plt.subplots()
        # print(self.recovery_steps,self.total_recovered)
          print(self.cnt)
          plt.plot(self.steps,self.cnt,'red')
          #plt.plot(self.recovery_steps,self.total_recovered,'green')
          print(len(self.death_steps[1:]),len(Total_death_list))
          plt.plot(self.death_steps[1:],Total_death_list,color="Black")

          #plt.plot(self.immune_steps,self.immune,'blue')
          #plt.plot(self.new_case_steps,self.new_case_list,color="Red")
          #plt.plot(self.recovery_steps,self.recovery_list,color="green")
          #plt.set(xlabel='STeps', ylabel='PoPulation',
                #title='MonteCarloSimulation')
          #plt.grid()
          plt.show()


          print(self.Total_income_list)
          TC=[0]
          for income in self.Total_income_list:
            TC.append(income)
          plt.plot(self.current_step_lst,TC,color="green",linestyle="dashed")
          #plt.grid()
          plt.show()
        self.per_day_reward = self.per_day_reward-1
        self.reward = self.reward + self.per_day_reward
        #print(self.reward)
        self.data=np.array([y,len(self.LOP)/100,contact,self.new_case,recovered_cases,immune,Total_deaths,Economy_state/10,SIZE_L/100,self.IF])
        return self.data, self.reward ,self.done


class PandemicOutbreak(gym.Env):
  def __init__(self):
    self.action_space = gym.spaces.Discrete(101)
    self.observation_space = np.array([1,2,3,4,5,6,7,8,9,10])
    #self.observation_space = gym.spaces.Discrete(6)
    self.SC=None
    self.PT=None
    self.day=None
    self.Limit=None
  def reset(self):
    self.SC = SimulationControl(100)
    self.PI,state = self.SC.population_initalize()
    print("POpulation INITIALIZING...")
    self.day=0
    self.Limit = 100
    self.reward=0
    new_state,reward,done=self.SC.non_display_stats(self.day,self.Limit)
    return new_state
  def step(self,action):
    self.day+=1
    self.SC.lockdown(action)
    if self.day==2 :self.SC.initialize_infection()
    for t in range(0,30):
          self.SC.population_movement(self.SC.location_registry())
          #if self.day>0:self.SC.hospitalization(self.PI,3,8,3,8)
          if self.day>7050:self.SC.vacination(self.PI,15,25,15,25)
    self.SC.economy_tracker();done=False
    if self.day%1==0:new_state,reward,done=self.SC.non_display_stats(self.day,self.Limit)
    if self.day>self.Limit:
      done=True

    return new_state, reward, done, {}

  def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]