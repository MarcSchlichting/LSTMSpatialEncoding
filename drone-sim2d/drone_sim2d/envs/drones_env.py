import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

'''Global Variables'''
use_maps = False
'''End Global Variables'''

class DronesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        '''Constants'''
        self.amount = 20
        self.xrange_init = [-20,20]
        self.yrange_init = [-20,20]
        self.xrange_target = [-20,20]
        self.yrange_target = [-20,20]
        self.min_dist = 10
        self.delta_t = 1
        self.eps = 1.5
        self.eps_cav = 3
        self.delta_cav = 10
        self.eps_arr = 1.1
        self.delta_arr = 10
        self.phi_segments = np.linspace(-np.pi,np.pi,num=25)
        self.r_segments = np.array([0,3,4,5,7,10,15,20])
        self.max_stepcount = 1000
        self.stepcount = 0
        self.arr_rew_amount = 1000
        '''End Constants'''
        self.action_space = spaces.Box(low = 0, high = 1, shape = (2,))
        self.observation_space = spaces.Box(low = np.array([-np.pi,-np.pi,-400]), high = np.array([np.pi,np.pi,400]))
        self.vm = VehicleManager()
        self.controller = Controller()
        self.integrator = Integrator()
        self.simulate = Simulate(self.delta_t)
        self.metrics = Metrics()
        
        print("All initialization procedures completed!")

    def step(self, action):
        no_active_vehicles = len(self.vm.active_vehicles)

        #write actions to corresponding vehicles
        for idx_1 in range(no_active_vehicles):
            the_attitude = self.vm.active_vehicles[idx_1].attitude[-1]

            ######Version with cartesian velocity inputs, clipped to max length 1#######
            clipped_action = vector_clip(action[idx_1],1)       #1=max length which is allowed
            transformed_action = transform_gf(clipped_action,the_attitude)

            self.controller.fancy_controller(self.vm.active_vehicles[idx_1],transformed_action)

        #simulate the environment
        self.simulate.step(self.vm.active_vehicles,self.integrator,self.delta_t,self.vm,self.eps)

        #get cav reward and acc reward
        cav_reward_vector,collision_status = self.metrics.cav_reward(self.vm)
        acc_reward = self.metrics.acc_reward(self.vm,self.delta_t,action)

        #arrival reward
        arr_reward_vector = self.metrics.arr_reward(self.vm,self.arr_rew_amount,self.eps_arr,self.delta_arr)

        #time reward
        time_reward_vec = self.metrics.time_reward(self.vm)

        #build observations
        #maps = self.vm.generate_map(self.phi_segments,self.r_segments)
        other_positions = calculate_rel_position(self.vm)

        #build output
        output_vector = []
        done_flag_vector = [False]*no_active_vehicles
        for idx_2 in range(no_active_vehicles):
            if use_maps == True:
                state = (self.vm.active_vehicles[idx_2].destination_relative[0],maps[idx_2])
            else:
                state = (self.vm.active_vehicles[idx_2].destination_relative[0],other_positions[idx_2])


            reward = cav_reward_vector[idx_2]+6*acc_reward[idx_2]+1.5*time_reward_vec[idx_2]+arr_reward_vector[idx_2]

            #determine if episode for single vehicle is done
            
            if collision_status[idx_2]==-1000:
                done_flag_vector[idx_2] = True
            if arr_reward_vector[idx_2]>=self.arr_rew_amount:
                done_flag_vector[idx_2] = True 

            output_vector.append((state,reward,done_flag_vector[idx_2],{'CAV Reward': cav_reward_vector[idx_2],'ACC Reward': acc_reward[idx_2],'TIME Reward': time_reward_vec[idx_2],'ARR Reward':arr_reward_vector[idx_2]}))

            #delete all vehicles from active list with done_flag_vector == True      
        self.simulate.delete_finished(self.vm,done_flag_vector) 

        return output_vector


    def reset(self,amount=20,xrange_init=[-100,100],yrange_init=[-100,100],xrange_target=[-100,100],yrange_target=[-100,100],min_distance=10,delta_t=1,eps_arr_abort=1.5,eps_cav=3,delta_cav=6,eps_arr=1.1,delta_arr=15,phi_segments=np.linspace(-np.pi,np.pi,num=25),r_segments=np.array([0,3,4,5,7,10,15,20]),max_stepcount=1000,arr_rew_amount=1000,seed=None,options={}):
        '''set hyper_parameters'''
        self.amount = amount
        self.xrange_init = xrange_init
        self.yrange_init = yrange_init
        self.xrange_target = xrange_target
        self.yrange_target = yrange_target
        self.min_dist = min_distance
        self.delta_t = delta_t
        self.eps = eps_arr_abort
        self.eps_cav = eps_cav
        self.delta_cav = delta_cav
        self.eps_arr = eps_arr
        self.delta_arr = delta_arr
        self.phi_segments = phi_segments
        self.r_segments = r_segments
        self.max_stepcount = max_stepcount
        self.arr_rew_amount = arr_rew_amount
        

        '''end hyperparameters'''
        
        #delete all old data
        self.vm.all_vehicles = []
        self.vm.active_vehicles = []
        self.vm.creation_buffer = []
        self.vm.inter_list = []

       
        #create new data
        self.vm.create_vehicles(self.amount,self.min_dist, self.xrange_init,self.yrange_init,self.xrange_target,self.yrange_target)
        self.vm.make_buffer_active()
        #initial heading for all vehicles into the direction of the target
        for idx in range(len(self.vm.active_vehicles)):
            self.controller.direct_const_speed(self.vm.active_vehicles[idx],1)
        #print("controller done")
        #maps = self.vm.generate_map(self.phi_segments,self.r_segments)
        other_positions = calculate_rel_position(self.vm)

        #undo first control 
        for idx in range(len(self.vm.active_vehicles)):
            self.vm.active_vehicles[idx].commands = []
        #create output
        output_vector = []
        for idx_2 in range(len(self.vm.active_vehicles)):
            ######## Version with state only including attitude with regard to global system and relative orientation to target #####
            if use_maps == True:
                inter = (self.vm.active_vehicles[idx_2].destination_relative[0],maps[idx_2])
            else:
                inter = (self.vm.active_vehicles[idx_2].destination_relative[0],other_positions[idx_2])
            
            
            output_vector.append(inter)
        #print("reset done!")
        self.stepcount = 0          #reset stepcount 
        return output_vector



    def render(self, mode='human', close = False):
        raise NotImplementedError
    
    def display_map(self,data):
        display_map(data)

import numpy as np
import math
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

class Vehicle:
    def __init__(self,name,initial_pos,destination):
        self.id=name
        self.trajectory = []
        self.trajectory.append(initial_pos)
        self.destination=destination
        self.commands=[]
        self.tripdistance = calculate_distance(initial_pos,destination)
        self.attitude = []
        diff = destination-initial_pos
        self.attitude.append(np.arctan2(diff[1],diff[0]))       #equals the heading angle (in 3D-case change to quaternion representation of attitute to avoid kinemativ singularity at pitch = +- pi/2)
        self.destination_relative = relative_polar_angles(self.attitude[-1],self.destination-self.trajectory[-1])

class VehicleManager:
    def __init__(self):
        self.all_vehicles = []
        self.active_vehicles = []
        self.creation_buffer = []
        self.inter_list = []

    def create_vehicles(self,amount,min_dist, xorgrange,yorgrange,xdestrange,ydestrange):
        self.creation_buffer = []
        existing_origins = []
        existing_destinations = []

        for idx in range(len(self.active_vehicles)):
            existing_origins.append(self.active_vehicles[idx].trajectory[-1])
            existing_destinations.append(self.active_vehicles[idx].destination)

        origins = create_points_w_minimal_distance(amount,xorgrange,yorgrange,min_dist,existing_origins) # add feature to check against current position of all active vehicles
        destinations = create_points_w_minimal_distance(amount,xdestrange,ydestrange,min_dist,existing_destinations)  # add feature to check conflicting destinations of all active vehicles
        
        

        for idx in range(amount):
            unique_id = 'veh'+'{0:04}'.format(idx+1)
            trajectory = origins[idx]
            destination = destinations[idx]
            created_vehicle = Vehicle(unique_id,trajectory,destination)
            self.all_vehicles.append(created_vehicle)
            self.creation_buffer.append(created_vehicle)


    def make_buffer_active(self):
        self.active_vehicles = self.active_vehicles+self.creation_buffer
    
    def make_all_active(self):
        self.active_vehicles = self.all_vehicles

    def delete_from_active(self,del_list):
        del_list.sort(reverse=True)
        for idx in range(len(del_list)):
            self.active_vehicles = self.active_vehicles[0:del_list[idx]]+self.active_vehicles[del_list[idx]+1:]

    def generate_map(self,phi_segments,r_segments):      #segments as numpy array using boundaries as values
        #generate the empty map
        no_phi_segments = np.shape(phi_segments)[0]-1
        no_r_segments = np.shape(r_segments)[0]-1
        org_r_segments = r_segments
        #collect the data for all permutations of vehicles
        no_vehicles = len(self.active_vehicles)
        all_maps = []
        for idx_1 in range(no_vehicles):
            r_segments = org_r_segments
            density_map = np.zeros((no_phi_segments,no_r_segments))
            the_vehicle = self.active_vehicles[idx_1]
            the_speed_vector = the_vehicle.commands[-1]
            the_attitude = the_vehicle.attitude[-1]         #formerly the_speed_vector
            the_position = the_vehicle.trajectory[-1]
            the_speed = np.linalg.norm(the_speed_vector)
            r_segments = np.multiply(r_segments,the_speed)

            #calculate the volume of all map elements
            volume_map = generate_volume_map(phi_segments,r_segments) 


            other_vehicles = []
            for idx_2 in [x for x in range(no_vehicles) if x != idx_1]:
                other_vehicles.append(self.active_vehicles[idx_2])

            no_other_vehicles = len(other_vehicles)
            for idx_3 in range(no_other_vehicles):
                other_position = other_vehicles[idx_3].trajectory[-1]
                difference_vector = other_position-the_position
                #phi, r = relative_spherical_angles(the_speed_vector,difference_vector)
                phi, r = relative_polar_angles(the_attitude,difference_vector)
                phi_idx = np.asscalar(np.digitize(phi,phi_segments))
                r_idx = np.asscalar(np.digitize(r,r_segments))
                if r_idx <= no_r_segments:       #ignore vehicles farther away than limit specified in the r_segments_vector
                    density_map[phi_idx-1,r_idx-1] = density_map[phi_idx-1,r_idx-1]+1
            
            density_map = np.divide(density_map,0.0001*volume_map)     #normalize over volume

            all_maps.append(density_map)
        return all_maps
            

            

class Controller:
    def direct_const_speed(self,vehicle,speed):
        current_pos = vehicle.trajectory[-1]
        destination = vehicle.destination
        diff = destination - current_pos
        abs_diff = np.linalg.norm(diff)
        command = diff*speed/abs_diff
        vehicle.commands.append(command)

    def fancy_controller(self,vehicle,command):
        vehicle.commands.append(command)

class Simulate:
    def __init__(self,delta_t):
        self.delta_t = delta_t

    def step(self,active_vehicles,integrator,delta_t,vm,eps):
        for veh in active_vehicles:
            veh.trajectory.append(integrator.ivp_euler(veh.commands[-1],veh.trajectory[-1],delta_t))
            diff = veh.trajectory[-1]-veh.trajectory[-2]
            veh.attitude.append(np.arctan2(diff[1],diff[0]))
            veh.destination_relative = relative_polar_angles(veh.attitude[-1],veh.destination-veh.trajectory[-1])
        
    def delete_finished(self,vm,done_flag_vector):
        del_list  = [idx for idx in range(len(done_flag_vector)) if done_flag_vector[idx] == True]
        vm.delete_from_active(del_list)

        



class Integrator:
    def ivp_euler(self,value,initial,delta_t):
        return value*delta_t+initial

class Metrics:
    def cav_reward(self,vm):
        #Definition of parameters
        epsilon = 3
        omega = 6
        delta = 10
        theta = -2
        cap_theta = -1000
        #End Definition

        #loop over all vehicles
        no_active_vehicles = len(vm.active_vehicles)
        cav_reward = []
        min_cav = []
        for idx_1 in range(no_active_vehicles):
            the_position = vm.active_vehicles[idx_1].trajectory[-1]

            #get all other vehicles
            other_vehicles = []
            for idx_2 in [x for x in range(no_active_vehicles) if x != idx_1]:
                other_vehicles.append(vm.active_vehicles[idx_2])

            no_other_vehicles = len(other_vehicles)
            cav_reward_vehicle = []
            for idx_3 in range(no_other_vehicles):
                other_position = other_vehicles[idx_3].trajectory[-1]
                distance = calculate_distance(the_position,other_position)
                cav_reward_vehicle.append(exp_clamp(distance,epsilon,omega,delta,theta,cap_theta))
            cav_reward.append(np.sum(cav_reward_vehicle))
            try:
                min_cav.append(np.min(cav_reward_vehicle))
            except:
                min_cav.append(0)
        
        return cav_reward, min_cav

    def acc_reward(self,vm,h,action):
        ####Version with the use of lateral deviation using the actions#####
        #loop over all vehicles
        no_active_vehicles = len(vm.active_vehicles)
        acc_reward = []
        for idx_1 in range(no_active_vehicles):
            lateral_diff = -np.abs(action[idx_1][1])
            acc_reward.append(lateral_diff)
        
        return acc_reward

    def arr_reward(self,vm,amount,eps,delta):
        no_active_vehicles = len(vm.active_vehicles)
        arr_reward = []
        for idx_1 in range(no_active_vehicles):
            the_position = vm.active_vehicles[idx_1].trajectory[-1]
            the_destination = vm.active_vehicles[idx_1].destination
            distance = calculate_distance(the_position,the_destination)
            #arr_reward.append(amount*clamp_inv(distance,eps,delta))
            if distance<=eps:
                arr_reward.append(amount)
            else:
                arr_reward.append(0)

        return arr_reward
    
    def time_reward(self,vm):
        tripdistance_list = [veh.tripdistance for veh in vm.active_vehicles]
        kappa = np.divide(100,tripdistance_list)
        return -kappa
        #return [0.0]*len(vm.active_vehicles)


def create_points_w_minimal_distance(amount,xrange,yrange,min_dist,existing_points):
    points_list = []
    
    max_count = 1000
    counter = 0

    while (len(points_list)<amount) and (max_count>counter):
        distance_list = []
        complete_list = existing_points+points_list
        candidate = create_rand_point_in_range(xrange,yrange)
        for i in range(len(complete_list)):
            distance_list.append(calculate_distance(candidate,complete_list[i]))
        if len(points_list)==0:
            points_list.append(candidate)
        elif min(distance_list)<min_dist:
            pass
        else:
            points_list.append(candidate)
        counter = counter + 1
        if counter == max_count:
            raise Exception('Not enough points found in allowed time, to satisfy boundary conditions')
    #print("Counter: "+str(counter))
    return points_list




def create_rand_point_in_range(xrange,yrange):
    delta_x = xrange[1]-xrange[0]
    delta_y = yrange[1]-yrange[0]

    x_ini = xrange[0]
    y_ini = yrange[0]
    
    return np.array([np.random.random_sample()*delta_x+x_ini,np.random.random_sample()*delta_y+y_ini])

def calculate_distance(point1, point2):
    return np.linalg.norm(point1-point2)


def clamp(x,eps,delta):
        if x<=eps:
            return 0
        if x>=delta:
            return 1
        else:
            return (x)/(delta-eps)-(eps)/(delta-eps)
    
def clamp_inv(x,eps,delta):
    if x<=eps:
        return 1
    if x>=delta:
        return 0
    else:
        return (-x)/(delta-eps)+(eps)/(delta-eps)+1

def exp_clamp(x,epsilon,omega,delta,theta,cap_theta): 
    #x              position to be examined
    #epsilon        inner boundary to vehicle
    #omega          intermediate boundray as transition between linear and quadratic part
    #delta          outer boundary
    #theta          value of function at omega
    #cap_theta      value of function at epsilon
    #note: value of function at delta = 0
    if x >= delta:
        return 0
    elif (x < delta) and (x >= omega):
        return linear_term(x,omega,delta,theta)
    elif (x < omega) and (x >= epsilon):
        return quadratic_term(x,epsilon,omega,delta,theta,cap_theta)
    else:
        return cap_theta

def quadratic_term(x,epsilon,omega,delta,theta,cap_theta):
    a = (-epsilon*theta + delta*(-cap_theta + theta) + cap_theta*omega)/((epsilon - omega)**2*(-delta + omega))
    b = (epsilon**2*theta + omega*(2*delta *(cap_theta - theta) + (-2*cap_theta + theta)*omega))/((epsilon - omega)**2*(-delta + omega))
    c = (epsilon**2*delta*theta + cap_theta*(delta - omega)*omega**2 + epsilon*theta*omega*(-2*delta + omega))/((epsilon - omega)**2*(delta - omega))
    return a*x**2+b*x+c

def linear_term(x,omega,delta,theta):
    m = -theta/(delta-omega)
    c = (delta*theta)/(delta-omega)
    return m*x+c


def relu(x):
    if x<=0:
        return 0
    else:
        return x

def second_derivative(liste,h):   
    #computes the second derivative of a list of numpy arrays using asymmetrical difference formulae
    
    get_dim = liste[-1].shape[0]        #determine the dimensionality of the space
    inter = [np.zeros((get_dim,))]*4    #empty array
    for idx_1 in range(4):
        #in case array is to short it is assumed that the values are the same before the initial condition (equal to the vehicle is at a static position)
        try:
            inter[idx_1] = liste[-idx_1-1]
        except:
            inter[idx_1] = inter[idx_1-1]
    
    #calculate derivative

    der = 2*inter[0]-5*inter[1]+4*inter[2]-inter[3]
    abs_der = np.linalg.norm(der)
    return abs_der
  

def create_environment(filename):
    file_name_points = filename+"_points.txt"
    file_name_values = filename+"_values.txt"

    points = np.loadtxt(file_name_points)
    values = np.loadtxt(file_name_values)

    delaunay = Delaunay(points)
    interpolator = LinearNDInterpolator(delaunay,values,fill_value=1)
    return interpolator

def relative_spherical_angles(vec1,vec2):   #vec1 represents the flight vector, vec2 represents the difference vector between the_vehicle and the_target
    #vec1_phi = vec1[[0,1]]
    #phi_0 = angle_between_vectors(np.array([1,0]),vec1_phi)
    phi_0 = np.arctan2(vec1[1],vec1[0])     

    #calculation of phi angle between (1,0,0) and vec2
    #vec2_phi = vec2[[0,1]]
    #phi = angle_between_vectors(np.array([1,0]),vec2_phi)
    phi = np.arctan2(vec2[1],vec2[0])

    phi = phi-phi_0

    #absolute of distance vector
    absolute = np.linalg.norm(vec2)
    phi, absolute = equal_polar_coordinates(phi,absolute)

    return phi, absolute

def relative_polar_angles(phi_0,vec):
    phi = np.arctan2(vec[1],vec[0])-phi_0
    absolute = np.linalg.norm(vec)
    phi, absolute = equal_polar_coordinates(phi,absolute)
    return np.array([phi, absolute])
    


def generate_volume_map(phi_segments,r_segments):
    no_phi_segments = np.shape(phi_segments)[0]-1
    no_r_segments = np.shape(r_segments)[0]-1
    empty_map = np.zeros((no_phi_segments,no_r_segments))

    for idx_101 in range(no_phi_segments):
        for idx_103 in range(no_r_segments):
            delta_phi = phi_segments[idx_101+1]-phi_segments[idx_101]
            surface_percentage = delta_phi/(2*np.pi)
            volume = np.pi*(np.power(r_segments[idx_103+1],2)-np.power(r_segments[idx_103],2))*surface_percentage
            empty_map[idx_101,idx_103] = volume
    
    return empty_map

def equal_polar_coordinates(phi,r):
    while phi>np.pi:
        phi = phi-2*np.pi
    
    while phi<-np.pi:
        phi = phi+2*np.pi

    return phi,r

def equal_angle(phi):
    while phi>np.pi:
        phi = phi-2*np.pi
    
    while phi<-np.pi:
        phi = phi+2*np.pi

    return phi

def transform_gf(vec,phi):
    dcm = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    return np.matmul(dcm,vec)

def transform_gf_polar(vec,the_attitude):
    x = np.multiply(np.cos(the_attitude+vec[0]),vec[1])
    y = np.multiply(np.sin(the_attitude+vec[0]),vec[1])
    return np.array([x,y])

def display_map(data):
    data_max = np.max(data)
    if data_max != 0:
        data_grad = 1/data_max
    else:
        data_grad = 0
    no_of_rings = data.shape[1]
    no_of_segments = data.shape[0]



    fig, ax = plt.subplots()


    size = 0.3          #size for each individual ring
    #color = mpl.colors.to_rgba(color)
    distributions = np.ones(no_of_segments,)

    #max radius
    min_radius = 0.5
    for i in range(no_of_rings):
        #color_creation
        color = np.zeros((no_of_segments,4))
        for idx_2 in range(no_of_segments):
            color[idx_2,0] = 1
            color[idx_2,1] = 1-data_grad*data[idx_2,i]
            color[idx_2,2] = 1-data_grad*data[idx_2,i]
            color[idx_2,3] = 1
        ax.pie(distributions, radius=min_radius+(i*size),colors=color ,wedgeprops=dict(width=size, edgecolor=(0.6,0.6,0.6,1)),counterclock=False,startangle=270)


    ax.set(aspect="equal")
    ax.set_xlim([-3,3])
    plt.show()

def save_state(return_from_step):
    import numpy as np
    no_vehicles = len(return_from_step)
    output = []
    for idx_1 in range(no_vehicles):
        try:
            row = np.hstack((return_from_step[idx_1][0][0],return_from_step[idx_1][0][1],return_from_step[idx_1][0][2]))
            output = np.vstack((output,row))
        except:
            output = np.hstack((return_from_step[idx_1][0][0],return_from_step[idx_1][0][1],return_from_step[idx_1][0][2]))

    #save to txt
    np.savetxt('state.txt',output)
        
def vector_clip(action,max_abs):
    abs_action = np.linalg.norm(action)
    if abs_action>max_abs:
        return np.divide(action,abs_action)
    else:
        return action

def calculate_rel_position(vm):
    no_vehicles = len(vm.active_vehicles)
    rel_pos_vector = []
    for idx_1 in range(no_vehicles):
        the_vehicle = vm.active_vehicles[idx_1]
        the_position = the_vehicle.trajectory[-1]
        the_attitude = the_vehicle.attitude[-1]
        
        idx_1_rel_position = []
        for idx_2 in [x for x in range(no_vehicles) if x != idx_1]:
            other_vehicle = vm.active_vehicles[idx_2]
            other_position = other_vehicle.trajectory[-1]
            other_attitude = other_vehicle.attitude[-1]
            
            diff_position = other_position-the_position
            pos = calculate_relative_angles(the_attitude,other_attitude,diff_position)
            idx_1_rel_position.append(pos)
        rel_pos_vector.append(idx_1_rel_position)
    return rel_pos_vector


def calculate_relative_angles(phi0,phi1,r):
    r_abs = np.linalg.norm(r)
    phi01 = np.arctan2(r[1],r[0])-phi0
    delta = phi1-phi01-phi0+np.pi
    phi01 = equal_angle(phi01)
    delta = equal_angle(delta)
    delta = np.abs(delta)
    return np.array([r_abs, phi01, delta])          


if __name__ == "__main__":    
    print("done test procedure")



