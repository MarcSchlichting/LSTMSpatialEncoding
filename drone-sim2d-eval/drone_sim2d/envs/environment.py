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

class VehicleManager:
    def __init__(self):
        self.all_vehicles = []
        self.active_vehicles = []
        self.creation_buffer = []

    def create_vehicles(self,amount,min_dist, xorgrange,yorgrange,zorgrange,xdestrange,ydestrange,zdestrange):
        self.creation_buffer = []
        existing_origins = []
        existing_destinations = []

        for idx in range(len(self.active_vehicles)):
            existing_origins.append(self.active_vehicles[idx].trajectory[-1])
            existing_destinations.append(self.active_vehicles[idx].destination)

        origins = create_points_w_minimal_distance(amount,xorgrange,yorgrange,zorgrange,min_dist,existing_origins) # add feature to check against current position of all active vehicles
        destinations = create_points_w_minimal_distance(amount,xdestrange,ydestrange,zdestrange,min_dist,existing_destinations)  # add feature to check conflicting destinations of all active vehicles
        
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

    def generate_map(self,phi_segments,theta_segments,r_segments):      #segments as numpy array using boundaries as values
        #generate the empty map
        no_phi_segments = np.shape(phi_segments)[0]-1
        no_theta_segments = np.shape(theta_segments)[0]-1
        no_r_segments = np.shape(r_segments)[0]-1

        #collect the data for all permutations of vehicles
        no_vehicles = len(self.active_vehicles)
        all_maps = []
        for idx_1 in range(no_vehicles):
            density_map = np.zeros((no_phi_segments,no_theta_segments,no_r_segments))
            the_vehicle = self.active_vehicles[idx_1]
            the_speed_vector = the_vehicle.commands[-1]
            the_position = the_vehicle.trajectory[-1]
            the_speed = np.linalg.norm(the_speed_vector)
            r_segments = np.multiply(r_segments,the_speed)

            #calculate the volume of all map elements
            volume_map = generate_volume_map(phi_segments,theta_segments,r_segments) 


            other_vehicles = []
            for idx_2 in [x for x in range(no_vehicles) if x != idx_1]:
                other_vehicles.append(self.active_vehicles[idx_2])

            no_other_vehicles = len(other_vehicles)
            for idx_3 in range(no_other_vehicles):
                other_position = other_vehicles[idx_3].trajectory[-1]
                difference_vector = the_position-other_position
                phi, theta, r = relative_spherical_angles(the_speed_vector,difference_vector)
                phi_idx = np.asscalar(np.digitize(phi,phi_segments))
                theta_idx = np.asscalar(np.digitize(theta,theta_segments))
                r_idx = np.asscalar(np.digitize(r,r_segments))
                density_map[phi_idx-1,theta_idx-1,r_idx-1] = density_map[phi_idx-1,theta_idx-1,r_idx-1]+1
            
            density_map = np.divide(density_map,volume_map)     #normalize over volume

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
        del_list  = []
        for idx in range(len(vm.active_vehicles)):
            if calculate_distance(vm.active_vehicles[idx].trajectory[-1],vm.active_vehicles[idx].destination)<eps:
                del_list.append(idx)
        vm.delete_from_active(del_list)

        



class Integrator:
    def ivp_euler(self,value,initial,delta_t):
        return value*delta_t+initial

class Metrics:
    def complete_reward(self,vm,eps_cav,delta_cav,eps_arr,delta_arr,time_step,env):
        cav_reward_vec = self.cav_reward(vm,eps_cav,delta_cav)
        env_reward_vec = self.env_reward(vm,env)
        arr_reward_vec = self.arr_reward(vm,eps_arr,delta_arr)
        vel_reward_vec = self.vel_reward(vm,time_step)
        acc_reward_vec = self.acc_reward(vm,time_step,vel_reward_vec)

        complete_reward_vec = (np.array(cav_reward_vec)*np.array(env_reward_vec)*np.array(arr_reward_vec))*(np.array(vel_reward_vec)+np.array(acc_reward_vec))
        return complete_reward_vec



    def cav_reward(self,vm,eps,delta):
        no_vehicles = len(vm.all_vehicles)
        cav_reward_vector = []
        for idx_1 in range(no_vehicles):
            the_vehicle = vm.all_vehicles[idx_1]
            other_vehicles = []
            #create list of all other vehicles 
            for idx_2 in [x for x in range(no_vehicles) if x != idx_1]:
                other_vehicles.append(vm.all_vehicles[idx_2])
            
            #loop over all time steps
            no_timesteps = len(the_vehicle.trajectory)
            cav_reward_all_time_steps = []
            for idx_3 in range(no_timesteps):
                the_position = the_vehicle.trajectory[idx_3]
                #create list of positions of all other vehicles
                position_list = []
                for idx_4 in range(no_vehicles-1): #minus the_vehicle
                    try:
                        position_list.append(other_vehicles[idx_4].trajectory[idx_3])
                    except:
                        pass
                
                #calculate the distance between other active vehicles at the timestep (included in position_list) and the_vehicle
                no_active_vehicles = len(position_list)
                distance_list = []
                for idx_5 in range(no_active_vehicles):
                    distance_list.append(calculate_distance(the_position,position_list[idx_5]))
                    #clamp the result
                    distance_list[idx_5] = clamp(distance_list[idx_5],eps,delta)
                #append reward for current time step to all time steps cav vector
                cav_reward_all_time_steps.append(np.product(distance_list))

            #calculate reward for all time steps 
            cav_reward_vector.append(np.product(cav_reward_all_time_steps))
        return cav_reward_vector
    
    def env_reward(self,vm,env):        #env must be NDInterpolator object from scipy (see header), Delaunay must be initialized in init function
        #use of Delaunay-Triangulation and barycentric interpolation to determine the reward with regard to the environment
        no_vehicles = len(vm.all_vehicles)
        env_reward_vector = []
        for idx_1 in range(no_vehicles):
            the_vehicle = vm.all_vehicles[idx_1]
            no_timesteps = len(the_vehicle.trajectory)
            env_reward_time_vector = []
            for idx_2 in range(no_timesteps):
                env_reward_time_vector.append(env(the_vehicle.trajectory[idx_2]))
            env_reward_vector.append(np.product(env_reward_time_vector))
        
        return env_reward_vector



    def arr_reward(self,vm,eps,delta):
        #calculate the arrival reward for each vehicle in the all_vehicle list
        no_vehicles = len(vm.all_vehicles)
        arr_reward_vector = []
        for idx_1 in range(no_vehicles):
            the_vehicle = vm.all_vehicles[idx_1]
            destination = the_vehicle.destination
            final_postion = the_vehicle.trajectory[-1]
            distance = calculate_distance(destination,final_postion)
            arr_reward_vector.append(clamp_inv(distance,eps,delta))
        
        return arr_reward_vector

    def vel_reward(self,vm,time_step):
        #calculate the reward for relative velocity (meaning how fast the vehicle reaches the target)
        no_vehicles = len(vm.all_vehicles)
        vel_reward_vector = []
        for idx_1 in range(no_vehicles):
            the_vehicle = vm.all_vehicles[idx_1]
            destination = the_vehicle.destination
            initial_pos = the_vehicle.trajectory[0]
            diff = np.linalg.norm(np.array(destination) - np.array(initial_pos))
            total_time = len(vm.all_vehicles[idx_1].trajectory)*time_step
            vel_reward_vector.append(diff/total_time)

        return vel_reward_vector

    def acc_reward(self,vm,time_step,vel_reward_vector):
       #determine the acceleration reward based on the porduct of the two rewards for mean_acceleration and the reward for max_acceleration
        no_vehicles = len(vm.all_vehicles)
        acc_reward_vector = []
        for idx_1 in range(no_vehicles):
            #Calculation of acceleration vector
            the_vehicle = vm.all_vehicles[idx_1]
            no_timesteps = len(the_vehicle.trajectory)
            acc_vector = []
            for idx_2 in range(no_timesteps):
                acc_vector.append(np.linalg.norm(second_derivative(the_vehicle.trajectory,time_step,idx_2)))
            
            #max acc reward
            max_acc = np.max(acc_vector)
            max_acc_reward = vel_reward_vector[idx_1]/(max_acc+1)

            #mean acc reward
            mean_acc = np.mean(acc_vector)
            mean_acc_reward = vel_reward_vector[idx_1]/(mean_acc+1)

            #final calculation of reward
            acc_reward_vector.append(max_acc_reward*mean_acc_reward)
        
        return acc_reward_vector

        


def create_points_w_minimal_distance(amount,xrange,yrange,zrange,min_dist,existing_points):
    points_list = []
    
    max_count = 1000
    counter = 0

    while (len(points_list)<amount) and (max_count>counter):
        distance_list = []
        complete_list = existing_points+points_list
        candidate = create_rand_point_in_range(xrange,yrange,zrange)
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
    print("Counter: "+str(counter))
    return points_list




def create_rand_point_in_range(xrange,yrange,zrange):
    delta_x = xrange[1]-xrange[0]
    delta_y = yrange[1]-yrange[0]
    delta_z = zrange[1]-zrange[0]

    x_ini = xrange[0]
    y_ini = yrange[0]
    z_ini = zrange[0]

    return np.array([np.random.random_sample()*delta_x+x_ini,np.random.random_sample()*delta_y+y_ini,np.random.random_sample()*delta_z+z_ini])

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

def relu(x):
    if x<=0:
        return 0
    else:
        return x

def second_derivative(liste,h,idx):
    length = len(liste)
    if (length>2) and (idx>=1) and (idx<length-1):
        res = 1/(h**2)*(liste[idx-1]-2*liste[idx]+liste[idx+1])
    elif (length>2) and (idx==0):
        res = 1/(2*h)*(-3*liste[idx]+4*liste[idx+1]-liste[idx+2])
    elif (length>2) and (idx==length-1):
        res = 1/(2*h)*(-liste[idx-2]+4*liste[idx-1]-3*liste[idx])
    else:
        res = 0

    return res

def create_environment(filename):
    file_name_points = filename+"_points.txt"
    file_name_values = filename+"_values.txt"

    points = np.loadtxt(file_name_points)
    values = np.loadtxt(file_name_values)

    delaunay = Delaunay(points)
    interpolator = LinearNDInterpolator(delaunay,values,fill_value=1)
    return interpolator

def relative_spherical_angles(vec1,vec2):   #vec1 represents the flight vector, vec2 represents the difference vector between the_vehicle and the_target
    #determine theta and phi between (1,0,0) and vec1
    theta_0 = np.arctan2(np.linalg.norm(vec1[[0,1]]),vec1[2])
    phi_0 = np.arctan2(vec1[1],vec1[0])     

    #calculation of theta angle between (1,0,0) and vec2
    theta = np.arctan2(np.linalg.norm(vec2[[0,1]]),vec2[2])

    #calculation of phi angle between (1,0,0) and vec2
    phi = np.arctan2(vec2[1],vec2[0])

    phi = phi-phi_0
    theta = theta-theta_0

    #absolute of distance vector
    absolute = np.linalg.norm(vec2)

    return phi,theta,absolute


def angle_between_vectors(vec1,vec2):
    numerator = np.dot(vec1,vec2)
    denominator = np.linalg.norm(vec1)*np.linalg.norm(vec2)
    sign  = np.sign(np.cross(vec1,vec2))      #equal to the determinant of the basis in 2D
    return np.arccos(numerator/denominator)*sign

def generate_volume_map(phi_segments,theta_segments,r_segments):
    no_phi_segments = np.shape(phi_segments)[0]-1
    no_theta_segments = np.shape(theta_segments)[0]-1
    no_r_segments = np.shape(r_segments)[0]-1
    empty_map = np.zeros((no_phi_segments,no_theta_segments,no_r_segments))

    for idx_101 in range(no_phi_segments):
        for idx_102 in range(no_theta_segments):
            for idx_103 in range(no_r_segments):
                delta_phi = phi_segments[idx_101+1]-phi_segments[idx_101]
                delta_theta = theta_segments[idx_102+1]-theta_segments[idx_102]
                surface_percentage = delta_phi/(2*np.pi)*delta_theta/(np.pi)
                volume = 4/3*np.pi*(np.power(r_segments[idx_103+1],3)-np.power(r_segments[idx_103],3))*surface_percentage
                empty_map[idx_101,idx_102,idx_103] = volume
    
    return empty_map





if __name__ == "__main__":
    print("done")

    