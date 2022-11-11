import matplotlib.pyplot as plt
import numpy as np
import glob, os

os.chdir("./trajectories/")
file_list = glob.glob("*.txt")
file_list.sort()
for i,file in enumerate(file_list):
    print(file)
    trajectory = np.loadtxt(file)
    plt.plot(trajectory[:,0],trajectory[:,1],label="Vehicle "+str(i+1))

plt.legend()
plt.show()
