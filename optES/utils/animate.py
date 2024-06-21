

from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
from optES.utils.paths import PLOTS_PATH


class AnimateArm2DOF():

    def __init__(self, system, params):
        '''
        This class is used to animate the 2DOF arm system
        @param system: the system to animate (object of class Arm2DOF)
        @param params: the parameters of the system (dictionary)
        '''
        self.system = system
        self.params = params
        self.data = None

        self.fig, self.ax = plt.subplots()

        # Initialize the line you want to clear in each frame
        self.line, = self.ax.plot([], [], 'o-')
        self.prev_x_tip, self.prev_y_tip = None, None

    def animate(self, i):
        '''
        This function is called at each frame of the animation
        @param i: the frame index
        '''
        # Remove the old line
        self.line.remove()

        q1 = self.data["q1"][i]
        q2 = self.data["q2"][i]
        x1 = - self.system.l1 * np.cos(q1)
        y1 = self.system.l1 * np.sin(q1)
        x2 = x1 - self.system.l2 * np.cos(q1+q2)
        y2 = y1 + self.system.l2 * np.sin(q1+q2)

        # Plot the new line and keep a reference to it
        self.line, = self.ax.plot([0, x1, x2], [0, y1, y2], 'b-', linewidth=5)
        # self.line, = self.ax.plot([x1], [y1], 'bo', markersize=10)
        # self.line, = self.ax.plot([x2], [y2], 'bo', markersize=10)
        
        x_tip = x2
        y_tip = y2
        # If the previous tip position is not None, draw a line segment
        if self.prev_x_tip is not None and self.prev_y_tip is not None: 
            self.ax.plot([self.prev_x_tip, x_tip], [self.prev_y_tip, y_tip], 'r*') 

        # plot tip target 
        target_pos = self.params["target_pos"] if "target_pos" in self.params else [0,0] # target joints positions
        target_x_tip = - self.system.l1 * np.cos(target_pos[0]) - self.system.l2 * np.cos(target_pos[0]+target_pos[1])
        target_y_tip = self.system.l1 * np.sin(target_pos[0]) + self.system.l2 * np.sin(target_pos[0]+target_pos[1])
        self.ax.plot([target_x_tip], [target_y_tip], 'g*', markersize=10)
 
        # Update the previous tip position
        self.prev_x_tip, self.prev_y_tip = x_tip, y_tip
 
        plt.xlim(-2,2)
        plt.ylim(-2,2) 

    def record(self, i, data):
        '''
        This function is called at each frame of the animation
        @param i: the frame index
        @param data: the data to be recorded (dictionary)
        '''
        self.data = data
        self.animate(i)

    def run(self, data, save_name="arm2dof"):
        '''
        This function runs the animation and saves it as a gif file 
        @param data: the data to be animated (dictionary)
        @param save_name: the name of the file to save the animation
        '''

        self.ax.clear()
        self.data = data
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.params["model"]["horizon"], interval=100)
        self.ani.save(os.path.join(PLOTS_PATH, f"{save_name}.gif"), writer='imagemagick', fps=60)
        plt.show()
 