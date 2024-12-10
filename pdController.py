import numpy as np

class PD_Controller(object):

    def __init__(self):
        self.previous_error_r = 0
        self.previous_error_l = 0

        self.time_step_r = 0.2
        self.time_step_l = 0.8

        self.t_max_new_r = 1.2 #time of my measurements
        self.leverArm_Hip = 0.1 #10cm
        self.leverArm_Knee = 0.05 #5cm
        self.bFlight_r = False
        self.bFlight_l = True
        self.threshold = 30
        self.bStance_r = True
        self.bStance_l = False
        self.LastSteps_r = np.zeros([5])
        self.LastSteps_l = np.zeros([5])
        self.counter_r = 0
        self.counter_l = 0


    def pd_control(self,observation, kp, kd):

        knee_angle_r = observation[83]
        knee_angle_l = observation[86]
        hip_angle_r = observation[82]
        hip_angle_l = observation[85]

        grf_r = observation[88]
        grf_l = observation[89]

        self.getLengthOfStep(grf_r, grf_l)

        desired_length_r = self.desiredLength(self.time_step_r)
        desired_length_l = self.desiredLength(self.time_step_l)

        current_length_r = self.currentLength(hip_angle_r, knee_angle_r)
        current_length_l = self.currentLength(hip_angle_l, knee_angle_l)

        error_r = desired_length_r-current_length_r
        error_l = desired_length_l-current_length_l

        derivative_r = (error_r - self.previous_error_r) / 0.01
        derivative_l = (error_l - self.previous_error_l) / 0.01

        action_r = kp * error_r + kd * derivative_r
        action_l = kp * error_l + kd * derivative_l

        action_hip_r = (self.leverArm_Hip*action_r)
        action_knee_r = (self.leverArm_Knee*action_r)

        action_hip_l = (self.leverArm_Hip*action_l)
        action_knee_l = (self.leverArm_Knee*action_l)

        return action_hip_r,action_knee_r, action_hip_l, action_knee_l

    def desiredLength(self,steptime, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, w):

        time = steptime

        if steptime > self.t_max_new_r:
            time = self.t_max_new_r

        # w =   5.223 * (1.203 / self.t_max_new_r)

        # a0_result = -5.634e-05
        # a1_result = 7.57
        # b1_result = -21.05
        # a2_result = -7.53
        # b2_result = -9.135
        # a3_result = -2.927
        # b3_result = -1.084
        # a4_result = -0.1472
        # b4_result = -0.002
        # a5_result = 0.3847
        # b5_result = -0.103
        # a6_result = -0.6134
        # b6_result = 0.3526
        # a7_result = -0.05978
        # b7_result = 0.3409
        # a8_result = 0.01217
        # b8_result = -0.04883

        a0_result = a0
        a1_result = a1
        b1_result = b1
        a2_result = a2
        b2_result = b2
        a3_result = a3
        b3_result = b3
        a4_result = a4
        b4_result = b4
        a5_result = a5
        b5_result = b5
        a6_result = a6
        b6_result = b6
        a7_result = a7 
        b7_result = b7
        a8_result = a8
        b8_result = b8


        result = a0_result + a1_result*np.cos(time*w) + b1_result*np.sin(time*w) +\
                        a2_result*np.cos(2*time*w) + b2_result*np.sin(2*time*w) + \
                        a3_result*np.cos(3*time*w) + b3_result*np.sin(3*time*w) + \
                        a4_result*np.cos(4*time*w) + b4_result*np.sin(4*time*w) +\
                        a5_result*np.cos(5*time*w) + b5_result*np.sin(5*time*w) + \
                        a6_result*np.cos(6*time*w) + b6_result*np.sin(6*time*w) + \
                        a7_result*np.cos(7*time*w) + b7_result*np.sin(7*time*w) + \
                        a8_result*np.cos(8*time*w) + b8_result*np.sin(8*time*w)
        
        desired_length =  0

        if result<=0 :
            desired_length = result/(0.71*400)
        else:
            desired_length = result/(1.5*300)

        return desired_length
	
    
    def currentLength(self, hip_angle, knee_angle):

        current_ham = -hip_angle*self.leverArm_Hip + knee_angle * self.leverArm_Knee
        current_rf = hip_angle*self.leverArm_Hip - knee_angle*self.leverArm_Knee

        current_length = current_rf - current_ham

        return current_length

    def changeForm(self, action):

        one = 360 * np.pi/180

        return 1/one * action
    
    def getLengthOfStep(self, grf_r, grf_l):

        #right leg
        if grf_r >= self.threshold and self.bStance_r:
            self.time_step_r+=0.001
    
        if grf_r < self.threshold and self.bStance_r:
            self.bFlight_r = True
            self.bStance_r = False

        if grf_r >= self.threshold and self.bFlight_r:
            self.bFlight_r = False
            self.bStance_r = True
            self.LastSteps_r[self.counter_r] = self.time_step_r
            if self.LastSteps_r.__len__()==5:
                self.t_max_new_r = np.mean(self.LastSteps_r)

            self.time_step_r = 0
            self.counter_r+= 1
            if self.counter_r ==5:
                self.counter_r = 0

        if grf_r < self.threshold and self.bFlight_r:
            self.time_step_r+=0.001


        #left leg
        if grf_l >= self.threshold and self.bStance_l:
            self.time_step_l+=0.001

        if grf_l < self.threshold and self.bStance_l:
            self.bFlight_l = True
            self.bStance_l = False

        if grf_l >= self.threshold and self.bFlight_l:
            self.bFlight_l = False
            self.bStance_l = True
            self.LastSteps_l[self.counter_l] = self.time_step_l
            if self.LastSteps_l.__len__()==5:
                self.t_max_new_l = np.mean(self.LastSteps_l)

            self.time_step_l = 0
            self.counter_l= 1
            if self.counter_l ==5:
                self.counter_l = 0

        if grf_l < self.threshold and self.bFlight_l:
            self.time_step_l+=0.001

        

            

