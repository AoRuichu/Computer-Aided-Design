import time
import numpy as np
import os
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
from scipy.optimize import differential_evolution
from .area_estimation import BPTM45nmAreaEstimator
from .ngspice_wrapper import NgspiceWrapper


class DUT(NgspiceWrapper):
    def measure_metrics(self):
        self.parse_outputs()
        spec_dict = {}
        # post process raw data
        area_estimator = BPTM45nmAreaEstimator(self.circuit_params, self.circuit_multipliers)
        spec_dict['area'] = area_estimator.find_area()
        spec_dict['current'] = -self.current
        spec_dict['gain'] = self.find_dc_gain(self.vout_complex)
        spec_dict['noise'] = self.noise
        spec_dict['phm'] = self.find_phm(self.freq, self.vout_complex)
        spec_dict['slewRate'] = self.find_slew_rate(self.time, self.vout_tran, threshold_low=0.1, threshold_high=0.9, time_unit='us')
        spec_dict['ugbw'] = self.find_ugbw(self.freq, self.vout_complex)
        
        return spec_dict
    
    def parse_outputs(self):

        tran_fname = os.path.join(self.output_files_folder, 'tran_'+self.random_name+'.csv')
        ac_fname = os.path.join(self.output_files_folder, 'ac_'+self.random_name+'.csv')
        dc_fname = os.path.join(self.output_files_folder, 'dc_'+self.random_name+'.csv')
        noise_fname = os.path.join(self.output_files_folder, 'noise_'+self.random_name+'.csv')
        # add these file names in a list
        self.output_files = [tran_fname, ac_fname, dc_fname, noise_fname]
        for file in self.output_files:
            if not os.path.isfile(file):
                print(f"{file} doesn't exist")
        tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=1)
        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        noise_raw_outputs = np.genfromtxt(noise_fname, skip_header=1)

        self.time = tran_raw_outputs[:, 0]
        self.vout_tran = tran_raw_outputs[:, 1]
        
        self.freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        self.vout_complex = vout_real + 1j*vout_imag
        
        self.current = - dc_raw_outputs[1]

        self.noise = noise_raw_outputs[0]
    
    def find_dc_gain(self, vout):
        """
        TODO: Implement the DC gain calculation
        
        Hint:
        Use numpy's abs() function to calculate the magnitude of the complex number at each point.
        """
        return np.abs(vout[0])
        
        
        
    def find_ugbw(self, freq, vout):
        """
        TODO: Implement the unity gain bandwidth (UGBW) calculation
        
        Hints:
        1. Calculate the magnitude of vout
        2. Find where the magnitude crosses 1 (unity gain)
        3. Use _get_best_crossing() to find the crossing point through interpolation
        4. What should you if no crossing is found? What situations can lead to this?
        """
        mag = np.abs(vout)
        
        ugbw,success = self._get_best_crossing(freq,mag,1.0)
        
        if ugbw is None or not success or np.isnan(ugbw): 
                return 0.0 # return 0 if no crossing found
            
        return ugbw 
    
    def find_phm(self, freq, vout):
        """
        TODO: Implement the phase margin (PHM) calculation
        
        Hints:
        1. Calculate gain array and phase array from vout
        2. Find the unity gain frequency (UGBW)
        3. Interpolate to find the phase at UGBW (you can use interp.interp1d quadratic interpolation)
        4. Calculate phase margin (watch out for radians/degrees units and phase wrap around)
        5. Handle edge cases (e.g., when gain is always < 1) --> hint: you can think in RL terms; worst case reward ...
        """
        gain = np.abs(vout) 
        phase = np.angle(vout, deg=True)
        
        ugbw = self.find_ugbw(freq,vout)
        if ugbw == 0.0 or np.isnan(ugbw):
                return 0.0 # return 0 if no UGBW found

        f_interp = interp.interp1d(freq,phase,kind='quadratic', fill_value="extrapolate")
        phase_at_ugbw = f_interp(ugbw)
        
        phm= 180 + phase_at_ugbw
        
         # Normalize phase margin into [0, 180]
        phm = np.mod(phm+360,180)
        if phm>180:
            phm = 360-phm
        return phm
         
         
    
    def find_slew_rate(self, time, signal, threshold_low=0.1, threshold_high=0.9, time_unit='us'):
        """
        TODO: Implement the slew rate calculation
        
        Hints:
        1. Find large rising edges in the signal
        2. Calculate slope for each rising edge
        3. Take the average of these slopes
        5. Handle edge cases (e.g., no rising edges found)
        6. Final value should be in V/us
        """
        low_level = np.percentile(signal,5)
        high_level = np.percentile(signal,95)
        
        sig_norm = (signal - low_level) / (high_level-low_level)
        sig_norm = np.clip(sig_norm,0,1)
        
        t_low = self._get_all_corssing_at_risedge(time, sig_norm, threshold_low)
        t_high = self._get_all_corssing_at_risedge(time, sig_norm, threshold_high)
        
        # print("t_low:",t_low)
        # print("t_high:",t_high)

        if len(t_low) == 0 or len(t_high) == 0:
            return np.nan
        
        dv = (threshold_high - threshold_low)*(high_level-low_level)
        n = min(len(t_low), len(t_high))
        if n == 0:
            return np.nan  
        dt = np.array(t_high[:n]) - np.array(t_low[:n])
        slew = np.mean(dv / dt)  # V/s
        if time_unit == 'us':
            slew /= 1e6  # V/us
            
        
        # fig, ax1 = plt.subplots(figsize=(10, 5))

        # # Left y-axis: original signal
        # ax1.plot(time, signal, label='Original Signal', color='tab:blue', linewidth=1.5)
        # ax1.axhline(low_level, color='tab:blue', linestyle='--', alpha=0.6, label='Low Level (5%)')
        # ax1.axhline(high_level, color='tab:blue', linestyle=':', alpha=0.6, label='High Level (95%)')
        # ax1.set_xlabel("Time [s]")
        # ax1.set_ylabel("Voltage [V]", color='tab:blue')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')
        # ax1.grid(True, linestyle='--', alpha=0.5)

        # # Right y-axis: normalized signal
        # ax2 = ax1.twinx()
        # ax2.plot(time, sig_norm, label='Normalized Signal', color='tab:orange', linewidth=1.5, alpha=0.8)
        # ax2.axhline(threshold_low, color='orange', linestyle='--', alpha=0.5, label=f'{threshold_low*100:.0f}% level')
        # ax2.axhline(threshold_high, color='orange', linestyle=':', alpha=0.5, label=f'{threshold_high*100:.0f}% level')
        # ax2.set_ylabel("Normalized Value", color='tab:orange')
        # ax2.tick_params(axis='y', labelcolor='tab:orange')

        # # Combine legends
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines + lines2, labels + labels2, loc='best')

        # plt.title("Signal vs Normalized Signal (for Slew Rate Calculation)")
        # plt.tight_layout()
        # plt.show()

        return slew

    # look for single xvec when yvec == val 
    def _get_best_crossing(cls, xvec, yvec, val): 
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            return xstop, False
    
    # look for all crossing at rising edge
    def _get_all_corssing_at_risedge(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec,yvec)
        def fzero(x):
            return interp_fun(x) - val
        crossings = []
        for i in range(len(xvec)-1):
            if fzero(xvec[i])*fzero(xvec[i+1])<0 and yvec[i]<yvec[i+1]:
                root = sciopt.brentq(fzero,xvec[i],xvec[i+1]) #inding the exact time at crossing point
                crossings.append(root)
        return crossings        
    