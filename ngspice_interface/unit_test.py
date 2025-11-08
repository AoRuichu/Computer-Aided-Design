import time
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as sciopt
from scipy.optimize import differential_evolution
from .dut_testbench import DUT
from .ngspice_wrapper import NgspiceWrapper
from .area_estimation import BPTM45nmAreaEstimator


if __name__ == "__main__":
    project_path = os.getcwd()
    yaml_path = os.path.join(project_path, 'ngspice_interface', 'files', 'yaml_files', 'TwoStage.yaml')
    rf_designs = {}
    rf_designs[0] = {
        "parameters": {
            'mp1': 6, 'wp1': 2.25e-06, 'lp1': 1.35e-07,
            'mn1': 10, 'wn1': 5e-07, 'ln1': 1.35e-07,
            'mp3': 10, 'wp3': 1.75e-06, 'lp3': 9e-08,
            'mn3': 1, 'wn3': 1e-06, 'ln3': 1.35e-07,
            'mn4': 9, 'wn4': 1.25e-06, 'ln4': 1.35e-07,
            'mn5': 6, 'wn5': 7.5e-07, 'ln5': 9e-08,
            'cap': 8e-13, 'res': 9500.0
        },
        "specifications": {
            'area': 9.773778452380953e-10,
            'current': 0.00050714721,
            'gain': 817.2324500046147,
            'noise': 0.039446578,
            'phm': 47.8961300512141,
            'slewRate': 3.9350095214166356,
            'ugbw': 287939270.26242465,
            'reward': -0.837118226303924
        }
    }
    rf_designs[1] = {
        "parameters": {
            'mp1': 15, 'wp1': 1.25e-06, 'lp1': 9e-08,
            'mn1': 10, 'wn1': 1e-06, 'ln1': 9e-08,
            'mp3': 12, 'wp3': 1.25e-06, 'lp3': 9e-08,
            'mn3': 15, 'wn3': 1.5e-06, 'ln3': 9e-08,
            'mn4': 11, 'wn4': 1.5e-06, 'ln4': 9e-08,
            'mn5': 11, 'wn5': 1e-06, 'ln5': 9e-08,
            'cap': 4.5e-12, 'res': 4800.0
        },
        "specifications": {
            'area': 1.8939731428571428e-09,
            'current': 6.3062567e-05,
            'gain': 89.637909012264,
            'noise': 0.0028557284,
            'phm': 52.440502130862455,
            'slewRate': 4.553232027350708,
            'ugbw': 4172793.7507562377,
            'reward': -0.9551500268343925
        }
    }

    rf_designs[2] = {
        "parameters": {
            'mp1': 7, 'wp1': 1.75e-06, 'lp1': 1.35e-07,
            'mn1': 20, 'wn1': 2.25e-06, 'ln1': 1.35e-07,
            'mp3': 22, 'wp3': 2.25e-06, 'lp3': 1.35e-07,
            'mn3': 21, 'wn3': 1e-06, 'ln3': 9e-08,
            'mn4': 8, 'wn4': 7.5e-07, 'ln4': 9e-08,
            'mn5': 17, 'wn5': 2.5e-07, 'ln5': 4.5e-08,
            'cap': 8e-13, 'res': 3900.0
        },
        "specifications": {
            'area': 6.338428452380952e-10,
            'current': 5.8465647e-05,
            'gain': 151.1023500114643,
            'noise': 0.0023655671,
            'phm': 44.35755013884386,
            'slewRate': 6.2056536666667075,
            'ugbw': 8581284.483782556,
            'reward': -0.6505644390925518
        }
    }

    rf_designs[3] = {
        "parameters": {
            'mp1': 8, 'wp1': 1.5e-06, 'lp1': 1.35e-07,
            'mn1': 15, 'wn1': 1.75e-06, 'ln1': 1.35e-07,
            'mp3': 18, 'wp3': 2e-06, 'lp3': 1.35e-07,
            'mn3': 15, 'wn3': 1.25e-06, 'ln3': 9e-08,
            'mn4': 13, 'wn4': 1.5e-06, 'ln4': 9e-08,
            'mn5': 17, 'wn5': 5e-07, 'ln5': 4.5e-08,
            'cap': 3.5e-12, 'res': 6000.0
        },
        "specifications": {
            'area': 1.666403095238095e-09,
            'current': 0.00011065298,
            'gain': 188.3550900311509,
            'noise': 0.0030385342,
            'phm': 103.4538075980825,
            'slewRate': 6.219140771079,
            'ugbw': 14700608.521007696,
            'reward': -0.4248731689770592
        }
    }
    rf_designs[4] = {
        "parameters": {
            'mp1': 12, 'wp1': 7.5e-07, 'lp1': 1.35e-07,
            'mn1': 15, 'wn1': 1.5e-06, 'ln1': 1.35e-07,
            'mp3': 10, 'wp3': 1.5e-06, 'lp3': 9e-08,
            'mn3': 19, 'wn3': 1.75e-06, 'ln3': 9e-08,
            'mn4': 13, 'wn4': 1.25e-06, 'ln4': 9e-08,
            'mn5': 13, 'wn5': 1e-06, 'ln5': 4.5e-08,
            'cap': 2.4e-12, 'res': 7500.0
        },
        "specifications": {
            'area': 1.3967347857142855e-09,
            'current': 7.4730222e-05,
            'gain': 54.21959600381273,
            'noise': 0.00101217,
            'phm': 96.87316631143796,
            'slewRate': 4.065660558268217,
            'ugbw': 4747450.820930383,
            'reward': -1.1152736676397283
        }
    }

    process = "TT"
    temp_pvt = 27
    vdd = 1.2
    dut = DUT(yaml_path)
    

    for i in range(len(rf_designs)):
        print("\n========================================")
        print("===== Reference Design Set:", i,"=======")
        print("========================================")
        new_netlist_path = dut.create_new_netlist(rf_designs[i]["parameters"], process, temp_pvt, vdd)
        info = dut.simulate(new_netlist_path)
        print(f"\nNew netlist created at: {new_netlist_path}")
        print("info:", info)
        print("trf:", dut.trf)
        print("period:", dut.period)
        print("VDD:", dut.VDD)

        spec_dict = dut.measure_metrics()
        print("\n===== DUT testbench, RadomName:", dut.random_name,"=======")
        print("Area:", spec_dict['area'],"m^2")
        print("Current:",spec_dict['current'],"A")
        print("Gain:", spec_dict['gain'],"dB")
        print("Noise:",spec_dict['noise'],"V")
        print('Phase Margin:',spec_dict['phm'],"degree")
        print("Slew rate:", spec_dict['slewRate'],"V/µs")
        print("UGBW:",spec_dict['ugbw'],"Hz")
        
        print("\n=== COMPARISON WITH REFERENCE TABLE ===")
        for key in rf_designs[i]["specifications"]:
            if key in spec_dict:
                ref = rf_designs[i]["specifications"][key]
                meas = spec_dict[key]
                error = (meas - ref) / ref * 100 if ref != 0 else float('nan')
                print(f"{key:10s}: Measured = {meas}, Reference = {ref}, Error = {error:+.2f}%")
    

    