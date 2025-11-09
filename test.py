import time
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as sciopt
from scipy.optimize import differential_evolution
# from .dut_testbench import DUT
# from .ngspice_wrapper import NgspiceWrapper
# from .area_estimation import BPTM45nmAreaEstimator
from circuit_env import CircuitEnv
import unittest


# ========= 你的参考设计，原样贴进来即可 =========
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

PVT = {"process": "TT", "temp": 27, "vdd": 1.2}


class TestCircuitEnvIntegration(unittest.TestCase):
    """
    真实 ngspice 集成测试：
    - 参数 -> 动作 -> 仿真 -> 归一化(手动) -> 奖励(手动) 对比 step() 返回的奖励
    - 校验 action 映射的往返一致性
    """
    # 由于手动公式 = 环境公式，误差应只来自数值/排序等细节
    TOL_NORM = 1e-6       # 归一化误差容忍（几乎应为 0）
    TOL_REWARD = 1e-6      # 奖励误差容忍（同公式时应为 0）

    def setUp(self):
        self.env = CircuitEnv(
            circuit_name='TwoStage',
            run_id=0,
            simulator='ngspice',
            success_threshold=0.0
        )
        # 固定 PVT
        self.env.pvt_corner = {'process': 'TT', 'voltage': 1.2, 'temp': 27}

        # 为了保证键顺序一致（与 env 内部构造 observation 一致）
        self.param_order = list(self.env.dict_params.keys())
        self.target_order = list(self.env.dict_targets.keys())

    # ---------- 工具函数 ----------
    def params_to_action(self, params: dict) -> np.ndarray:
        """真实参数 → 归一化动作 [-1, 1]，顺序按 env.dict_params"""
        a = []
        for name in self.param_order:
            pmin = self.env.param_ranges[name]['min']
            pmax = self.env.param_ranges[name]['max']
            norm = 2 * (params[name] - pmin) / (pmax - pmin) - 1
            a.append(np.clip(norm, -1, 1))
        return np.array(a, dtype=np.float32)

    def manual_normalize(self, specs: dict, targets: dict) -> dict:
        """按要求手动归一化 (spec - goal) / (spec + goal)"""
        out = {}
        for k in targets.keys():
            s = float(specs[k])
            g = float(targets[k])
            if (s + g) == 0:
                out[k] = 0.0
            else:
                out[k] = (s - g) / (s + g)
        return out

    def manual_reward(self, norm_specs: dict) -> float:
        """
        手动奖励计算：
        - 硬约束：noise → -max(norm, 0)；其他 → min(norm, 0)，累加为 r_H
        - 目标优化：r_T = - Σ(norm[optimization_targets])
        - 若 r_H >= success_threshold：reward = 0.3 + r_T
          否则：reward = r_H + 0.05 * r_T
        """
        r_H = 0.0
        r_T = 0.0

        # 硬约束
        for name in self.env.hard_constraints:
            v = norm_specs[name]
            if name == "noise":
                r_H += -max(v, 0.0)
            else:
                r_H += min(v, 0.0)

        # 目标优化
        for name in self.env.optimization_targets:
            r_T += -norm_specs[name]

        if r_H >= self.env.success_threshold:
            return 0.3 + r_T
        else:
            return r_H + 0.05 * r_T

    # ---------- 测试项 ----------
    def test_roundtrip_action_mapping(self):
        """参数 -> 动作 -> action_refine → 参数：往返映射应接近原值"""
        for idx, design in rf_designs.items():
            with self.subTest(design=idx):
                action = self.params_to_action(design["parameters"])
                refined = self.env.action_refine(action)
                for k in self.param_order:
                    self.assertAlmostEqual(
                        float(refined[k]), float(design["parameters"][k]),
                        delta=self.TOL_NORM,
                        msg=f"Roundtrip mismatch on {k} for design {idx}"
                    )

    def test_manual_norm_and_reward_match_step(self):
        """
        调用真实 evaluate/step（内部会 simulate）得到真实 specs，
        用手动公式归一化和计算奖励，并与 step 返回的 reward 对比。
        """
        for idx, design in rf_designs.items():
            with self.subTest(design=idx):
                action = self.params_to_action(design["parameters"])

                # 调 step：产生 reward（以及 evaluate/normalize）
                ob, reward_from_step, done, info = self.env.step(action)

                # 使用“真实测量到的”specs 做手动验证
                measured_specs = self.env.real_specs           # 来自 simulate()
                targets = self.env.dict_targets

                # 1) 归一化手算
                manual_norm = self.manual_normalize(measured_specs, targets)

                # 2) 与环境当前归一化结果逐项对比
                env_norm = self.env.cur_norm_specs
                for k in self.target_order:
                    self.assertAlmostEqual(
                        float(manual_norm[k]), float(env_norm[k]),
                        delta=self.TOL_NORM,
                        msg=f"Normalized spec mismatch on '{k}' for design {idx}"
                    )

                # 3) 奖励手算
                manual_reward_val = self.manual_reward(manual_norm)

                # 4) 与 step() 的 reward 对比
                self.assertAlmostEqual(
                    float(manual_reward_val), float(reward_from_step),
                    delta=self.TOL_REWARD,
                    msg=f"Reward mismatch for design {idx}: manual={manual_reward_val}, step={reward_from_step}"
                )

                # 5) 基本有效性检查
                self.assertTrue(np.all(np.isfinite(ob)))
                self.assertIsInstance(info.get("goal", False), bool)

if __name__ == "__main__":
    unittest.main()
#RUN_REAL_SPICE=1 /users/micas/mahmadza/public/bin/cad_env/bin/python /users/students/r0926886/CAD/circuit_env_test.py