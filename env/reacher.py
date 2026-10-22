"""Goal-free dm_control Reacher with the paper's two-substep control clock."""
from pathlib import Path

import numpy as np
from dm_control import mjcf
from dm_control.suite import reacher


class ReacherEnv:
    """Explicit physical parameters and recorded qpos/qvel; no success freezing."""

    def __init__(self, physics, render_size=224):
        root = mjcf.from_path(Path(reacher.__file__).with_suffix('.xml'))
        root.find('geom', 'target').remove()
        root.find('geom', 'arm').fromto = (0, 0, 0, physics['link1_len'], 0, 0)
        root.find('geom', 'arm').mass = physics['link1_mass']
        root.find('body', 'hand').pos = (physics['link1_len'], 0, 0)
        root.find('geom', 'hand').fromto = (0, 0, 0, physics['link2_len'], 0, 0)
        root.find('geom', 'hand').mass = physics['link2_mass']
        root.find('body', 'finger').pos = (physics['link2_len'] + .02, 0, 0)
        root.find('joint', 'shoulder').damping = physics['joint1_damping']
        root.find('joint', 'wrist').damping = physics['joint2_damping']
        self.physics = mjcf.Physics.from_mjcf_model(root)
        self.render_size = render_size
        joints = [self.physics.model.name2id(name, 'joint') for name in ('shoulder', 'wrist')]
        self.qpos_indices = self.physics.model.jnt_qposadr[joints]
        self.qvel_indices = self.physics.model.jnt_dofadr[joints]

    def state(self):
        return np.concatenate((self.physics.data.qpos[self.qpos_indices],
                               self.physics.data.qvel[self.qvel_indices])).astype(np.float32)

    def render(self):
        return self.physics.render(height=self.render_size, width=self.render_size, camera_id=0)

    def reset(self, state):
        state = np.asarray(state, dtype=np.float32)
        self.physics.reset()
        self.physics.data.qpos[self.qpos_indices] = state[:2]
        self.physics.data.qvel[self.qvel_indices] = state[2:]
        self.physics.forward()
        return self.render(), self.state()

    def step(self, action):
        self.physics.data.ctrl[:] = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        for _ in range(2):
            self.physics.step()
        return self.render(), self.state()
