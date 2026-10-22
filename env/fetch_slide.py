"""FetchSlide physics and rendering for the released XY retraction collection."""

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np


gym.register_envs(gymnasium_robotics)


class FetchSlideEnv:
    def __init__(self, physics, render=True):
        options = dict(render_mode='rgb_array', width=224, height=224) if render else {}
        self.env = gym.make('FetchSlide-v4', **options)
        self.base = self.env.unwrapped
        self.model, self.data = self.base.model, self.base.data
        model = self.model
        mass = physics['mass']
        radial = mass * (3 * .05**2 + .04**2) / 12
        model.body('object0').mass[:] = mass
        model.body('object0').inertia[:] = [radial, radial, mass * .05**2 / 2]
        model.geom('object0').size[0] = .05
        for name in ('object0', 'table0'):
            model.geom(name).friction[0] = physics['friction']
        model.opt.density, model.opt.viscosity = physics['fluid_density'], 0.0
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        model.mat('puck_mat').rgba[:] = [.68, .28, .23, 1.0]
        model.mat('table_mat').rgba[:] = [.94, .935, .92, 1.0]
        for name in ('object0', 'target0'):
            model.site(name).rgba[3] = 0.0
        model.geom_rgba[model.geom_bodyid == model.body('robot0:mocap').id, 3] = 0.0
        pixels = int(model.tex_height[0] * model.tex_width[0])
        model.tex_data[:3 * pixels] = np.tile([216, 204, 188], pixels)
        mujoco.mj_setConst(model, self.data)
        if render:
            self.env.reset(seed=18)
            self.base.mujoco_renderer.default_cam_config = dict(
                distance=1.35, azimuth=180.0, elevation=-46.0,
                lookat=np.array([1.49438680, .75, .60610683]))
            self.env.render()
            viewer = self.base.mujoco_renderer._viewers['rgb_array']
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    def reset(self):
        """Prepare the deterministic pre-hit pose without changing the puck."""
        self.base.reset(seed=18)
        model, data = self.model, self.data
        position = data.joint('object0:joint').qpos
        table_y = float(data.geom('table0').xpos[1])
        position[:2] = [.80, table_y]
        position[3:] = [1.0, 0.0, 0.0, 0.0]
        data.joint('object0:joint').qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        fixed = position.copy()
        hit = np.array([.80 - .08, table_y, position[2]])
        for _ in range(30):
            gripper = self.base._get_obs()['observation'][:3]
            action = np.r_[np.clip((hit - gripper) / .05, -1.0, 1.0), 0.0].astype(np.float32)
            self.base.step(action)
        position[:] = fixed
        data.qvel[:] = 0.0
        data.time = 0.0
        mujoco.mj_forward(model, data)
        return self.base._get_obs()

    def render(self):
        self.base.mujoco_renderer._viewers['rgb_array'].make_context_current()
        return self.env.render()

    def render_qposes(self, qposes, with_masks=False):
        images, masks = [], []
        for qpos in qposes:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            images.append(self.render())
            if with_masks:
                viewer = self.base.mujoco_renderer._viewers['rgb_array']
                segmentation = viewer.render('rgb_array', camera_id=-1, segmentation=True)
                mask = ((segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) &
                        (segmentation[..., 1] == self.model.geom('object0').id))
                masks.append(mask.astype(np.uint8))
        rgb = np.stack(images)
        return (rgb, np.stack(masks)) if with_masks else rgb

    def close(self):
        # Free GL buffers while their context is alive, before another env is created.
        for viewer in self.base.mujoco_renderer._viewers.values():
            viewer.make_context_current()
            viewer.con.free()
        self.env.close()

    def robot_puck_contact(self):
        puck = self.model.geom('object0').id
        for contact in self.data.contact[:self.data.ncon]:
            if puck in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == puck else contact.geom1
                if self.model.body(self.model.geom_bodyid[other]).name.startswith('robot0:'):
                    return True
        return False

    def rollout(self, action_xy):
        """Hit twice, wait for substep-level release, lift, park, and retain the tail."""
        action_xy = np.asarray(action_xy, dtype=np.float32)
        tolerance = 2 * np.finfo(np.float32).eps
        if action_xy.shape != (2,) or not np.all(np.isfinite(action_xy)) or np.any(action_xy < np.array([.35, -.8]) - tolerance) or np.any(action_xy > np.array([.85, .8]) + tolerance):
            raise ValueError('FetchSlide hit must be inside [0.35,0.85] x [-0.8,0.8]')
        obs = self.reset()
        initial = obs['observation'].astype(np.float32, copy=True)
        start = self.data.joint('object0:joint').qpos[:2].copy()
        previous = start.copy()
        center = self.data.geom('table0').xpos[:2].copy()
        halfsize = self.model.geom('table0').size[:2].copy()
        park = initial[:3].astype(np.float64) + [-.10, 0.0, .15]
        lift = None
        lifted = False
        integral = np.zeros(3, dtype=np.float64)
        released, parked, still = 0, 0, 0
        onset, retracted, terminal_step, completion = -1, -1, -1, -1
        launched = False
        terminal = 'watchdog'
        states, actions, qposes, contacts = [], [], [], []
        grip_site = self.model.site('robot0:grip').id
        velocity = np.zeros(6)

        def append(action):
            states.append(obs['observation'].astype(np.float32, copy=True))
            actions.append(np.asarray(action, dtype=np.float32))
            qposes.append(self.data.qpos.copy())
            contacts.append(self.robot_puck_contact())

        for step in range(1200):
            action = np.r_[action_xy, 0.0, 0.0].astype(np.float32) if step < 2 else np.zeros(4, dtype=np.float32)
            if step >= 2 and (lift is not None or released >= 3):
                if lift is None:
                    onset = step
                    lift = obs['observation'][:3].copy()
                    lift[2] = park[2]
                if not lifted and abs(obs['observation'][2] - park[2]) < .001:
                    lifted = True
                    integral.fill(0.0)
                error = (park if lifted else lift) - obs['observation'][:3]
                update = np.abs(error) < .025
                integral[update] = np.clip(integral[update] + .2 * error[update], -.02, .02)
                action = np.r_[np.clip((error + integral) / .05, -1.0, 1.0), 0.0].astype(np.float32)
            append(action)
            if lift is None:
                self.base._set_action(action)
                ready = True
                for _ in range(self.base.n_substeps):
                    mujoco.mj_step(self.model, self.data)
                    mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, grip_site, velocity, 0)
                    ready &= not self.robot_puck_contact() and np.linalg.norm(velocity[3:]) < .01
                self.base._step_callback()
                obs = self.base._get_obs()
                if step + 1 >= 2:
                    released = released + 1 if ready else 0
            else:
                obs, _, _, _, _ = self.base.step(action)
            at_park = lifted and np.linalg.norm(obs['observation'][:3] - park) <= .001
            parked = parked + 1 if at_park else 0
            if parked >= 5 and retracted < 0:
                retracted = step + 1
            position = self.data.joint('object0:joint').qpos[:2].copy()
            delta = float(np.linalg.norm(position - previous))
            previous = position
            launched |= bool(np.linalg.norm(position - start) > .001)
            still = still + 1 if step >= 2 and launched and delta < 5e-4 else 0
            reason = None
            if position[0] >= center[0] + halfsize[0] or abs(position[1] - center[1]) >= halfsize[1]:
                reason = 'off_table'
            elif still >= 2:
                reason = 'settled'
            elif step >= 12 and not launched:
                reason = 'no_hit'
            if reason is not None and terminal_step < 0:
                terminal, terminal_step = reason, len(actions)
            if retracted < 0 and step - 2 + 1 >= 100:
                raise RuntimeError('FetchSlide retraction reached its watchdog limit')
            if terminal_step < 0 or retracted < 0:
                continue
            if completion < 0:
                completion = len(actions)
            if len(actions) - completion >= 2:
                append(np.zeros(4, dtype=np.float32))
                break
        if terminal_step < 0 or retracted < 0:
            raise RuntimeError('FetchSlide rollout did not complete')
        return dict(qpos=np.asarray(qposes), state=np.asarray(states), action=np.asarray(actions),
                    initial_state=initial, final_state=states[-1], terminal=terminal, terminal_step=terminal_step,
                    retraction_start_step=onset, retraction_step=retracted, park_position=park,
                    robot_contacts=np.asarray(contacts), endpoint_displacement_m=(position-start).tolist())
