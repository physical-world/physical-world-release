"""Aligned-v2 BeerPong geometry and the released Cartesian throw controller."""

from pathlib import Path

import mujoco
import numpy as np


START_QPOS = np.array([0., 1.2124367524428588, 0., 1.8290572268339087,
                       0., -1.297493979276767, -1.59])
ACTIVE_JOINTS = [1, 3, 5]
CUP_GEOMS = [f'cup_geom_table{i}' for i in (3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17)]
CUP_GEOMS.append('cup_geom1_table8')


class BeerPongEnv:
    def __init__(self, physics):
        xml = Path(__file__).parent / 'assets/beerpong/beerpong_wo_cup_big_table.xml'
        spec = mujoco.MjSpec.from_file(str(xml))
        bodies = {body.name: body for body in spec.bodies}
        geoms = {geom.name: geom for geom in spec.geoms}
        bodies['wam/base_link'].pos[1] = -.15
        geoms['table_robot'].pos[1] = -.15
        bodies['table_body'].pos[1] = -1.85
        for name in ('table', 'table_contact_geom'):
            geoms[name].size[:2] = [.45, .83984375]
        geoms['table'].size[2] = .41
        geoms['table'].rgba[:] = [.94, .935, .92, 1.]
        geoms['table'].contype = 0
        geoms['table'].conaffinity = 0
        geoms['table_contact_geom'].rgba[3] = 0.
        geoms['table_robot'].rgba[:] = [.2, .2, .2, 1.]
        room_color = [216 / 255, 204 / 255, 188 / 255, 1.]
        wall_y = -3.68984375
        geoms['wall'].pos[:] = [0., wall_y, 1.35]
        geoms['wall'].size[:] = [2.5, .04, 1.4]
        geoms['wall'].rgba[:] = room_color
        for name, x in (('room_left_wall', -2.5), ('room_right_wall', 2.5)):
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, (wall_y - .45) / 2, 1.35],
                size=[.04, (-.45 - wall_y) / 2, 1.4], rgba=room_color,
                contype=0, conaffinity=0)
        geoms['ball_geom'].size[:] = .049
        geoms['ball_geom'].rgba[:] = [.68, .28, .23, 1.]
        site = spec.find_site('init_ball_pos')
        site.pos[2] += .049 - .025 + .040
        site.rgba[3] = 0.
        for mesh in spec.meshes:
            if mesh.name.startswith('cup'):
                mesh.scale[:2] *= .10 / .07
        geoms['cup_base_table'].size[0] *= .10 / .07
        geoms['cup_base_table_contact'].size[0] = .10
        for name in CUP_GEOMS:
            geoms[name].rgba[3] = .30
            geoms[name].priority = 0
        geoms['cup_base_table'].rgba[3] = .15
        geoms['cup_base_table_contact'].rgba[3] = 0.
        for index in (0, 2, 4, 6):
            equality = spec.add_equality(name=f'lock_joint_{index}')
            equality.type = mujoco.mjtEq.mjEQ_JOINT
            equality.name1 = spec.joints[index].name
            equality.data[0] = START_QPOS[index]
            equality.solref = [.002, 1.]
            equality.solimp = [.9999, .9999, .001, .5, 2.]
        self.model = model = spec.compile()
        model.vis.quality.shadowsize = 0
        self.data = mujoco.MjData(model)
        model.body('cup_table').pos[:2] = [0., physics['cup_body_y']]
        model.geom('ball_geom').solref[:] = [-10000., -physics['contact_damping']]
        mujoco.mj_setConst(model, self.data)
        self.cup_ids = {model.geom(name).id for name in CUP_GEOMS}
        self._align_cup_axis()
        self.gear = model.actuator_gear[:, 0].copy()
        self.site_id = model.site('init_ball_pos').id
        self.jacobian = np.zeros((3, model.nv))
        self.rot_jacobian = np.zeros((3, model.nv))
        self.renderer = None

    def _align_cup_axis(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        vertices = []
        for geom_id in sorted(self.cup_ids):
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            geom = data.geom(geom_id)
            vertices.append(model.mesh_vert[start:start + count]
                            @ geom.xmat.reshape(3, 3).T + geom.xpos)
        vertices = np.concatenate(vertices)
        rim = vertices[vertices[:, 2] >= vertices[:, 2].max() - .002, :2]
        axis = (rim.min(axis=0) + rim.max(axis=0)) / 2
        body = data.body('cup_table')
        for kind, names in (('geom', ('cup_base_table', 'cup_base_table_contact')),
                            ('site', ('cup_goal_table', 'cup_goal_final_table'))):
            for name in names:
                world = getattr(data, kind)(name).xpos.copy()
                world[:2] = axis
                getattr(model, kind)(name).pos[:] = body.xmat.reshape(3, 3).T @ (world - body.xpos)
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = START_QPOS
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[7:10] = self.data.site('init_ball_pos').xpos
        self.data.qvel[7:10] = 0.
        mujoco.mj_forward(self.model, self.data)
        return np.r_[self.data.qpos, self.data.qvel].copy()

    def render_qposes(self, qposes, with_masks=False):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=224, width=224)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0., -1.48, .88]
        camera.distance, camera.azimuth, camera.elevation = 2.2, 220., -28.
        images, masks = [], []
        for qpos in qposes:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = 0.
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=camera)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            images.append(self.renderer.render().copy())
            if with_masks:
                self.renderer.enable_segmentation_rendering()
                self.renderer.update_scene(self.data, camera=camera)
                segmentation = self.renderer.render()
                self.renderer.disable_segmentation_rendering()
                mask = ((segmentation[..., 0] == self.model.geom('ball_geom').id) &
                        (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
                masks.append(mask.astype(np.uint8))
        rgb = np.stack(images)
        return (rgb, np.stack(masks)) if with_masks else rgb

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def raw_control(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or np.any(action < [2.5, 6.]) or np.any(action > [3., 8.]):
            raise ValueError('BeerPong action must be within [2.5, 6] and [3, 8]')
        mujoco.mj_jacSite(self.model, self.data, self.jacobian,
                         self.rot_jacobian, self.site_id)
        desired = np.linalg.pinv(self.jacobian[[1, 2]][:, ACTIVE_JOINTS], rcond=1e-4) @ [-action[0], action[1]]
        command = np.array([.10, .08, .03]) * (desired - self.data.qvel[ACTIVE_JOINTS])
        raw = np.zeros(7)
        raw[ACTIVE_JOINTS] = np.clip(command, -1., 1.)
        return raw

    def physics_step(self, raw, tether_ball=False):
        self.data.ctrl[:] = raw + self.data.qfrc_bias[:7] / self.gear
        mujoco.mj_step(self.model, self.data)
        if tether_ball:
            self.data.qpos[7:10] = self.data.site('init_ball_pos').xpos
            self.data.qvel[7:10] = self.data.sensor('init_ball_vel').data

    def release(self, action):
        self.reset()
        qpos, qvel, controls = [self.data.qpos.copy()], [self.data.qvel.copy()], []
        max_lock_drift = 0.0
        for _ in range(5):
            raw = self.raw_control(action)
            controls.append(raw)
            for _ in range(2):
                self.physics_step(raw, tether_ball=True)
                drift = np.max(np.abs(self.data.qpos[[0, 2, 4, 6]] - START_QPOS[[0, 2, 4, 6]]))
                max_lock_drift = max(max_lock_drift, float(drift))
            qpos.append(self.data.qpos.copy())
            qvel.append(self.data.qvel.copy())
        return dict(qpos=np.stack(qpos), qvel=np.stack(qvel), controls=np.stack(controls),
                    max_locked_joint_drift_rad=max_lock_drift)

    def in_cup(self, pairs):
        ball = self.model.geom('ball_geom').id
        base = self.model.geom('cup_base_table_contact').id
        radial = np.linalg.norm(self.data.geom(ball).xpos[:2] - self.data.geom(base).xpos[:2])
        return {ball, base} in pairs and radial <= .10

    def rollout(self, action):
        released = self.release(action)
        qpos, qvel = list(released['qpos']), list(released['qvel'])
        ball = self.model.geom('ball_geom').id
        ground = self.model.geom('ground').id
        table = self.model.geom('table_contact_geom')
        ever_in_cup, off_table, entered_airspace = False, False, False
        contact_history = []
        first_in_step = first_in_substep = first_off_step = first_off_substep = -1
        table_bounces, previous_table_contact = 0, False
        off_direction = None
        for control_step in range(5, 300):
            for substep in range(2):
                self.physics_step(np.zeros(7))
                pairs = [{int(c.geom1), int(c.geom2)} for c in self.data.contact[:self.data.ncon]]
                final_in_cup = bool(self.in_cup(pairs))
                table_contact = {ball, table.id} in pairs
                if table_contact and not previous_table_contact and first_in_step < 0:
                    table_bounces += 1
                previous_table_contact = table_contact
                if final_in_cup and first_in_step < 0:
                    first_in_step, first_in_substep = control_step, substep
                ever_in_cup |= final_in_cup
                center = self.data.geom(table.id).xpos
                position = self.data.qpos[7:10]
                outside = bool(np.any(np.abs(position[:2] - center[:2]) > table.size[:2] + .049))
                below = position[2] <= center[2] + table.size[2] + .049
                entered_airspace |= not outside or not below
                ground_contact = {ball, ground} in pairs
                if not off_table and (ground_contact or (entered_airspace and outside and below)):
                    off_table = True
                    first_off_step, first_off_substep = control_step, substep
                    offsets = position[:2] - center[:2]
                    axis = int(np.argmax(np.abs(offsets) - (table.size[:2] + .049)))
                    off_direction = ('+' if offsets[axis] >= 0 else '-') + ('x' if axis == 0 else 'y')
                    if ground_contact and not outside:
                        off_direction = 'ground_within_xy'
                contact_history.append(final_in_cup)
            qpos.append(self.data.qpos.copy())
            qvel.append(self.data.qvel.copy())
        return dict(qpos=np.stack(qpos), qvel=np.stack(qvel), controls=released['controls'],
                    max_locked_joint_drift_rad=released['max_locked_joint_drift_rad'],
                    final_in_cup=final_in_cup, ever_in_cup=bool(ever_in_cup),
                    off_table=bool(off_table), in_cup_substeps=np.array(contact_history),
                    first_in_cup_control_step=first_in_step, first_in_cup_physics_substep=first_in_substep,
                    table_bounces_before_cup=table_bounces if ever_in_cup else -1,
                    first_off_table_control_step=first_off_step, first_off_table_physics_substep=first_off_substep,
                    off_table_direction=off_direction)


def render_qpos(history):
    """Return the arm to its start pose for display only, over frames 6 through 10."""
    rendered = np.asarray(history).copy()
    for index in range(6, len(rendered)):
        alpha = min((index - 5) / 5., 1.)
        rendered[index, :7] = (1. - alpha) * rendered[5, :7] + alpha * START_QPOS
    return rendered
