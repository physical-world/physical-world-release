"""Bound PushT T-body initialization and offscreen rendering using Pymunk.

Matches the paper's use_ori_inertia=False geometry, reset and PD control.
Returns physical states and images; task metrics belong to evaluation.
"""

import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor


def sample_t_cog(seed):
    """Reproduce the released area-weighted CoG draw without rounding to float32."""
    rng = np.random.default_rng(int(seed))
    if int(rng.choice(2, p=(4 / 7, 3 / 7))) == 0:
        return np.array([rng.uniform(-60, 60), rng.uniform(0, 30)])
    return np.array([rng.uniform(-15, 15), rng.uniform(30, 120)])


def _light_color(color):
    values = np.minimum(1.2 * np.float32(color), np.float32([255]))
    return SpaceDebugColor(*values)


class PushTDrawOptions(pymunk.pygame_util.DrawOptions):
    """Paper colors: lighter interiors, outlined T, and no agent angle spoke.

    The pinned Pymunk implementation already supplies identical wall/edge
    drawing. Override only the two styles that differ from its defaults.
    """

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        center = (round(pos.x), round(pos.y))
        pygame.draw.circle(self.surface, fill_color.as_int(), center, round(radius))
        pygame.draw.circle(self.surface, _light_color(fill_color).as_int(), center, round(radius - 4))

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        super().draw_polygon(verts, 2, fill_color, _light_color(fill_color))


class PushTEnv:
    """One CPU simulator with explicit reset state and physical CoG metadata.

    Model preprocessing stays in datasets.pusht. In particular, physical CoG
    never enters Vanilla proprio. Explicit reset states require no random pose
    draw, so the reference rollout seed has no effect on this reset path.
    """

    def __init__(self, render_size=224):
        self.render_size = render_size
        self.sim_hz = 100
        self.control_hz = 10
        self.k_p, self.k_v = 100, 20
        self.goal_pose = np.array([256, 256, np.pi / 4])

    def reset(self, state, cog_seed, cog_local_normed):
        """Return post-settle raw RGB and 7D physical state; rebuild on every reset.

        The bound Vanilla input has zero agent velocity. Preserve the reference
        0.01-second reset step, including any contact-induced state adjustment.
        """
        cog = sample_t_cog(cog_seed)
        cog32 = cog.astype(np.float32)
        normalized = np.array([cog32[0] / 60, cog32[1] / 60 - 1], dtype=np.float32)
        if not np.allclose(normalized, cog_local_normed, atol=1e-6):
            raise ValueError("CoG seed disagrees with recorded physical parameters")
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0
        corners = [(5, 506), (5, 5), (506, 5), (506, 506)]
        for start, stop in zip(corners, corners[1:] + corners[:1]):
            wall = pymunk.Segment(self.space.static_body, start, stop, 2)
            wall.color = pygame.Color("LightGray")
            self.space.add(wall)
        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.agent.position = (256, 400)
        agent_shape = pymunk.Circle(self.agent, 15)
        agent_shape.color = pygame.Color("RoyalBlue")
        self.space.add(self.agent, agent_shape)
        polygons = [
            [(-60, 30), (60, 30), (60, 0), (-60, 0)],
            [(-15, 30), (-15, 120), (15, 120), (15, 30)],
        ]
        inertia = sum(pymunk.moment_for_poly(1, vertices) for vertices in polygons)
        self.block = pymunk.Body(1, inertia)
        shapes = [pymunk.Poly(self.block, vertices) for vertices in polygons]
        for shape in shapes:
            shape.color = pygame.Color("LightSlateGray")
        self.block.center_of_gravity = (shapes[0].center_of_gravity + shapes[1].center_of_gravity) / 2
        self.block.position = (256, 300)
        self.block.angle = 0
        self.space.add(self.block, *shapes)
        self.block.center_of_gravity = tuple(cog)
        state = np.asarray(state).tolist()
        self.agent.velocity = state[5:7]
        self.agent.position = state[:2]
        # Rotation changes the geometric origin when CoG is off-origin.
        self.block.angle = state[4]
        self.block.position = state[2:4]
        self.space.step(1.0 / self.sim_hz)
        return self.render(), self.state()

    def step(self, absolute_action):
        """Execute one 0.1-second pixel cursor target, without clipping."""
        target = np.asarray(absolute_action)
        dt = 1.0 / self.sim_hz
        for _ in range(self.sim_hz // self.control_hz):
            acceleration = self.k_p * (target - self.agent.position) + self.k_v * (
                pymunk.Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * dt
            self.space.step(dt)
        return self.render(), self.state()

    def rollout(self, state, actions, cog_seed, cog_local_normed):
        """Reset and execute dense absolute controls; include the initial frame."""
        image, actual = self.reset(state, cog_seed, cog_local_normed)
        images, states = [image], [actual]
        for action in actions:
            image, actual = self.step(action)
            images.append(image)
            states.append(actual)
        return np.stack(images), np.stack(states)

    def state(self):
        """Raw physical state; angle wrapped to [0, 2*pi), velocity not masked."""
        return np.array(tuple(self.agent.position) + tuple(self.block.position)
                        + (self.block.angle % (2 * np.pi),) + tuple(self.agent.velocity), dtype=np.float32)

    def render(self):
        """Paper-style 512px canvas -> uint8 HWC RGB, without opening a window."""
        canvas = pygame.Surface((512, 512))
        canvas.fill((255, 255, 255))
        options = PushTDrawOptions(canvas)
        goal = pymunk.Body(1, pymunk.moment_for_box(1, (50, 100)))
        goal.position = self.goal_pose[:2].tolist()
        goal.angle = self.goal_pose[2]
        for shape in self.block.shapes:
            points = [pymunk.pygame_util.to_pygame(goal.local_to_world(v), canvas) for v in shape.get_vertices()]
            pygame.draw.polygon(canvas, pygame.Color("LightGreen"), points + [points[0]])
        self.space.debug_draw(options)
        rgb = np.array(pygame.surfarray.pixels3d(canvas)).transpose(1, 0, 2)
        return cv2.resize(rgb, (self.render_size, self.render_size))
