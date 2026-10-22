"""Control-aligned four-panel videos for visual world model planning."""
from pathlib import Path
import numpy as np
import torch

@torch.inference_mode()
def planning_video(path, model, initial, actions, recorded, planned, goal, *, frameskip=5, fps=25):
    """Open-loop imagination from the initial observation and executed controls."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import get_data_path
    import imageio_ffmpeg
    imagined = None
    if model.decoder is not None:
        latent, _ = model.rollout(initial, actions)
        decoded, _ = model.decode_obs(latent)
        imagined = ((decoded['visual'][0] + 1) / 2).clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    font = ImageFont.truetype(str(Path(get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), 16)
    writer = imageio_ffmpeg.write_frames(str(path), (896, 284), fps=fps, macro_block_size=1, output_params=['-crf', '18'])
    writer.send(None)
    for step, frame in enumerate(planned):
        canvas = Image.new('RGB', (896, 284), 'white')
        draw = ImageDraw.Draw(canvas)
        panels = [('GT recorded', recorded[min(step, len(recorded)-1)], f't={min(step,len(recorded)-1)}' + (' (held)' if step >= len(recorded) else '')),
                  ('Planned Env', frame, f't={step}'),
                  ('WM imagination', np.zeros_like(frame) if imagined is None else imagined[min(step // frameskip, len(imagined)-1)],
                   'unavailable: no decoder' if imagined is None else f'token t={min(step//frameskip,len(imagined)-1)*frameskip}' + (' (held)' if step % frameskip or step//frameskip >= len(imagined) else '')),
                  ('Fixed goal', goal, 'fixed target')]
        for column, (title, pixels, label) in enumerate(panels):
            canvas.paste(Image.fromarray(pixels), (column*224, 60))
            draw.text((column*224+6, 6), title, fill='black', font=font)
            draw.text((column*224+6, 31), label, fill='black', font=font)
            if column:
                draw.line((column*224, 0, column*224, 284), fill='red', width=2)
        writer.send(np.array(canvas))
    writer.close()


@torch.inference_mode()
def replay_planning_video(path, model, collection, normalization, planning_run, target_manifest, environment):
    """Render the first Vanilla/TSWM result with its decoder and saved selected controls.

    Candidate search and success metrics are not rerun. Reuse canonical target
    readers and action expansion; imagination is open loop from the original
    initial observation, even when the saved controls came from MPC feedback.
    """
    import json
    from datasets.common import preprocess_rgb
    path = Path(path).resolve()
    path.relative_to((Path(__file__).parents[1] / 'logs').resolve())
    if path.exists():
        raise FileExistsError(path)
    planning_run = Path(planning_run)
    target = json.loads((planning_run / '0_result_planning.json').read_text())['per_target'][0]
    with np.load(planning_run / 'target_000.npz') as trace:
        planned = trace['rgb']
        states = trace['state'] if environment in ('pusht', 'reacher') else None
    if environment == 'pusht':
        from planning.targets import read_planning_target, read_simulator_parameters
        from datasets.pusht import pusht_observation
        from env.pusht import PushTEnv
        sample = read_planning_target(collection, target_manifest, 0, 224, normalization)
        physical = read_simulator_parameters(collection, sample)
        replay_rgb, _ = PushTEnv(render_size=224).rollout(
            sample['state'][0].numpy(), sample['replay_absolute_actions'].numpy(), **physical)
        initial, _ = pusht_observation(planned[:1], torch.from_numpy(states[:1, :5]), 224, normalization)
        recorded = collection.group(sample['shard_path'])['data/img'][
            int(sample['shard_frames'][0]):int(sample['shard_frames'][1]) + 1]
        goal = replay_rgb[-1]
        actions = torch.tensor(target['actions'])[None]
    else:
        episode = collection.read_episode(target['source_episode'])
        recorded = episode['rgb'][target['current_row']:]
        goal = episode['rgb'][target['goal_row']]
        initial = {'visual': preprocess_rgb(recorded[:1], 224)}
        if environment == 'reacher':
            from datasets.reacher import reacher_observation
            initial, _ = reacher_observation(planned[:1], torch.from_numpy(states[:1]), 224, normalization)
            actions = torch.tensor(target['actions'])[None]
        else:
            raw = target['action_xy'] if environment == 'fetch_slide' else target['selected_action']
            width = len(raw)
            proposal = (torch.tensor(raw) - normalization['action_mean'][:width]) / normalization['action_std'][:width]
            if environment == 'fetch_slide':
                from planning.fetch_slide import fetch_execution
                actions = fetch_execution(proposal[None, None], normalization, target['wm_horizon_tokens'])
            else:
                from planning.beerpong import beerpong_execution
                actions = beerpong_execution(proposal[None, None], normalization)
    device = next(model.parameters()).device
    initial = {key: value[None].to(device) for key, value in initial.items()}
    frameskip = 15 if environment == 'beerpong' else 5
    planning_video(path, model, initial, actions.to(device), recorded, planned, goal,
                   frameskip=frameskip, fps=100 if environment == 'beerpong' else 10)
    return {'executed_controls': len(planned) - 1, 'wm_action_groups': actions.shape[1],
            'frameskip': frameskip, 'candidate_search_rerun': False}
