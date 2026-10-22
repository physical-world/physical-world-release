"""One frozen Context estimate per target, expressed in world-model physics input coordinates."""
import torch

from models.physics.estimator import convert_physics


class EstimatedPhysics:
    """Cache a target's fixed estimate across observations, feedback and goals."""
    def __init__(self,estimator,metadata,collection,records,read_context,physics_mean,physics_std):
        self.estimator = estimator.eval().requires_grad_(False)
        self.metadata = metadata
        self.collection = collection
        self.records = records
        self.read_context = read_context
        self.physics_mean = physics_mean
        self.physics_std = physics_std
        self.normalization = {key:torch.tensor(value) for key,value in metadata['input_normalization'].items()}
        self.cache = {}
        self.estimates = {}

    @torch.inference_mode()
    def __getitem__(self,episode):
        if episode not in self.cache:
            record = self.records[episode]
            dataset = self.metadata['dataset']
            context = self.read_context(self.collection,record,dataset['ctx_length'],dataset['ctx_frameskip'],
                                        self.metadata['img_size'],self.normalization)
            device = next(self.estimator.parameters()).device
            prediction = self.estimator({key:value[None].to(device) for key,value in context.items()})[0].cpu()
            condition = convert_physics(prediction,self.metadata['mean'],self.metadata['std'],
                                        self.physics_mean,self.physics_std)
            self.cache[episode] = condition
            self.estimates[episode] = dict(record,prediction=prediction.tolist(),physics_input=condition.tolist(),
                                           physical_value=(condition*self.physics_std+self.physics_mean).tolist(),
                                           ctx_length=dataset['ctx_length'],ctx_frameskip=dataset['ctx_frameskip'])
        return self.cache[episode]
