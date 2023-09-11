import torch
from labml.configs import option
from labml import tracker
import neptune.new as neptune

from labml import monit
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.train_valid import BatchIndex, hook_model_outputs
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
import neptune.new as neptune

from .model import AutoregressiveModel
from .metric import Accuracy
import configparser

class Configs(NLPAutoRegressionConfigs):
    
    
    model: AutoregressiveModel


    d_model: int = 128
    heads: int = 8
    dropout: float = 0
    d_ff: int = 256
    n_layers: int = 6
    mem_len: int = 128
    memory = SimpleStateModule()
    acc = Accuracy()
    
    def init(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        print("==========")
        print(config.get('Credentials', 'project'))
        self._run = neptune.init_run(
            project=config.get('Credentials', 'project'),
            api_token=config.get('Credentials', 'api_token'),
        )

        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        hook_model_outputs(self.mode, self.model, 'model')
        self.state_modules = [self.accuracy, self.memory]
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self._run["parameters/params"] = pytorch_total_params
        self._run["parameters/d_model"] = self.d_model
        self._run["parameters/heads"] = self.heads
        self._run["parameters/dropout"] = self.dropout
        self._run["parameters/d_ff"] = self.d_ff
        self._run["parameters/n_layers"] = self.n_layers
        self._run["parameters/mem_len"] = self.mem_len
        
    def merge_memory(self, old_mem, new_mem):
        if self.mem_len == 0:
            return []
        if old_mem:
            mem = [torch.cat((m, x), dim=0) for m, x in zip(old_mem, new_mem)]
        else:
            mem = new_mem

        if len(mem[0]) > self.mem_len:
            mem = [m[-self.mem_len:] for m in mem]

        return mem

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with self.mode.update(is_log_activations=batch_idx.is_last):
            mem = self.memory.get()
            output, new_mem = self.model(data, mem)
            mem = self.merge_memory(mem, new_mem)
            self.memory.set(mem)

        loss = self.loss_func(output, target)
        perplexity  = torch.exp(loss)
        self._run["train/ppi"].log(perplexity)
        self._run["train/loss"].log(loss)

        self.acc(output, target)
        acc = self.acc.track()
        self._run["train/accuracy"].log(acc)

        if self.mode.is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
    def sample(self):
        prompt = self.prompt
        mem = []
        for i in monit.iterate('Sample', 25):
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            output, new_mem = self.model(data, mem)
            output = output.argmax(dim=-1).squeeze(1)
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            prompt = prompt[-1:]
            mem = self.merge_memory(mem, new_mem)

