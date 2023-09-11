from labml import experiment
from labml.configs import option

from rcmha.model import AutoregressiveModel, TransformerXL, TransformerXLLayer
from rcmha.config import Configs



@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    from labml_nn.transformers.xl import RelativeMultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward
    m = AutoregressiveModel(c.n_tokens, c.d_model, TransformerXL(
        TransformerXLLayer(d_model=c.d_model,
                            self_attn=RelativeMultiHeadAttention(c.heads, c.d_model, c.dropout),
                            feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                            dropout_prob=c.dropout), c.n_layers))
    return m.to(c.device)


def main():
    params = {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1.,
                        'optimizer.optimizer': 'Noam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'train_loader': 'sequential_train_loader',
                        'valid_loader': 'sequential_valid_loader',

                        'seq_len': 2,
                        'mem_len': 32,
                        'epochs': 10,
                        'batch_size': 32,
                        'inner_iterations': 25,
                        }
    experiment.create(name="RCMHA1", comment='Eksperimen ke 1', disable_screen=True)
    conf = Configs()
    experiment.configs(conf, params)

    experiment.add_pytorch_models({'model': conf.model})
    with experiment.start():
        conf.run()

if __name__ == '__main__':
    main()