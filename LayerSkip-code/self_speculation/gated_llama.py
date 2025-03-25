import torch.nn as nn
from transformers import LlamaForCausalLM

class GatedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, 1),
                nn.Sigmoid()
            ) for _ in range(config.num_hidden_layers)
        ])
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Write your modified forward pass here with gate-weighted execution
        hidden_states = self.model.embed_tokens(input_ids)

        for i, layer in enumerate(self.model.layers):
            gate_prob = self.gates[i](hidden_states).unsqueeze(-1)  # shape: [batch, 1, 1]
            output, _ = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = (1 - gate_prob) * hidden_states + gate_prob * output

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}
