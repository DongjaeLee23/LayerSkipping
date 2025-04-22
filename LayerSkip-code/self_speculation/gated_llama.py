import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from llama_model_utils import _make_causal_mask, _expand_mask

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
        # Embedding
        hidden_states = self.model.embed_tokens(input_ids)

        # Compute position ids for rotary embeddings
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Compute 4D attention mask using your existing helpers
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

        causal_mask = _make_causal_mask(
            input_ids_shape=(batch_size, seq_length),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            past_key_values_length=0,
        )

        expanded_mask = _expand_mask(attention_mask, dtype=hidden_states.dtype, tgt_len=seq_length)

        attention_mask = causal_mask + expanded_mask  # Final shape: [batch, 1, tgt_len, src_len]

        # Iterate through layers with gating
        for i, layer in enumerate(self.model.layers):
            print(i)
            # Compute gate probabilities
            gate_prob = self.gates[i](hidden_states).unsqueeze(-1)  # [batch, 1, 1]

            # Layer forward
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False
            )
            layer_output = layer_outputs[0]  # Fallback if it's a tuple

            # Blend according to gate
            # Compute a gate from the mean hidden representation
            # Compute gate from mean hidden state over sequence
            gate_input = hidden_states.mean(dim=1)  # shape: [batch_size, hidden_dim]
            gate_prob = self.gates[i](gate_input)   # shape: [batch_size, 1]

            # Expand gate_prob from [batch, 1] → [batch, seq_len, hidden_dim]
            gate_prob = gate_prob.squeeze(-1)       # shape: [batch]
            gate_prob = gate_prob.unsqueeze(1).unsqueeze(2)  # shape: [batch, 1, 1]
            gate_prob = gate_prob.expand(-1, hidden_states.size(1), hidden_states.size(2))  # [batch, seq_len, hidden_dim]


            # Apply gated layer skipping
            hidden_states = (1 - gate_prob) * hidden_states + gate_prob * layer_output

        # Final normalization and lm head
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}
