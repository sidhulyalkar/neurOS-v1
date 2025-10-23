"""
NeuroFM-X Multi-Task Training Model.

This derived class from NeuroFMXComplete modifies the forward pass to
simultaneously execute all enabled task heads, returning a dictionary
of task predictions for multi-task loss calculation in the Trainer.
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use relative imports as before
from .neurofmx_complete import NeuroFMXComplete

class NeuroFMXMultiTask(NeuroFMXComplete):
    """
    Derived class from NeuroFMXComplete to enable simultaneous output
    from all active task heads for multi-task training/evaluation.
    """
    
    def __init__(self, **kwargs):
        # Initialize the base NeuroFMXComplete class (which now initializes tokenizer)
        super().__init__(**kwargs) 

    def forward(
        self,
        tokens_raw: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
        unit_indices: Optional[torch.Tensor] = None,
        session_id: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass. If task='multi-task', returns a dictionary of all
        enabled task predictions. Otherwise, runs a single task.
        """
        
        # --- 1. Backbone and Aggregation Flow (Explicit Core Logic) ---

        # 1. Tokenization: (B, S, N) -> (B, S, d_model=128)
        x, unit_mask = self.tokenizer(tokens_raw) 
        
        # 2. Mamba Backbone: (B, S, d_model) -> (B, S, d_model)
        # This executes the multi-rate logic and returns the 128-dim output.
        backbone_output = self.backbone(x, unit_mask)
        
        # 3. Perceiver-IO Fusion: (B, S, d_model) -> (B, N_latents, d_latent=512)
        latents = self.fusion(backbone_output, unit_mask)
        
        # 4. PopT Aggregator (if enabled)
        # NOTE: Assumes self.popt is an instance of PopTWithLatents
        if self.popt is not None:
            latents = self.popt(
                latents, 
                unit_indices=unit_indices, 
                padding_mask=None # Padding mask usually handled by Perceiver output in this flow
            )

        # 5. Apply Adapters 
        if self.unit_id_adapter is not None and unit_indices is not None:
            latents = self.unit_id_adapter(latents, unit_indices)

        if self.session_stitcher is not None and session_id is not None:
            latents = self.session_stitcher(latents, session_id)
            
        # 6. Final Head Input Pooling: (Aggregate N_latents) -> (B, D_latent=512)
        head_input = latents.mean(dim=1) 

        # --- 7. Task Head Execution ---

        if task == 'multi-task':
            # Execute all enabled heads
            results = {}
            for task_name in self.heads.get_available_tasks():
                results[task_name] = self.heads(head_input, task_name)
            return results
        
        elif task in self.heads.get_available_tasks():
            # Single task execution
            return self.heads(head_input, task)
        
        # Default: return the pooled latent representation
        return head_input