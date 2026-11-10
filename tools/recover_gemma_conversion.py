"""Convert recovered weights, avoiding an 11 GB temporary float32 embedding.

The official converter supplies all model metadata and all other tensors. The
unchanged BF16 per-layer embedding is copied directly from its safetensors bytes.
"""
from pathlib import Path
import json
import runpy
import struct
import sys
import numpy as np

LLAMA = Path.home()/'.unsloth/llama.cpp'
SOURCE = Path.home()/'.ollama/models/ollama-safetensors3119175340/model.safetensors'
sys.path.insert(0, str(LLAMA))
sys.path.insert(0, str(LLAMA/'gguf-py'))
import gguf

with SOURCE.open('rb') as stream:
    header_length = struct.unpack('<Q', stream.read(8))[0]
    header = json.loads(stream.read(header_length))
info = header['model.language_model.embed_tokens_per_layer.weight']
assert info['dtype'] == 'BF16'
embedding = np.memmap(SOURCE, mode='r', dtype=np.uint16,
                     offset=8+header_length+info['data_offsets'][0], shape=tuple(info['shape']))
original = gguf.GGUFWriter.add_tensor


def add_tensor(writer, name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
    if name == 'per_layer_token_embd.weight':
        assert raw_dtype == gguf.GGMLQuantizationType.BF16
        direct = embedding.view(np.uint8)
        assert direct.shape == tensor.shape and direct.nbytes == tensor.nbytes
        tensor = direct
    return original(writer, name, tensor, raw_shape, raw_dtype, tensor_endianess)


gguf.GGUFWriter.add_tensor = add_tensor
runpy.run_path(str(LLAMA/'convert_hf_to_gguf.py'), run_name='__main__')
