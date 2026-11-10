"""Bounded-memory GGUF quantization using llama.cpp's Python quantizers.

Preserves model/tokenizer metadata and tensor order. Matrix weights use Q4_0;
token and per-layer embeddings use Q8_0. Vectors retain their original type.
"""
from pathlib import Path
import sys
import math
import numpy as np
sys.path.insert(0, str(Path.home()/'.unsloth/llama.cpp/gguf-py'))
import gguf

source = Path(sys.argv[1])
target = Path(sys.argv[2])
assert not target.exists(), 'Refusing to overwrite a model'
reader = gguf.GGUFReader(source, 'r')
writer = gguf.GGUFWriter(target, reader.fields['general.architecture'].contents())
for field in reader.fields.values():
    if field.name == 'general.architecture' or field.name.startswith('GGUF.'):
        continue
    kind = field.types[0]
    value = field.contents()
    if field.name == 'general.file_type':
        value = int(gguf.LlamaFileType.MOSTLY_Q4_0)
    writer.add_key_value(field.name, value, kind,
                         sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)

plans = []
for tensor in reader.tensors:
    shape = tuple(int(x) for x in reversed(tensor.shape))
    kind = tensor.tensor_type
    convert = kind == gguf.GGMLQuantizationType.BF16 and len(shape)>1 and shape[-1]%32==0
    if convert:
        kind = gguf.GGMLQuantizationType.Q8_0 if tensor.name in ('token_embd.weight','per_layer_token_embd.weight') else gguf.GGMLQuantizationType.Q4_0
    block, size = gguf.GGML_QUANT_SIZES[kind]
    nbytes = math.prod(shape)//block*size
    writer.add_tensor_info(tensor.name, shape, np.dtype('float32'), nbytes, raw_dtype=kind)
    plans.append((tensor,kind,convert,nbytes))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
writer.close()

with target.open('ab') as output:
    output.write(b'\x00'*((-output.tell())%32))
    for index,(tensor,kind,convert,nbytes) in enumerate(plans):
        begin = output.tell()
        if convert:
            raw=tensor.data.view(np.uint16).reshape(-1)
            for offset in range(0,len(raw),1024*1024):
                values=(raw[offset:offset+1024*1024].astype(np.uint32)<<16).view(np.float32)
                gguf.quants.quantize(values,kind).tofile(output)
        else:
            raw=tensor.data.view(np.uint8).reshape(-1)
            for offset in range(0,len(raw),8*1024*1024):
                raw[offset:offset+8*1024*1024].tofile(output)
        assert output.tell()-begin == nbytes, tensor.name
        output.write(b'\x00'*((-nbytes)%32))
        if index%25==0 or nbytes>500_000_000:
            print(f'{index+1}/{len(plans)} {tensor.name} {kind.name} {output.tell()/1e9:.2f} GB',flush=True)

result = gguf.GGUFReader(target,'r')
assert len(result.tensors)==len(reader.tensors)
for before,after in zip(reader.tensors,result.tensors):
    assert before.name==after.name and np.array_equal(before.shape,after.shape)
print('Verified tensor names, shapes and count:',len(result.tensors),flush=True)
