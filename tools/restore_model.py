"""Download verified release parts and reconstruct exact model bytes (stdlib only)."""
import argparse,hashlib,json,os,pathlib,urllib.request,time

def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def restore(item,root):
 target=(root/item['path']).resolve()
 if not target.is_relative_to(root.resolve()):raise ValueError('Unsafe output path')
 if target.exists():
  if target.stat().st_size==item['size'] and file_sha(target)==item['sha256']:
   print('Already verified:',item['path']);return
  raise FileExistsError('Refusing to overwrite a different existing file: '+str(target))
 target.parent.mkdir(parents=True,exist_ok=True)
 temporary=target.with_name(target.name+'.partial')
 total=hashlib.sha256();written=0
 try:
  with temporary.open('wb') as out:
   for i,part in enumerate(item['parts'],1):
    print(f"Downloading {item['key']} {i}/{len(item['parts'])}: {part['name']}",flush=True)
    offset=out.tell()
    for attempt in range(4):
     out.seek(offset);out.truncate();h=hashlib.sha256();received=0
     try:
      request=urllib.request.Request(part['url'],headers={'User-Agent':'gemma4-model-restore/1.0'})
      with urllib.request.urlopen(request,timeout=180) as response:
       while True:
        block=response.read(4*1024*1024)
        if not block:break
        received+=len(block)
        if received>part['size']:raise ValueError('Part exceeded declared size')
        h.update(block);out.write(block)
      if received!=part['size'] or h.hexdigest()!=part['sha256']:raise ValueError('Part hash/size mismatch')
      break
     except Exception:
      if attempt==3:raise
      time.sleep(2**attempt)
    written+=received
   out.flush();os.fsync(out.fileno())
  if written!=item['size'] or file_sha(temporary)!=item['sha256']:raise ValueError('Whole-file hash/size mismatch')
  os.replace(temporary,target)
  print('Verified:',target,flush=True)
 except Exception:
  temporary.unlink(missing_ok=True)
  raise

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--only',choices=['gguf','original','all'],default='gguf')
 parser.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
 parser.add_argument('--manifest',type=pathlib.Path)
 args=parser.parse_args();manifest=json.loads((args.manifest or args.root/'backup-manifest.json').read_text(encoding='utf-8'))
 for item in manifest['files']:
  if args.only in ('all',item['key']):restore(item,args.root)

if __name__=='__main__':main()
