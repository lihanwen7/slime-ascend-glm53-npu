"""Run-only train-worker candidate; remove one source-verified extra pool release."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ENTRY_SHA = '8e8ed297fa215706174229acf9aa6416d3690c51a75b8801c973b35e05cc015b'
MEMORY_SHA = 'ed9177073fa526ab6536862cbd7c9a2c7858d8674e584ad1b9c241a45dd6c3af'
VERSIONS = {'torch': '2.10.0', 'torch-npu': '2.10.0.post7', 'torch-memory-saver': '0.0.8'}


def _make_disable(source, namespace, filename):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_TorchMemorySaverImpl')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'disable')
    matches = [n for n in ast.walk(fn) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == 'torch_npu._C._npu_releasePool']
    if len(matches) != 1 or ast.unparse(matches[0].value) != 'torch_npu._C._npu_releasePool(torch.npu.current_device(), pool.id)':
        raise RuntimeError('Expected exactly one known extra release statement')
    target = matches[0]

    class RemoveOne(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if node is target else self.generic_visit(node)

    fn = RemoveOne().visit(fn)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    isolated = dict(namespace)
    exec(compile(module, filename + ':runonly-no-extra-release', 'exec'), isolated)
    return isolated['disable']


def install():
    # This flag belongs only to the new owned train actors' runtime env.
    # No torch/framework import happens for any process without this exact flag.
    if os.environ.get('GLM53_TMS_POST7_RELEASE_COMPAT') != '1':
        return
    base = Path('/mnt/share/l00804615/glm5.3_flash_native')
    env = Path('/mnt/share/l00804615/conda/envs/lhw_glm_53_flash_native')
    if not Path.cwd().resolve().is_relative_to(base.resolve()):
        raise RuntimeError('TMS compatibility requires task-owned cwd')
    if json.loads((base / '.native-owner.json').read_text()).get('owner') != 'glm53-native-20261019':
        raise RuntimeError('TMS compatibility task owner mismatch')
    for name, expected in VERSIONS.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError('TMS compatibility version guard failed: ' + name)
    entry = Path(importlib.metadata.distribution('torch-memory-saver').locate_file('torch_memory_saver/entrypoint.py')).resolve()
    memory = Path(importlib.metadata.distribution('torch-npu').locate_file('torch_npu/npu/memory.py')).resolve()
    if not (entry.is_relative_to(env.resolve()) and memory.is_relative_to(env.resolve())):
        raise RuntimeError('TMS compatibility source prefix guard failed')
    source = entry.read_bytes()
    if hashlib.sha256(source).hexdigest() != ENTRY_SHA or hashlib.sha256(memory.read_bytes()).hexdigest() != MEMORY_SHA:
        raise RuntimeError('TMS compatibility exact source guard failed')
    from torch_memory_saver import entrypoint
    cls = entrypoint._TorchMemorySaverImpl
    if getattr(cls, '_glm53_post7_no_extra_release', False):
        return
    if Path(entrypoint.__file__).resolve() != entry:
        raise RuntimeError('TMS compatibility imported source mismatch')
    cls.disable = _make_disable(source.decode('utf-8'), entrypoint.__dict__, str(entry))
    cls._glm53_post7_no_extra_release = True
    os.write(2, ('GLM53_TMS_POST7_RELEASE_COMPAT ' + json.dumps({
        'pid': os.getpid(), 'entry_sha256': ENTRY_SHA, 'memory_sha256': MEMORY_SHA,
        'removed': 'one_explicit_release_only', 'versions': VERSIONS,
    }) + '\n').encode())
