"""Reconstructed export wrappers for retained legacy CASMI preparation code.

These wrappers were written in 2026-10-18, not recovered historical launchers.
They make the two missing serialization steps explicit. No training runs here.
Run from the recipe workspace with the byte-pinned companion sources present.
"""
import argparse
from pathlib import Path
import pickle
import numpy as np


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Only primitive self-generated structure metadata is accepted')


def export_flat(train_path, output_dir):
    """Exact retained speclib.load_flat transform; new explicit file writer."""
    import speclib
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    library = speclib.load_flat(str(train_path))
    for key, name in [('off', 'lib_off.npy'), ('allmz', 'lib_mz.npy'), ('allin', 'lib_in.npy')]:
        with (destination / name).open('xb') as stream:
            np.save(stream, library[key], allow_pickle=False)
    metadata = {key: value for key, value in library.items() if key not in ('off', 'allmz', 'allin')}
    with (destination / 'lib_meta.pkl').open('xb') as stream:
        pickle.dump(metadata, stream, protocol=4)
    return {'spectra': int(len(library['off']) - 1), 'peaks': int(len(library['allmz']))}


def select_fingerprints(packed, metadata, low=.005, high=.995):
    """The retained train_fp.SpecData frequency/column-order recipe, unchanged."""
    if not (type(metadata) is dict and set(metadata) == {'keys', 'smiles', 'nbits'}
            and type(metadata['keys']) is list and type(metadata['smiles']) is list
            and type(metadata['nbits']) is int and metadata['nbits'] == 10407):
        raise ValueError('Expected self-generated full10407 primitive structure metadata')
    if not (packed.dtype == np.uint8 and packed.ndim == 2
            and len(metadata['keys']) == len(metadata['smiles']) == len(packed)
            and packed.shape[1] == (metadata['nbits'] + 7) // 8):
        raise ValueError('Fingerprint/metadata shape mismatch')
    if not (all(type(x) is str and x for x in metadata['keys'])
            and all(type(x) is str and x for x in metadata['smiles'])
            and len(set(metadata['keys'])) == len(metadata['keys'])):
        raise ValueError('Invalid or duplicated structure identity metadata')
    full = np.unpackbits(packed, axis=1)[:, :metadata['nbits']]
    frequency = full.mean(0)
    bits = np.where((frequency >= low) & (frequency <= high))[0]
    selected = np.ascontiguousarray(full[:, bits])
    return bits, np.packbits(selected, axis=1), {
        'keys': metadata['keys'], 'smiles': metadata['smiles'], 'nbits': int(len(bits))}


def export_selected(data_dir, output_dir):
    """No held split is used: this reproduces the legacy all-structure selection."""
    source, destination = Path(data_dir), Path(output_dir)
    with (source / 'struct_fp_meta.pkl').open('rb') as stream:
        metadata = PrimitiveUnpickler(stream).load()
    packed = np.load(source / 'struct_fp.npy', allow_pickle=False)
    bits, selected, selected_metadata = select_fingerprints(packed, metadata)
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in [('fp_bits.npy', bits), ('fp_sel.npy', selected)]:
        with (destination / name).open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
    with (destination / 'fp_meta.pkl').open('xb') as stream:
        pickle.dump(selected_metadata, stream, protocol=4)
    return {'structures': int(len(packed)), 'selected_bits': int(len(bits)),
            'selection_scope': 'all legacy training structures, including legacy validation structures'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    flat = commands.add_parser('flat')
    flat.add_argument('--train', type=Path, required=True)
    flat.add_argument('--output', type=Path, required=True)
    selected = commands.add_parser('selected')
    selected.add_argument('--data', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_flat(args.train, args.output) if args.command == 'flat' else export_selected(args.data, args.output)
    import json
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
