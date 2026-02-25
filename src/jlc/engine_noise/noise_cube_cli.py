from __future__ import annotations
import argparse
import json
from .noise_cube_model import NoiseCube
from .noise_cube_volume import bounds_from_noise_cube, simulation_volume_from_noise_cube


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='noise-cube', description='Noise cube utilities')
    p.add_argument('--noise-cube', dest='fits', help='Path to noise cube FITS file')
    p.add_argument('--stats', action='store_true', help='Print basic bounds and volume stats (default)')
    p.add_argument('--as-json', action='store_true', help='Emit JSON instead of human text')
    args = p.parse_args(argv)

    if not args.fits:
        p.error('Please provide --noise-cube PATH')

    cube = NoiseCube.from_fits(args.fits)
    bounds = bounds_from_noise_cube(cube)
    vol = simulation_volume_from_noise_cube(cube)

    if args.as_json:
        print(json.dumps({'bounds': bounds, 'volume': vol}, indent=2))
    else:
        print('Noise cube bounds:')
        for k, v in bounds.items():
            print(f'  {k}: {v}')
        print('Volume stats:')
        for k, v in vol.items():
            if k == 'bounds':
                continue
            print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
