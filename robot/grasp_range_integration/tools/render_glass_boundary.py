"""Render configuration geometry only; no localization or motor IO."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.glass_boundary_guard import load_verified_boundaries, attach_boundaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--map',required=True)
    parser.add_argument('--map-yaml',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    d=load_verified_boundaries(args.config,args.map,args.map_yaml)
    m=KnownOccupancyMap.load(args.map,args.map_yaml)
    mask=attach_boundaries(m,d)
    extent=(m.origin_x_m,m.origin_x_m+m.width*m.resolution_m,
            m.origin_y_m,m.origin_y_m+m.height*m.resolution_m)
    fig,ax=plt.subplots(figsize=(8,8))
    ax.imshow(m.image,cmap='gray',vmin=0,vmax=255,origin='lower',extent=extent)
    rgba=np.zeros((*mask.shape,4));rgba[mask]=[1.,.15,.10,.26]
    ax.imshow(rgba,origin='lower',extent=extent)
    for a,b in d['segments_xy_m']:
        ax.plot([a[0],b[0]],[a[1],b[1]],color='red',lw=2,label='Continuous left boundary (map estimate)')
    ref=d.get('allowed_reference_xy_m')
    if ref is not None:
        ax.plot(*ref,'bo',label='Indoor-side reference (NOT current pose)')
    ax.set(xlim=(-4.8,3.5),ylim=(-4.9,3.2),xlabel='Map X (m)',ylabel='Map Y (m)',
           title='Whole left side + exterior: forbidden\nUser-designated policy, not a field measurement')
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=8)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)
    print(output.resolve())


if __name__=='__main__':main()
