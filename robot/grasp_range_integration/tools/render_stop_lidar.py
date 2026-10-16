"""Plot an archived stop scan without changing masks or contacting chassis."""
import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-scan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import paramiko
    client=paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('pi.example.invalid',username='dachuang',password=os.environ['PI_SSH_PASSWORD'],
                   look_for_keys=False,allow_agent=False,timeout=15)
    try:
        with client.open_sftp() as sftp:
            with sftp.open(args.remote_scan) as stream:
                data=json.load(stream)
    finally:
        client.close()
    frame,assessment=data['frame'],data['assessment']
    angles=np.asarray(frame['angles_rad'])
    distances=np.asarray(frame['distances_m'],dtype=float)
    valid=np.isfinite(angles)&np.isfinite(distances)&(distances>0)
    body=np.zeros(len(angles),dtype=bool)
    body[frame['legacy_mask_indices']+frame['installation_patch_indices']]=True
    bearing=(np.degrees(angles)+180)%360-180
    index=int(np.argmin(abs(bearing-float(assessment['min_bearing_deg']))))
    # Front is UP; left of car is LEFT. Exact sensor coordinates, not a
    # guessed room/global map, and retain the mask saved at incident time.
    horizontal=-distances*np.sin(angles)
    vertical=distances*np.cos(angles)
    plt.rcParams['font.family']='Microsoft YaHei'
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,6.8),layout='constrained')
    for ax in axes:
        ax.scatter(horizontal[valid&~body],vertical[valid&~body],s=18,c='#1682b4',label='当时未过滤的回波')
        ax.scatter(horizontal[valid&body],vertical[valid&body],s=23,c='#969da5',label='当时已过滤的车体回波')
        ax.scatter([horizontal[index]],[vertical[index]],s=95,c='#df3037',zorder=8,label='触发停车的近回波')
        ax.plot(0,0,'k+',ms=13,mew=2,label='雷达中心')
        ax.plot(0,-.14,'s',ms=6,c='#6958a5',label='代码中的车体中心（雷达后14cm）')
        ax.add_patch(plt.Circle((0,0),.30,fill=False,color='#df3037',ls='--',lw=1.2,label='当时0.30m旋转保护圈'))
        ax.set_aspect('equal');ax.grid(alpha=.22)
        ax.set_xlabel('← 小车左侧     横向距离（米）     小车右侧 →')
        ax.set_ylabel('后方 ← 纵向距离（米） → 前方')
    axes[0].set_title('完整扫描：全部原始点，未平滑、未补墙')
    extent=max(1,float(np.nanmax(distances[valid])))+.3
    axes[0].set_xlim(-extent,extent);axes[0].set_ylim(-extent,extent)
    axes[0].legend(loc='lower left',fontsize=8,framealpha=.95)
    axes[1].set_title('雷达附近放大：右侧略偏后的近回波')
    axes[1].set_xlim(-.37,.37);axes[1].set_ylim(-.37,.37)
    axes[1].annotate('停车触发点\n方向 -107°，距离 0.156m\n不在当时车体掩膜内',
        (horizontal[index],vertical[index]),xytext=(.02,.13),fontsize=10,
        arrowprops={'arrowstyle':'->','color':'#df3037','lw':1.5},color='#a91421',
        bbox={'boxstyle':'round,pad=.4','facecolor':'white','alpha':.95,'edgecolor':'#df3037'})
    axes[1].annotate('前方',xy=(-.30,.32),xytext=(-.30,.20),ha='center',fontsize=10,
                      arrowprops={'arrowstyle':'->','color':'black'})
    fig.suptitle('2026-09-27 16:24:44 终止前雷达原始扫描（clearance_005）\n'
                 '以雷达为原点、车头朝上；颜色表示当时程序分类，不代表已确认障碍来源',fontsize=14)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=170)
    print(destination.resolve())
    print(json.dumps({'trigger_sensor_xy_m':[float(vertical[index]),float(-horizontal[index])],
                      'range_m':float(distances[index]),'bearing_deg':float(bearing[index]),
                      'raw_points':int(np.count_nonzero(valid)),
                      'saved_body_mask_points':int(np.count_nonzero(body))},ensure_ascii=False))


if __name__=='__main__':
    main()
