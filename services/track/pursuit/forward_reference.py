"""Fixed body-forward projection calibrated with the operator's floor line.

Raw, non-mirrored views; toe-in mounting means two different slanted lines.
No runtime floor-line detection, stereo depth, or hardware access here.
"""
import math


def line_x(line, y, width, height):
    (x0,y0),(x1,y1)=line
    values=(x0,y0,x1,y1,y,width,height)
    if (not all(math.isfinite(float(v)) for v in values) or
            width<=0 or height<=0 or abs(y1-y0)<1e-6 or
            not all(0<=v<=1 for v in (x0,y0,x1,y1))):
        raise ValueError('车身中线参考坐标无效')
    return width*(x0+(y/height-y0)*(x1-x0)/(y1-y0))


def reference_error(config, camera, center, width, height, *, band=True):
    if camera not in ('left','right'):
        raise ValueError('无效相机名称')
    x,y=center
    if (not all(math.isfinite(float(v)) for v in (x,y)) or
            not 0<=x<=width or not 0<=y<=height):
        raise ValueError('目标坐标不在当前相机画面内')
    line=line_for(config,camera,width,height)
    error=x-line_x(line,y,width,height)
    if not band:
        return error
    ratio=float(config.forward_reference_band_ratio)
    if not math.isfinite(ratio) or not 0<=ratio<=0.15:
        raise ValueError('车身中线容差无效')
    return math.copysign(max(0,abs(error)-width*ratio),error)


def line_for(config,camera,width,height):
    if camera not in ('left','right'):
        raise ValueError('无效相机名称')
    if (width,height)==(640,480):
        return config.forward_left_line if camera=='left' else config.forward_right_line
    if (width,height)==(1280,720):
        return config.forward_left_line_1280 if camera=='left' else config.forward_right_line_1280
    raise ValueError(f'相机模式 {width}x{height} 未标定车身中线，禁止套用不同视场')


def reference_payload(config,width=640,height=480):
    return dict(enabled=config.forward_reference_enabled,
                lines={side:line_for(config,side,width,height) for side in ('left','right')},
                calibrated_modes=['640x480','1280x720'],
                frame_size=[width,height],
                band_ratio=config.forward_reference_band_ratio,
                source='operator_floor_line_20260926',coordinates='normalized_raw_view')
