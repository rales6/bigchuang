"""Asynchronous transit observation; no motor API and no grasp authorization.

Fast colour tracks keep current bearings while a slower classifier reviews
immutable camera frames. Lost/ambiguous tracks never inherit an old identity.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time

from .search_handoff import near_region_geometry, ground_colour_boxes, boxes_overlap


def angle_error(a,b):
    return (a-b+math.pi)%(2*math.pi)-math.pi


def image_bearing(box,fov=82.):
    offset=((box[0]+box[2])/2-500)/500
    return -math.atan(offset*math.tan(math.radians(fov/2)))


@dataclass(frozen=True)
class LiveSettings:
    capture_interval_s: float = 1.5
    model_interval_s: float = 4.
    maximum_frame_age_s: float = 2.
    maximum_identity_age_s: float = 20.
    guidance_step_m: float = .30
    maximum_guidance_deg: float = 15.

    def __post_init__(self):
        values=list(vars(self).values())
        if (not all(isinstance(v,(int,float)) and not isinstance(v,bool)
                    and math.isfinite(v) and v>0 for v in values)
                or not 1 <= self.capture_interval_s <= 10
                or self.model_interval_s < self.capture_interval_s
                or not 1 <= self.maximum_frame_age_s <= 3
                or not 5 <= self.maximum_identity_age_s <= 30
                or not .20 <= self.guidance_step_m <= .35
                or not 3 <= self.maximum_guidance_deg <= 15):
            raise ValueError('invalid live observation settings')


def load_settings(path=None):
    source=Path(path or os.environ.get('PHASE1_LIVE_OBSERVATION_CONFIG',
                                      str(Path(__file__).with_name('live_observation_config.json'))))
    data=json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or set(data)-set(LiveSettings.__dataclass_fields__):
        raise ValueError('unknown live observation settings')
    return LiveSettings(**data)


def guided_segment_clear(known_map,start,goal,center,radius):
    """Authorize a geometric proposal only if the actual DIRECT segment fits.

    A* finding a detour to the endpoint cannot authorize cutting through it.
    The command still passes the executor's fresh lidar and glass output gates.
    """
    from .visibility_route_planner import work_area_path
    from .rotation_guard import map_rotation_evidence
    from raspberry_pi.mapping import Pose2D
    if known_map is None or not work_area_path(known_map,start,goal,center,radius):return False
    dx,dy=goal[0]-start[0],goal[1]-start[1];length=math.hypot(dx,dy)
    if length<=0 or not math.isfinite(length):return False
    fraction=max(0.,min(1.,((center[0]-start[0])*dx+(center[1]-start[1])*dy)/(length*length)))
    if math.dist((start[0]+fraction*dx,start[1]+fraction*dy),center)<radius:return False
    count=max(1,math.ceil(length/.025))
    for i in range(count+1):
        p=Pose2D(start[0]+dx*i/count,start[1]+dy*i/count,math.atan2(dy,dx))
        if not map_rotation_evidence(known_map,p,.44)['clear']:return False
    return True


class BearingTracks:
    """At most one unambiguous colour cue per eye carries a temporal ID."""
    def __init__(self):
        self.current={};self.serial=0

    def update(self,cues,pose,stamp):
        fresh={}
        for eye in ('left','right'):
            rows=[c for c in cues if c.get('camera_id')==eye and c.get('complete_in_frame')]
            if len(rows)!=1:
                continue  # Two similar objects: do not silently pick one.
            row=deepcopy(rows[0]);box=row['bbox_1000']
            bearing=pose[2]+image_bearing(box)
            old=self.current.get(eye)
            area=(box[2]-box[0])*(box[3]-box[1])
            continuous=(old is not None and 0 < stamp-old['stamp'] <= 3.
                and math.dist(pose[:2],old['pose'][:2]) <= .80
                and abs(angle_error(bearing,old['world_bearing'])) <= math.radians(18)
                and .4 <= area/old['area'] <= 2.5)
            if continuous:
                identity=old['track_id']
            else:
                self.serial+=1;identity=self.serial
            row.update(track_id=identity,pose=list(pose),stamp=stamp,area=area,world_bearing=bearing)
            fresh[eye]=row
        self.current=fresh
        return deepcopy(fresh)


def template_cue(path,template,eye):
    """For other classes, corroborate the model crop in each NEW image.

    Reject textureless/ambiguous patches; a lost template must be classified
    again. Matching supports a bearing only, never a physical object depth.
    """
    import cv2
    from PIL import Image
    import numpy as np
    with Image.open(path) as picture:rgb=np.asarray(picture.convert('RGB'))
    patch=template['pixels'];h,w=rgb.shape[:2]
    best=None
    for scale in (.8,1.,1.25):
        resized=cv2.resize(patch,None,fx=scale,fy=scale)
        ph,pw=resized.shape[:2]
        if min(ph,pw)<12 or ph>=h or pw>=w:continue
        scores=cv2.matchTemplate(rgb,resized,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        scores[max(0,y-ph):y+ph+1,max(0,x-pw):x+pw+1]=-1
        second=float(scores.max())
        if score<.90 or score-second<.04:continue
        row=dict(camera_id=eye,bbox_1000=[1000*x/w,1000*y/h,1000*(x+pw)/w,1000*(y+ph)/h],
                 complete_in_frame=x>=w*.015 and x+pw<=w*.985 and y>=h*.015 and y+ph<=h*.985,
                 confidence=score,source='classified_template_bearing')
        if row['complete_in_frame'] and (best is None or score>best['confidence']):best=row
    return best


def review_moving_frame(raw,target,images,cues,*,confirm,same_target,verify_green=None):
    """Bad/copied Qwen coordinates fall back to actual-pixel DS review."""
    from .observation_recovery import reject_copied_stereo_boxes
    checked=reject_copied_stereo_boxes(confirm(raw,target,images))
    colour=str(target.get('color') or '').lower()
    review={'accepted_cameras':[],'same_object':False,'reason':'not_needed'}
    if colour in {'red','green'} and str(target.get('class_name') or '').lower()=='block':
        checked=ground_colour_boxes(checked,cues)
    if colour=='green' and str(target.get('class_name') or '').lower()=='block' and not any(
            i.get('dual_camera_confirmed') for i in checked):
        from .colour_target_review import verify_green_cues
        pair={c:rows[0] for c in ('left','right')
              if len(rows:=[i for i in cues if i.get('camera_id')==c and i.get('complete_in_frame')])==1}
        review=(verify_green or verify_green_cues)(images,pair)
        additions=[]
        for c in review.get('accepted_cameras',[]):
            if c not in pair:
                continue
            additions.append(dict(camera_id=c,class_name='block',source='phase1_deepseek_green_block',
                confidence=review.get('camera_confidences',{}).get(c,review.get('minimum_confidence',0)),
                bbox_1000=list(pair[c]['bbox_1000']),attributes={'color':'green'},
                single_camera_match=True,complete_in_frame=True,confirmed_match=False,dual_camera_confirmed=False,
                search_evidence=dict(source_ok=True,class_ok=True,colour_ok=True,bbox_valid=True,
                    bbox_complete=True,pixel_geometry_source='phase1_green_visual_cue',deepseek_review=review)))
        if len(additions)==2 and review.get('same_object') and same_target(*additions):
            for i in additions:i.update(confirmed_match=True,dual_camera_confirmed=True,stereo_pair_id='live_ds_green_pair')
        checked.extend(additions)
    # A duplicate image pair is never stereo evidence, even with different names.
    if Path(images['left']).read_bytes()==Path(images['right']).read_bytes():
        for i in checked:
            i.update(single_camera_match=False,confirmed_match=False,dual_camera_confirmed=False)
            i.pop('stereo_pair_id',None)
    return checked,review


def classify_transit(mission,gpu,target,images,cues,context):
    """For independently visible green pixels use DS FIRST, not after Qwen's
    known copied-box failure. Other classes retain the deployed classifier.
    No model makes up coordinates or identifies a target from colour alone.
    """
    from copy import copy
    from .colour_target_review import verify_green_cues
    first_review=None
    def verify(images,pair):
        return verify_green_cues(images,pair,request_timeout_s=8.)
    if (str(target.get('color') or '').lower()=='green'
            and str(target.get('class_name') or '').lower()=='block' and cues):
        checked,first_review=review_moving_frame([],target,images,cues,
            confirm=mission._phase1_raw_confirm,same_target=mission._same_stereo_target,
            verify_green=verify)
        if any(i.get('single_camera_match') for i in checked):return checked,first_review
    client=copy(gpu)
    if hasattr(client,'timeout_s'):client.timeout_s=min(client.timeout_s,8.)
    try:
        value=mission._phase1_raw_perceive(client,images['left'],images['right'],context)
    except Exception:
        value={'detections':[]}
    return review_moving_frame(value.get('detections') or [],target,images,cues,
        confirm=mission._phase1_raw_confirm,same_target=mission._same_stereo_target,
        verify_green=(lambda *a:first_review) if first_review is not None else verify)


class LiveObserver:
    def __init__(self,mission,*,settings=None,clock=time.monotonic):
        self.mission=mission;self.settings=settings or LiveSettings();self.clock=clock
        self.lock=threading.RLock();self.camera=None;self.gpu=None;self.emit=lambda *a,**k:None
        self.tracks=BearingTracks();self.proof=None;self.latest=None;self.pose=None
        self.target={};self.instruction='';self.directory=None;self.closed=False
        self.capture_busy=False;self.model_busy=False;self.last_capture=-math.inf;self.last_model=-math.inf
        self.sequence=0;self.queue=[];self.handoff=False
        self.epoch=0
        self.guidance_origin=None
        self.templates={}

    def stopped_capture(self):
        with self.lock:
            self.epoch+=1;self.proof=None;self.handoff=False
            self.latest=None;self.tracks=BearingTracks()
            self.guidance_origin=None
            self.templates={}

    def bind(self,args,camera,gpu,directory,emit):
        decision=json.loads(Path(args.decision).read_text(encoding='utf-8'))
        self.camera=camera;self.gpu=gpu;self.target=decision.get('target') or {}
        self.instruction=decision.get('instruction','')
        self.directory=Path(directory)/'transit_observations';self.directory.mkdir(parents=True,exist_ok=True)
        self.emit=emit or (lambda *a,**k:None)

    def close(self):
        with self.lock:self.closed=True;self.proof=None;self.handoff=False

    def _event(self,name,**fields):
        with self.lock:
            if not self.closed:self.queue.append((name,fields))

    def _capture(self,route):
        model_due=False;review_started=False
        try:
            with self.lock:
                self.sequence+=1;number=self.sequence;before=list(self.pose);epoch=self.epoch
            path=self.directory/f'transit_{number:04d}_left.jpg'
            started=self.clock()
            info=self.mission._phase1_capture_unarchived(self.camera,path)
            right=info.get('camera_2_image')
            if not right:raise ValueError('live observation requires two actual camera frames')
            images={'left':path,'right':path.with_name(right)}
            from .observation_recovery import colour_visual_cues
            cues=colour_visual_cues(images,str(self.target.get('color') or '').lower())
            with self.lock:templates=dict(self.templates)
            generic_tracks={}
            for eye,template in templates.items():
                row=template_cue(images[eye],template,eye)
                if row is not None:
                    generic_tracks[eye]=dict(row,track_id=template['track_id'])
                else:
                    with self.lock:
                        if self.templates.get(eye) is template:self.templates.pop(eye,None)
            stamp=self.clock()
            with self.lock:
                if self.closed or epoch!=self.epoch:return
                pose=list(self.pose)
                stable=math.dist(before[:2],pose[:2])<=.35 and abs(angle_error(before[2],pose[2]))<=math.radians(25)
                tracks=self.tracks.update(cues if stable else [],pose,stamp)
                if stable:
                    for eye,row in generic_tracks.items():
                        tracks[eye]=dict(row,pose=pose,stamp=stamp,
                            world_bearing=pose[2]+image_bearing(row['bbox_1000']))
                frame=dict(sequence=number,epoch=epoch,stamp=stamp,capture_started_s=started,pose=pose,
                    images={k:str(v) for k,v in images.items()},tracks=tracks,stable_capture=stable)
                self.latest=frame
                model_due=stable and not self.model_busy and stamp-self.last_model>=self.settings.model_interval_s
                if model_due:self.model_busy=True;self.last_model=stamp
            (self.directory/f'transit_{number:04d}.json').write_text(json.dumps(frame,ensure_ascii=False,indent=2),encoding='utf-8')
            self._event('transit_photo',sequence=number,images=frame['images'],pixel_cameras=sorted(tracks),
                        robot_pose=pose,stable_capture=stable)
            if model_due:
                threading.Thread(target=self._review,args=(frame,route,cues),daemon=True).start()
                review_started=True
        except Exception as exc:
            self._event('transit_observation_unavailable',error=str(exc),navigation_continues=True)
        finally:
            with self.lock:
                self.capture_busy=False
                if model_due and not review_started:self.model_busy=False

    def _review(self,frame,route,cues):
        try:
            paths={k:Path(v) for k,v in frame['images'].items()}
            context=dict(observation_id=f'{self.directory.parent.name}_transit_{frame["sequence"]}',
                task_instruction=self.instruction,target=self.target,robot_pose=frame['pose'],
                search_phase='moving_target_observation',camera_grounding_contract='Independent boxes per camera; no coordinate copying.')
            checked,review=classify_transit(self.mission,self.gpu,self.target,paths,cues,context)
            classified={};stereo=None
            templates={}
            for eye,track in frame['tracks'].items():
                matches=[i for i in checked if i.get('camera_id')==eye and i.get('single_camera_match')
                    and boxes_overlap(i.get('bbox_1000'),track['bbox_1000'])]
                if matches:
                    best=max(matches,key=lambda i:(bool(i.get('dual_camera_confirmed')),
                                                   float(i.get('confidence') or 0)))
                    classified[eye]=dict(track_id=track['track_id'],detection=best)
            if str(self.target.get('color') or '').lower() not in {'red','green'}:
                from PIL import Image
                import numpy as np
                for eye in ('left','right'):
                    rows=[i for i in checked if i.get('camera_id')==eye and i.get('single_camera_match')]
                    if len(rows)!=1:continue
                    with Image.open(paths[eye]) as picture:rgb=np.asarray(picture.convert('RGB'))
                    b=rows[0]['bbox_1000'];h,w=rgb.shape[:2]
                    patch=rgb[int(b[1]*h/1000):int(b[3]*h/1000),int(b[0]*w/1000):int(b[2]*w/1000)].copy()
                    if min(patch.shape[:2])<12 or float(patch.std())<8:continue
                    identity=f'template_{frame["sequence"]}_{eye}'
                    templates[eye]=dict(track_id=identity,pixels=patch)
                    classified[eye]=dict(track_id=identity,detection=rows[0])
            pairs={i.get('stereo_pair_id') for i in checked if i.get('dual_camera_confirmed')}
            for pair in pairs:
                if pair and all(classified.get(c,{}).get('detection',{}).get('stereo_pair_id')==pair for c in ('left','right')):
                    stereo=pair
            proof=dict(frame_sequence=frame['sequence'],frame_stamp=frame['stamp'],classified=classified,
                       stereo_pair_id=stereo,detections=checked,review=review)
            (self.directory/f'transit_{frame["sequence"]:04d}.review.json').write_text(
                json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
            with self.lock:
                if (not self.closed and frame['epoch']==self.epoch
                        and self.clock()-frame['stamp']<=self.settings.maximum_identity_age_s):
                    self.proof=proof
                    if templates:self.templates=templates
            self._event('transit_model_review',sequence=frame['sequence'],classified_cameras=sorted(classified),
                        stereo_candidate=bool(stereo),deepseek_review=review,task_success_allowed=False)
        except Exception as exc:
            self._event('transit_model_unavailable',error=str(exc),navigation_continues=True)
        finally:
            with self.lock:self.model_busy=False

    def select(self,pose,route,known_map,*,schedule=True):
        position=[pose.x_m,pose.y_m,pose.yaw_rad];now=self.clock()
        near=near_region_geometry(route,position)['near']
        with self.lock:
            self.pose=position
            events=self.queue;self.queue=[]
            if (schedule and near and not self.closed and self.camera is not None and not self.capture_busy
                    and not self.handoff and now-self.last_capture>=self.settings.capture_interval_s):
                self.capture_busy=True;self.last_capture=now
                threading.Thread(target=self._capture,args=(deepcopy(route),),daemon=True).start()
            frame=deepcopy(self.latest);proof=deepcopy(self.proof)
        for name,fields in events:self.emit(name,**fields)
        if self.handoff:return {'action':'stopped_recheck','reason':'early_stereo_recheck_pending'}
        if (not near or not frame or not proof or not 0<=now-frame['stamp']<=self.settings.maximum_frame_age_s
                or not 0<=now-proof['frame_stamp']<=self.settings.maximum_identity_age_s
                or math.dist(position[:2],frame['pose'][:2])>.35
                or abs(angle_error(position[2],frame['pose'][2]))>math.radians(20)):
            return {'action':'route'}
        valid={c:row for c,row in frame['tracks'].items()
               if proof['classified'].get(c,{}).get('track_id')==row['track_id']}
        if not valid:
            self.guidance_origin=None
            return {'action':'route'}
        if set(valid)=={'left','right'} and proof.get('stereo_pair_id'):
            # Drain navigation waypoints without driving; frozen executor then
            # stops, takes NEW images, checks tracker ROI and verifies handoff.
            self.handoff=True
            self.emit('transit_stereo_preemption',sequence=frame['sequence'],
                      robot_pose=position,reason='two_current_tracks_model_confirmed',
                      next_action='fresh_stopped_stereo_and_tracker_entry_check',task_success_allowed=False)
            return {'action':'stopped_recheck','reason':'early_stereo_candidate'}
        cue=min(valid.values(),key=lambda i:abs((i['bbox_1000'][0]+i['bbox_1000'][2])/2-500))
        bearing=angle_error(cue['world_bearing'],pose.yaw_rad)
        step=max(-math.radians(self.settings.maximum_guidance_deg),
                 min(math.radians(self.settings.maximum_guidance_deg),bearing))
        if abs(step)<math.radians(3):step=0.
        xy=(pose.x_m+self.settings.guidance_step_m*math.cos(pose.yaw_rad+step),
            pose.y_m+self.settings.guidance_step_m*math.sin(pose.yaw_rad+step))
        region=route['target_region']
        center=tuple(region['center_xy_m']);radius=float(region['radius_m'])+.80
        if not guided_segment_clear(known_map,position[:2],xy,center,radius):
            self.handoff=True
            return {'action':'stopped_recheck','reason':'single_target_needs_stationary_stereo_alignment'}
        if self.guidance_origin is None:self.guidance_origin=position[:2]
        if math.dist(position[:2],self.guidance_origin) >= .45:
            self.handoff=True
            return {'action':'stopped_recheck','reason':'guided_advance_completed_recheck_stereo'}
        return dict(action='guided_waypoint',waypoint=xy,heading_tolerance_deg=2.,
                    camera=cue['camera_id'],track_id=cue['track_id'],step_deg=math.degrees(step),
                    object_depth_estimated=False)


def install_live_observation(mission,*,settings=None):
    observer=LiveObserver(mission,settings=settings or load_settings())
    mission._phase1_live_observation=observer
    original_localize=mission._localize;original_run=mission.run
    def localize(*args,**kwargs):
        result=original_localize(*args,**kwargs)
        observer.bind(args[0],kwargs.get('camera'),kwargs.get('gpu'),
                      kwargs.get('visual_directory'),kwargs.get('event_callback'))
        return result
    mission._localize=localize
    def run(*args,**kwargs):
        try:return original_run(*args,**kwargs)
        finally:observer.close()
    mission.run=run
    return observer
