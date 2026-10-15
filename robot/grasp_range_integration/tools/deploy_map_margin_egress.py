"""Audited transit egress deployment; does not start robot motion."""
from grasp_range_integration.tools import deploy_approach_navigation as deploy

deploy.DEPLOYMENT_NAME='map_margin_egress'
deploy.FILES={
 'phase1/relationship_find_only_entry.py':('ad9d1fbde9cdf73a2991176b23a88d28edd666fbacbc070a57061a1623ca84d2',),
 'phase1/map_margin_egress.py':(None,),
 'tests/test_phase1_map_margin_egress.py':(None,),
 'tests/fixtures/map_margin_stop_20261002.json':(None,),
}
deploy.TESTS=('test_phase1_map_margin_egress','test_phase1_rotation_guard',
 'test_phase1_glass_boundary_guard','test_phase1_navigation_avoidance',
 'test_phase1_midrun_view_change','test_phase1_approach_navigation',
 'test_phase1_fresh_motion_feedback','test_phase1_body_occlusion')

if __name__=='__main__':deploy.main()
