"""Deploy near-region target / absolute range handoff fixes, without motion."""
from grasp_range_integration.tools import deploy_approach_navigation as deploy

deploy.DEPLOYMENT_NAME='search_handoff'
deploy.FILES={
 'phase1/observation_recovery.py':('d520eddb4e99e97d359fd70a1956bae3d1a3ab6fc5274aa230c443762cabd774',),
 'phase1/startup_active_recovery.py':('13ce2a721744e2208be3d9e000904f824479ddb0d9a98a93a063b33efb47e6f0',
                                    '40f4582b5f55e72ae4cd4b8afa8357c964ce4fbf1d074d23b53a9eef2a9945f6'),
 'phase1/search_handoff.py':(None,),
 'tests/test_phase1_search_handoff.py':(None,),
 'tests/test_phase1_midrun_view_change.py':('7db28508a49720622627f5bcdefb203e99a24e6dfd9c63b7d6fb1f36559a5258',
                                        'a053e8f5c274c9dfa6777211394616220994f82b394cba7f0760fa90d844891b'),
}
deploy.TESTS=('test_phase1_search_handoff','test_phase1_midrun_view_change',
 'test_phase1_green_target_review','test_phase1_verified_search',
 'test_phase1_observation_recovery','test_phase1_locked_target_recovery',
 'test_phase1_arrival_visibility','test_phase1_approach_navigation',
 'test_phase1_startup_active_recovery','test_phase1_startup_route_recovery',
 'test_visibility_route_planner','test_phase1_rotation_guard','test_phase1_body_occlusion',
 'test_phase1_glass_boundary_guard','test_phase1_map_margin_egress')

if __name__=='__main__':deploy.main()
