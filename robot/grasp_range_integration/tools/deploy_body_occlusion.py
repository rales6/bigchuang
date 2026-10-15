"""Deploy only the rear-body occlusion repair; never sends motion commands."""
from grasp_range_integration.tools import deploy_approach_navigation as deploy

deploy.DEPLOYMENT_NAME='body_occlusion'
deploy.FILES={
    'phase1/rotation_guard.py':('89782cb430e294d540da9f9f9dabd3aaa7971fa3670212f195b2b07025ed31d9',),
    'tests/test_phase1_body_occlusion.py':(None,),
    'tests/fixtures/body_occlusion_20261002.json':(None,),
}
deploy.TESTS=('test_phase1_body_occlusion','test_phase1_rotation_guard',
              'test_phase1_glass_boundary_guard','test_phase1_navigation_avoidance',
              'test_phase1_midrun_view_change','test_phase1_approach_navigation')

if __name__=='__main__':deploy.main()
