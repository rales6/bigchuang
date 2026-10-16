import unittest
from tools.maintenance.public_hygiene import violations


class PublicHygieneTests(unittest.TestCase):
    def test_reject_private_assets_without_logging_contents(self):
        for name in ('maps/current/semantic_runtime.json', 'photo.jpg',
                     'docs/maintenance/source_manifest.json', 'local/.env',
                     'robot/grasp_range_integration/tests/fixtures/scan.json'):
            self.assertTrue(violations(name, b'private content'), name)
        report = str(violations('code.py', ('sk-'+'inventedcredential'*3).encode()))
        self.assertNotIn('inventedcredential', report)

    def test_allow_templates_loopback_and_documentation_network(self):
        self.assertEqual(violations('.env.example', b'PI_HOST=pi.example.invalid\nPI_SSH_PASSWORD=\n'), [])
        self.assertEqual(violations('demo.py', b'host="192.0.2.20"; bind="127.0.0.1"'), [])
        self.assertTrue(violations('accident.py', ('host="'+'172.'+'18.'+'1.'+'10"').encode()))
        self.assertEqual(violations('maps/current/README.md', b'Provision privately'), [])
