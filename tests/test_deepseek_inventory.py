"""The handoff scanner must never copy a credential into its output."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.maintenance.inventory_deepseek import scan


class InventoryPrivacyTests(unittest.TestCase):
    def test_config_paths_and_presence_only(self):
        key = 'sk-' + 'inventedcredential' * 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deepseek_vision.json'
            path.write_text(json.dumps({'api_key': key}), encoding='utf-8')
            (Path(directory) / '.env').write_text('DEEPSEEK_API_KEY='+key+'\n', encoding='utf-8')
            result = scan([directory])
            self.assertNotIn(key, json.dumps(result))
            json_entry = next(row for row in result['credential_locations'] if row['path'] == str(path.absolute()))
            self.assertEqual(json_entry['fields'],
                             [{'field': 'api_key', 'configured': True}])
            self.assertTrue(any('DEEPSEEK_API_KEY' in row['environment_variable_names']
                                for row in result['credential_locations']))

    def test_database_is_read_only_and_reports_no_other_settings(self):
        key = 'private-' + 'inventedcredential' * 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.sqlite3'
            connection = sqlite3.connect(path)
            connection.execute('CREATE TABLE app_settings(key TEXT, value_json TEXT)')
            connection.execute('INSERT INTO app_settings VALUES (?,?)',
                ('api_credentials', json.dumps({'deepseek_api': key, 'gpt_api': 'unrelated-secret'})))
            connection.commit()
            connection.close()
            before = path.read_bytes()
            result = scan([directory])
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn(key, json.dumps(result))
            self.assertNotIn('unrelated-secret', json.dumps(result))
            self.assertEqual(result['databases'][0]['fields'],
                             [{'field': 'deepseek_api', 'configured': True}])


if __name__ == '__main__':
    unittest.main()
