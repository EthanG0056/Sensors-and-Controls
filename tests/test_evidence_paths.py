import hashlib
from pathlib import Path
import tempfile
import unittest
from simulation.evidence_paths import verify_saved_hash


class EvidencePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'simulation').mkdir()
        self.content = b'original movement test\n'
        self.digest = hashlib.sha256(self.content).hexdigest()
        (self.root/'simulation/turtlebot.py').write_bytes(b'teammate code\n')
        (self.root/'simulation/turtlebot_movement_test.py').write_bytes(self.content)

    def test_strict_default_rejects_different_original(self):
        with self.assertRaises(AssertionError):
            verify_saved_hash(self.root, 'simulation/turtlebot.py', self.digest)

    def test_exact_mapping_both_metadata_roots(self):
        for root, path in ((self.root,'simulation/turtlebot.py'),(self.root/'simulation','turtlebot.py')):
            checked = verify_saved_hash(root, path, self.digest, True)
            self.assertEqual(checked.read_bytes(), self.content)
        self.assertEqual((self.root/'simulation/turtlebot.py').read_bytes(), b'teammate code\n')

    def test_corrupt_or_missing_renamed_file_never_falls_back(self):
        target=self.root/'simulation/turtlebot_movement_test.py'
        (self.root/'simulation/turtlebot.py').write_bytes(self.content)
        target.write_bytes(b'changed')
        with self.assertRaises(AssertionError):
            verify_saved_hash(self.root,'simulation/turtlebot.py',self.digest,True)
        target.unlink()
        with self.assertRaises(AssertionError):
            verify_saved_hash(self.root,'simulation/turtlebot.py',self.digest,True)

    def test_other_paths_remain_strict(self):
        p=self.root/'simulation/other.py';p.write_bytes(b'changed')
        with self.assertRaises(AssertionError):
            verify_saved_hash(self.root,'simulation/other.py',self.digest,True)
        p.write_bytes(self.content)
        self.assertEqual(verify_saved_hash(self.root,'simulation/other.py',self.digest,True),p)


if __name__=='__main__': unittest.main()
