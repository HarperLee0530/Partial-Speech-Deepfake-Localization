"""Exercise archive extraction using the official string-label representation."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cnn_ps import manifests
from cnn_ps.io import save_json, sha256


class LabelArchiveTests(unittest.TestCase):
    def extract_fixture(self, root, values):
        archive = root / 'labels.tar.gz'
        ids = {split: [{'uid': split}] for split in ('train', 'dev', 'eval')}
        save_json(root / 'ids.json', ids)
        with tarfile.open(archive, 'w:gz') as tar:
            for split in ids:
                stream = io.BytesIO()
                np.save(stream, {split: values})
                payload = stream.getvalue()
                member = tarfile.TarInfo(f'database/segment_labels/{split}_seglab_0.02.npy')
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
        with patch.object(manifests, 'LABEL_ARCHIVE_SHA256', sha256(archive)):
            manifests.extract_labels(archive, root / 'ids.json', root / 'out')

    def test_unicode_and_numeric_binary_labels_are_inverted_exactly(self):
        for values in (np.array(['1', '0', '1'], dtype='<U21'),
                       np.array([1, 0, 1], dtype=np.uint8)):
            with self.subTest(dtype=values.dtype), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.extract_fixture(root, values)
                for split in ('train', 'dev', 'eval'):
                    with np.load(root / 'out' / f'{split}.npz') as labels:
                        np.testing.assert_array_equal(labels[split], [0, 1, 0])
                        self.assertEqual(labels[split].dtype, np.uint8)
                self.assertTrue((root / 'out/provenance.json').is_file())

    def test_invalid_values_are_not_rounded_or_truncated_into_labels(self):
        for values in (['1', 'bad'], [0, .5], [0, 1 + 1e-8], [0, 2],
                       [np.nan], [], [[0, 1]]):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'train: invalid official fine labels'):
                    self.extract_fixture(Path(tmp), np.asarray(values))


if __name__ == '__main__':
    unittest.main()
